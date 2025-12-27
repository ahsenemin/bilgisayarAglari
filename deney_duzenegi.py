# =====================================================
# GEREKLİ KÜTÜPHANELER
# =====================================================
import argparse                  # Komut satırı argümanları için
import math                      # Matematiksel işlemler
import random                    # Rastgele sayı üretimi
import statistics                # İstatistiksel hesaplamalar (ortalama, std sapma)
import time                      # Zaman ölçümü
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
import concurrent.futures        # Paralel işlem (Multiprocessing) için
import pandas as pd              # Veri okuma işlemleri (CSV)

# =====================================================
# PROJE MODÜLLERİ
# =====================================================
try:
    from ag import G  # Ağ topolojisi (NetworkX Graph nesnesi)
    from genetik_proje import GenetikAlgoritma
    from karinca import ACORouting
    from q_learning import (
        greedy_path as ql_greedy_path,
        make_reward_fn,
        normalize_weights as ql_normalize_weights,
        q_learning as ql_train,
    )
except ImportError as e:
    print(f"HATA: Gerekli modüller eksik! ({e})")
    exit(1)

# =====================================================
# SABİTLER VE AYARLAR
# =====================================================
DEFAULT_WEIGHTS = [0.4, 0.4, 0.2]  # Varsayılan [Gecikme, Güvenilirlik, Kaynak] ağırlıkları
DEMAND_FILE = "BSM307_317_Guz2025_TermProject_DemandData.csv"
RELIABILITY_SCALE = 100.0

# Q-Learning performansını artırmak için komşuluk listesi önbelleğe alınır
QL_NEIGHBORS = {n: list(G.neighbors(n)) for n in G.nodes()}

# =====================================================
# VERİ YAPILARI
# =====================================================
@dataclass
class RunRecord:
    """Tek bir algoritma çalıştırma sonucunu tutan veri yapısı."""
    run_id: int
    success: bool
    reason: Optional[str]
    duration: float
    path: Optional[List[int]] = None
    metrics: Optional[Dict[str, float]] = None
    raw_score: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

# =====================================================
# 1. YARDIMCI FONKSİYONLAR
# =====================================================
def normalize_weight_list(weights: Sequence[float]) -> List[float]:
    """Ağırlıkları toplamı 1 olacak şekilde normalize eder."""
    total = sum(weights)
    if total <= 0:
        return list(DEFAULT_WEIGHTS)
    return [w / total for w in weights]

def load_demands(csv_path: str, count: int, offset: int) -> List[Tuple[int, int, float]]:
    """CSV dosyasından belirli sayıdaki talep (Kaynak, Hedef, Bant Genişliği) verisini okur."""
    try:
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df = df[["src", "dst", "demand_mbps"]].dropna()
        df = df.iloc[offset : offset + count]
        combos = []
        for _, row in df.iterrows():
            combos.append((int(row["src"]), int(row["dst"]), float(row["demand_mbps"])))
        return combos
    except FileNotFoundError:
        print(f"HATA: {csv_path} dosyası bulunamadı.")
        return []

def evaluate_path(graph, path: Optional[Sequence[int]], bandwidth_req: float, weights: Sequence[float]) -> RunRecord:
    """
    Bulunan yolun QoS metriklerini (Gecikme, Güvenilirlik, Maliyet) hesaplar ve
    bant genişliği kısıtlamasını sağlayıp sağlamadığını kontrol eder.
    """
    if not path or len(path) < 2:
        return RunRecord(run_id=0, success=False, reason="Algoritma geçerli bir rota döndürmedi.", duration=0.0)

    total_delay = 0.0
    log_reliability_cost = 0.0
    resource_cost = 0.0
    bottleneck = float("inf")

    # Düğüm (Node) Maliyetleri
    for idx, node in enumerate(path):
        node_data = graph.nodes[node]
        rel = float(node_data.get("reliability", 0.99))
        
        # Logaritmik güvenilirlik toplamı (Çarpım işlemini toplama çevirmek için)
        log_reliability_cost += -math.log(rel if rel > 1e-6 else 1e-6)
        
        # Başlangıç ve bitiş düğümleri hariç işlem gecikmesi eklenir
        if idx != 0 and idx != len(path) - 1:
            total_delay += float(node_data.get("processing_delay", 0.0))

    # Kenar (Edge) Maliyetleri
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return RunRecord(
                run_id=0, success=False, reason=f"Rota hatalı kenar içeriyor: ({u}, {v}) grafikte yok.", duration=0.0
            )
        edge = graph.edges[u, v]
        total_delay += float(edge.get("delay", 0.0))
        
        e_rel = edge.get("reliability", 0.99)
        log_reliability_cost += -math.log(e_rel if e_rel > 1e-6 else 1e-6)
        
        bw = float(edge.get("bandwidth", 1.0))
        resource_cost += 1000.0 / (bw if bw > 1.0 else 1.0)
        bottleneck = min(bottleneck, bw)

    # Nihai Skorlar
    reliability_value = math.exp(-log_reliability_cost)
    weighted_cost = (
        weights[0] * total_delay + 
        weights[1] * (log_reliability_cost * RELIABILITY_SCALE) + 
        weights[2] * resource_cost
    )

    success = bottleneck >= bandwidth_req
    reason = None
    if not success:
        reason = f"Minimum bant genişliği {bottleneck:.2f} Mbps < talep {bandwidth_req:.2f} Mbps"

    metrics = {
        "delay_ms": total_delay,
        "reliability": reliability_value,
        "resource_cost": resource_cost,
        "log_reliability_cost": log_reliability_cost,
        "bottleneck_mbps": bottleneck,
        "weighted_cost": weighted_cost,
        "hop_count": len(path) - 1,
    }

    return RunRecord(run_id=0, success=success, reason=reason, duration=0.0, path=list(path), metrics=metrics)

def summarize_runs(records: List[RunRecord]) -> Dict[str, Optional[float]]:
    """Bir algoritmanın tüm tekrarları için istatistiksel özet çıkarır."""
    success_records = [r for r in records if r.success and r.metrics]
    base = {
        "attempts": len(records),
        "success_count": len(success_records),
        "failure_count": len(records) - len(success_records),
        "avg_cost": None,
        "std_cost": None,
        "best_cost": None,
        "worst_cost": None,
        "best_path": None,
        "worst_path": None,
        "avg_time": statistics.mean(r.duration for r in records) if records else None,
        "best_time": min((r.duration for r in records), default=None),
        "worst_time": max((r.duration for r in records), default=None),
        "failures": [{"run": r.run_id, "reason": r.reason} for r in records if not r.success],
    }

    if success_records:
        costs = [r.metrics["weighted_cost"] for r in success_records]
        base["avg_cost"] = statistics.mean(costs)
        base["std_cost"] = statistics.stdev(costs) if len(costs) > 1 else 0.0
        best = min(success_records, key=lambda r: r.metrics["weighted_cost"])
        worst = max(success_records, key=lambda r: r.metrics["weighted_cost"])
        base["best_cost"] = best.metrics["weighted_cost"]
        base["worst_cost"] = worst.metrics["weighted_cost"]
        base["best_path"] = best.path
        base["worst_path"] = worst.path

    return base

# =====================================================
# 2. PARALEL İŞLEM MOTORU
# =====================================================
def run_single_experiment_batch(
    idx: int,
    combo: Tuple[int, int, float],
    algorithms: List[str],
    repeats: int,
    weights: List[float],
    seed: Optional[int],
    # Genetik Algoritma Parametreleri
    ga_pop: int, ga_generations: int, ga_mutation: float,
    # ACO Parametreleri
    aco_ants: int, aco_iterations: int, aco_alpha: float, aco_beta: float, aco_evap: float, aco_q: float,
    # Q-Learning Parametreleri
    ql_episodes: int, ql_alpha: float, ql_gamma: float, ql_max_steps: int, 
    ql_eps_start: float, ql_eps_end: float, ql_decay: int
):
    """
    Tek bir talep (Source-Dest-Bandwidth) kombinasyonu için seçilen algoritmaları çalıştırır.
    Bu fonksiyon ayrı bir işlem (process) içinde çalıştırılır.
    """
    
    # Deterministik sonuçlar için işlem bazlı rastgelelik tohumu (seed) atanır
    if seed is not None:
        random.seed(seed)
    
    source, dest, bandwidth = combo
    algo_records: Dict[str, List[RunRecord]] = {}
    summaries: Dict[str, Dict[str, Optional[float]]] = {}
    
    # --- 1. Genetik Algoritma (GA) ---
    if "ga" in algorithms:
        records: List[RunRecord] = []
        for run_idx in range(1, repeats + 1):
            try:
                ga = GenetikAlgoritma(
                    G, source, dest, pop_size=ga_pop, mutasyon_orani=ga_mutation, nesil=ga_generations, agirliklar=weights
                )
                best_path, raw_score, duration = ga.calistir()
                evaluation = evaluate_path(G, best_path, bandwidth, weights)
                evaluation.run_id = run_idx
                evaluation.duration = duration
                evaluation.raw_score = raw_score
                records.append(evaluation)
            except Exception as exc:
                records.append(RunRecord(run_id=run_idx, success=False, reason=f"Genetik algoritma hatası: {exc}", duration=0.0))
        
        algo_records["ga"] = records
        summaries["ga"] = summarize_runs(records)

    # --- 2. Karınca Kolonisi (ACO) ---
    if "aco" in algorithms:
        records: List[RunRecord] = []
        for run_idx in range(1, repeats + 1):
            try:
                start = time.perf_counter()
                aco = ACORouting(
                    G, source, dest, bandwidth, weights,
                    n_ants=aco_ants, n_iterations=aco_iterations,
                    alpha=aco_alpha, beta=aco_beta, evaporation=aco_evap, Q=aco_q
                )
                path, fitness, _ = aco.solve()
                duration = time.perf_counter() - start
                evaluation = evaluate_path(G, path, bandwidth, weights)
                evaluation.run_id = run_idx
                evaluation.duration = duration
                evaluation.raw_score = fitness
                records.append(evaluation)
            except Exception as exc:
                records.append(RunRecord(run_id=run_idx, success=False, reason=f"ACO çalıştırma hatası: {exc}", duration=0.0))
        
        algo_records["aco"] = records
        summaries["aco"] = summarize_runs(records)

    # --- 3. Q-Learning (RL) ---
    if "qlearning" in algorithms:
        records: List[RunRecord] = []
        q_weights = ql_normalize_weights(weights[0], weights[1], weights[2])
        reward_fn = make_reward_fn(q_weights, demand_mbps=bandwidth)

        for run_idx in range(1, repeats + 1):
            try:
                start = time.perf_counter()
                q_table = ql_train(
                    G, QL_NEIGHBORS,
                    start_node=source, goal_node=dest,
                    reward_fn=reward_fn, episodes=ql_episodes,
                    alpha=ql_alpha, gamma=ql_gamma,
                    epsilon_start=ql_eps_start, epsilon_end=ql_eps_end,
                    epsilon_decay_steps=ql_decay, max_steps_per_episode=ql_max_steps,
                    stochastic_fail=False,
                )
                duration = time.perf_counter() - start
                path = ql_greedy_path(q_table, QL_NEIGHBORS, source, dest)
                evaluation = evaluate_path(G, path, bandwidth, weights)
                evaluation.run_id = run_idx
                evaluation.duration = duration
                records.append(evaluation)
            except Exception as exc:
                records.append(RunRecord(run_id=run_idx, success=False, reason=f"Q-learning hatası: {exc}", duration=0.0))
        
        algo_records["qlearning"] = records
        summaries["qlearning"] = summarize_runs(records)

    return idx, combo, summaries, algo_records

# =====================================================
# 3. RAPORLAMA
# =====================================================
def build_report_section(
    case_idx: int,
    combo: Tuple[int, int, float],
    summaries: Dict[str, Dict[str, Optional[float]]],
    records: Dict[str, List[RunRecord]],
) -> List[str]:
    """Her deney kombinasyonu için metin tabanlı rapor bloğu oluşturur."""
    lines = []
    source, dest, bandwidth = combo
    lines.append(f"\n=== Deney {case_idx:02d}: S={source}, D={dest}, B={bandwidth:.2f} Mbps ===")
    
    for algo_name, summary in summaries.items():
        # Başarı Oranı ve Maliyet
        cost_str = f"{summary['avg_cost']:.4f}" if summary["avg_cost"] is not None else "---"
        lines.append(
            f"\n[{algo_name}] Başarı: {summary['success_count']}/{summary['attempts']} | "
            f"Avg Cost: {cost_str}"
        )
        
        # Zaman İstatistikleri
        if summary["avg_time"] is not None:
            lines.append(
                f"   Süre (sn) -> Ortalama: {summary['avg_time']:.4f}, En iyi: {summary['best_time']:.4f}, En kötü: {summary['worst_time']:.4f}"
            )
        
        # Maliyet İstatistikleri ve Rota
        if summary["avg_cost"] is not None:
            lines.append(
                f"   Maliyet -> Ortalama: {summary['avg_cost']:.4f}, Std: {summary['std_cost']:.4f}, "
                f"En iyi: {summary['best_cost']:.4f}, En kötü: {summary['worst_cost']:.4f}"
            )
            lines.append(f"   En iyi rota: {summary['best_path']}")
        
        # Hatalar
        if summary["failures"]:
            lines.append("   Başarısız denemeler:")
            for fail in summary["failures"]:
                lines.append(f"      · Tekrar {fail['run']}: {fail['reason']}")
        else:
            lines.append("   Başarısız deneme yok.")

        # Detaylı Log
        for rec in records[algo_name]:
            if rec.success and rec.metrics:
                lines.append(
                    f"      -> Tekrar {rec.run_id}: Delay={rec.metrics['delay_ms']:.2f} ms | "
                    f"Rel={rec.metrics['reliability']:.5f} | Bottleneck={rec.metrics['bottleneck_mbps']:.2f} Mbps | "
                    f"Maliyet={rec.metrics['weighted_cost']:.4f}"
                )
    return lines

# =====================================================
# 4. ANA PROGRAM (MAIN)
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="BSM307 ağ rotalama algoritmalarını otomatik deney düzeneğinde kıyaslar."
    )
    # Temel Ayarlar
    parser.add_argument("--repeats", type=int, default=5, help="Her kombinasyon için algoritma tekrar sayısı.")
    parser.add_argument("--demands", type=int, default=20, help="Demand dosyasından kaç adet (S,D,B) alınacak.")
    parser.add_argument("--demand-offset", type=int, default=0, help="Demand dosyasında başlanacak satır indexi.")
    parser.add_argument("--weights", type=float, nargs=3, default=DEFAULT_WEIGHTS, metavar=("W_DELAY", "W_REL", "W_RES"))
    parser.add_argument("--algorithms", nargs="+", default=["ga", "aco", "qlearning"], choices=["ga", "aco", "qlearning"])
    parser.add_argument("--output", type=str, default=None, help="Raporun kaydedileceği dosya adı.")
    parser.add_argument("--demand-file", type=str, default=DEMAND_FILE, help="Demand CSV dosyası yolu.")
    
    # Algoritma Hiperparametreleri (İsteğe Bağlı)
    parser.add_argument("--ga-pop", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=200)
    parser.add_argument("--ga-mutation", type=float, default=0.1)
    parser.add_argument("--aco-ants", type=int, default=25)
    parser.add_argument("--aco-iterations", type=int, default=50)
    parser.add_argument("--aco-alpha", type=float, default=1.0)
    parser.add_argument("--aco-beta", type=float, default=2.0)
    parser.add_argument("--aco-evap", type=float, default=0.5)
    parser.add_argument("--aco-q", type=float, default=100.0)
    parser.add_argument("--ql-episodes", type=int, default=2500)
    parser.add_argument("--ql-alpha", type=float, default=0.15)
    parser.add_argument("--ql-gamma", type=float, default=0.95)
    parser.add_argument("--ql-max-steps", type=int, default=200)
    parser.add_argument("--ql-epsilon-start", type=float, default=1.0)
    parser.add_argument("--ql-epsilon-end", type=float, default=0.05)
    parser.add_argument("--ql-epsilon-decay", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Tekrarlanabilirlik için ana seed belirleyici
    seed_generator = random.Random(args.seed) if args.seed is not None else random.Random()
    
    weights = normalize_weight_list(args.weights)
    combos = load_demands(args.demand_file, args.demands, args.demand_offset)
    if len(combos) < args.demands:
        print(f"⚠️  Demand dosyasında {args.demands} adet kayıt bulunamadı. {len(combos)} adet kombinasyon çalıştırılacak.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"deney_detay_{timestamp}.txt"

    # Rapor Başlığı
    overall_report: List[str] = []
    overall_report.append(f"Deney Tarihi: {timestamp}")
    overall_report.append(f"Kullanılan ağırlıklar (normalize): {weights}")
    overall_report.append(
        f"Algoritmalar: {', '.join(args.algorithms)} | Demand kayıt sayısı: {len(combos)} | Tekrar sayısı: {args.repeats}"
    )
    overall_report.append(f"Mod: Yüksek Performans (Parallel Execution)")

    print(f"🚀 Deney başlatılıyor... {len(combos)} kombinasyon, Çoklu İşlem (Multiprocessing) kullanılıyor.")
    
    # Görev listesini hazırlar
    tasks = []
    for idx, combo in enumerate(combos, start=1):
        # Her görev için ayrı bir rastgele seed üret
        task_seed = seed_generator.randint(0, 2**32 - 1) if args.seed is not None else None
        
        tasks.append((
            idx, combo, args.algorithms, args.repeats, weights, task_seed,
            args.ga_pop, args.ga_generations, args.ga_mutation,
            args.aco_ants, args.aco_iterations, args.aco_alpha, args.aco_beta, args.aco_evap, args.aco_q,
            args.ql_episodes, args.ql_alpha, args.ql_gamma, args.ql_max_steps, 
            args.ql_epsilon_start, args.ql_epsilon_end, args.ql_epsilon_decay
        ))

    overall_summaries: Dict[str, List[int]] = {algo: [] for algo in args.algorithms}
    results_buffer = []

    # Paralel İşlem Başlatma
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_experiment_batch, *task): task[0] for task in tasks}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                res_idx, res_combo, res_summaries, res_records = future.result()
                results_buffer.append((res_idx, res_combo, res_summaries, res_records))
                
                completed_count += 1
                print(f"✅ [{completed_count}/{len(combos)}] Tamamlandı: Deney {res_idx} (S={res_combo[0]}, D={res_combo[1]})")
                
            except Exception as e:
                print(f"❌ Hata oluştu Deney {idx}: {e}")

    # Sonuçları indeks sırasına göre sırala
    results_buffer.sort(key=lambda x: x[0])

    # Raporu Oluştur
    readable_names = {"ga": "Genetik Algoritma", "aco": "Karınca Kolonisi", "qlearning": "Q-Learning"}
    
    for idx, combo, summaries, algo_records in results_buffer:
        for algo in args.algorithms:
            if algo in summaries:
                overall_summaries[algo].append(summaries[algo]["success_count"])
        
        readable_summaries = {readable_names.get(k, k): v for k, v in summaries.items()}
        readable_records = {readable_names.get(k, k): v for k, v in algo_records.items()}
        overall_report.extend(build_report_section(idx, combo, readable_summaries, readable_records))

    # Özet Bölümü
    overall_report.append("\n=== Genel Başarı Özeti ===")
    for algo in args.algorithms:
        success_counts = overall_summaries[algo]
        total_cases = len(success_counts)
        fully_successful = sum(1 for c in success_counts if c > 0)
        overall_report.append(
            f"{algo.upper()}: {fully_successful}/{total_cases} kombinasyonda en az bir geçerli rota bulundu."
        )

    # Dosyaya Yazma
    report_text = "\n".join(overall_report)
    print("\nDeney raporu oluşturuldu. Kaydediliyor...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ Rapor: {output_path}")

if __name__ == "__main__":
    main()