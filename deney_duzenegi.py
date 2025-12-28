# =====================================================
# GEREKLİ KÜTÜPHANELER
# =====================================================
import argparse                  # Komut satırı argüman yönetimi
import math                      # Matematiksel işlemler
import random                    # Rastgele sayı üretimi
import statistics                # İstatistiksel hesaplamalar
import time                      # Zaman ölçümü
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
import concurrent.futures        # Paralel işlem (Multiprocessing)
import pandas as pd              # CSV veri okuma işlemleri

# =====================================================
# PROJE MODÜLLERİ VE KONTROLLERİ
# =====================================================
try:
    from ag import G  # Ağ topolojisi (NetworkX nesnesi)
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
DEFAULT_WEIGHTS = [0.4, 0.4, 0.2]  # [Gecikme, Güvenilirlik, Kaynak]
DEMAND_FILE = "BSM307_317_Guz2025_TermProject_DemandData.csv"
RELIABILITY_SCALE = 100.0

# Q-Learning için komşuluk listesi önbelleği (Performans için)
QL_NEIGHBORS = {n: list(G.neighbors(n)) for n in G.nodes()}

# =====================================================
# VERİ YAPILARI
# =====================================================
@dataclass

class RunRecord:
    run_id: int
    success: bool
    reason: Optional[str]
    duration: float
    seed: Optional[int] = None   
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
    Bulunan yolun QoS metriklerini hesaplar ve kısıtlamaları (Bant Genişliği) kontrol eder.
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
        log_reliability_cost += -math.log(rel if rel > 1e-6 else 1e-6)
        
        # Başlangıç ve bitiş hariç işlem gecikmesi
        if idx != 0 and idx != len(path) - 1:
            total_delay += float(node_data.get("processing_delay", 0.0))

    # Kenar (Edge) Maliyetleri
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return RunRecord(run_id=0, success=False, reason=f"Hatalı kenar: ({u}, {v}) grafikte yok.", duration=0.0)
        
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
        reason = f"Yetersiz bant genişliği ({bottleneck:.2f} < {bandwidth_req:.2f})"

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
    """Algoritma tekrarlarının istatistiksel özetini çıkarır."""
    success_records = [r for r in records if r.success and r.metrics]
    base = {
        "attempts": len(records),
        "success_count": len(success_records),
        "failure_count": len(records) - len(success_records),
        "avg_cost": None, "std_cost": None,
        "best_cost": None, "worst_cost": None,
        "best_path": None, "worst_path": None,
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
# 2. PARALEL İŞLEM MOTORU (WORKER)
# =====================================================

def run_single_experiment_batch(
    
    idx: int,
    combo: Tuple[int, int, float],
    algorithms: List[str],
    repeats: int,
    weights: List[float],
    seed: Optional[int],
    # GA Parametreleri
    ga_pop: int, ga_generations: int, ga_mutation: float,
    # ACO Parametreleri
    aco_ants: int, aco_iterations: int, aco_alpha: float, aco_beta: float, aco_evap: float, aco_q: float,
    # QL Parametreleri
    ql_episodes: int, ql_alpha: float, ql_gamma: float, ql_max_steps: int, 
    ql_eps_start: float, ql_eps_end: float, ql_decay: int,
):
    base_seed = seed
    """
    Tek bir talep (S, D, BW) için seçilen tüm algoritmaları belirtilen tekrar sayısı kadar çalıştırır.
    Bu fonksiyon Multiprocessing havuzunda ayrı bir işlem olarak çalışır.
    """
    source, dest, bandwidth = combo
    algo_records: Dict[str, List[RunRecord]] = {}
    summaries: Dict[str, Dict[str, Optional[float]]] = {}

    try:
        # seed
        if seed is not None:
            random.seed(seed)
            try:
                import numpy as np
                np.random.seed(seed)
            except Exception:
                pass

        # --- GA ---
        if "ga" in algorithms:
            records: List[RunRecord] = []
            for run_idx in range(1, repeats + 1):
                try:
                    run_seed = None
                    # (opsiyonel) her tekrar farklı seed
                    if base_seed is not None:
                        run_seed = (base_seed + 1009 * run_idx) % (2**32 - 1)
                        random.seed(run_seed)

                    ga = GenetikAlgoritma(G, source, dest,
                                          pop_size=ga_pop,
                                          mutasyon_orani=ga_mutation,
                                          nesil=ga_generations,
                                          agirliklar=weights,
                                          min_bw=bandwidth, seed=run_seed)
                    best_path, raw_score, duration = ga.calistir()
                    evaluation = evaluate_path(G, best_path, bandwidth, weights)
                    evaluation.run_id = run_idx
                    evaluation.duration = duration
                    evaluation.raw_score = raw_score
                    evaluation.seed = run_seed   
                    records.append(evaluation)
                except Exception as exc:
                    records.append(RunRecord(run_id=run_idx, success=False,
                                             reason=f"GA Hatası: {exc}", duration=0.0))
            algo_records["ga"] = records
            summaries["ga"] = summarize_runs(records)

        # --- ACO ---
        if "aco" in algorithms:
            records: List[RunRecord] = []
            for run_idx in range(1, repeats + 1):
                try:
                    run_seed = None
                    if base_seed is not None:
                        run_seed = (base_seed + 2003 * run_idx) % (2**32 - 1)
                        random.seed(run_seed)

                    start = time.perf_counter()
                    aco = ACORouting(G, source, dest, bandwidth, weights,
                                     n_ants=aco_ants, n_iterations=aco_iterations,
                                     alpha=aco_alpha, beta=aco_beta,
                                     evaporation=aco_evap, Q=aco_q)
                    path, fitness, _ = aco.solve()
                    duration = time.perf_counter() - start

                    evaluation = evaluate_path(G, path, bandwidth, weights)
                    evaluation.run_id = run_idx
                    evaluation.duration = duration
                    evaluation.raw_score = fitness
                    evaluation.seed = run_seed
                    records.append(evaluation)
                except Exception as exc:
                    records.append(RunRecord(run_id=run_idx, success=False,
                                             reason=f"ACO Hatası: {exc}", duration=0.0))
            algo_records["aco"] = records
            summaries["aco"] = summarize_runs(records)

        # --- QL ---
        if "qlearning" in algorithms:
            records: List[RunRecord] = []
            q_weights = ql_normalize_weights(weights[0], weights[1], weights[2])
            reward_fn = make_reward_fn(q_weights, demand_mbps=bandwidth)

            for run_idx in range(1, repeats + 1):
                try:
                    run_seed = None
                    if base_seed is not None:
                        run_seed = (base_seed + 3001 * run_idx) % (2**32 - 1)
                        random.seed(run_seed)
                        try:
                            import numpy as np
                            np.random.seed(run_seed)
                        except Exception:
                            pass

                    start = time.perf_counter()
                    q_table = ql_train(
                        G, QL_NEIGHBORS,
                        start_node=source, goal_node=dest,
                        reward_fn=reward_fn,
                        episodes=ql_episodes,
                        alpha=ql_alpha, gamma=ql_gamma,
                        epsilon_start=ql_eps_start, epsilon_end=ql_eps_end,
                        epsilon_decay_steps=ql_decay,
                        max_steps_per_episode=ql_max_steps,
                        stochastic_fail=False,
                    )
                    duration = time.perf_counter() - start

                    path = ql_greedy_path(q_table, QL_NEIGHBORS, source, dest)
                    evaluation = evaluate_path(G, path, bandwidth, weights)
                    evaluation.run_id = run_idx
                    evaluation.duration = duration
                    evaluation.seed = run_seed
                    records.append(evaluation)
                except Exception as exc:
                    records.append(RunRecord(run_id=run_idx, success=False,
                                             reason=f"QL Hatası: {exc}", duration=0.0))

            algo_records["qlearning"] = records
            summaries["qlearning"] = summarize_runs(records)

        # ✅ HER ZAMAN RETURN
        return idx, combo, summaries, algo_records

    except Exception as e:
        # ✅ Worker asla None dönmesin
        import traceback
        traceback.print_exc()
        summaries["_worker_error"] = {"avg_cost": None, "error": str(e)}
        return idx, combo, summaries, algo_records

# =====================================================
# 3. RAPORLAMA MODÜLÜ
# =====================================================
def build_report_section(
    case_idx: int,
    combo: Tuple[int, int, float],
    summaries: Dict[str, Dict[str, Optional[float]]],
    records: Dict[str, List[RunRecord]],
) -> List[str]:
    """Her deney durumu için metin tabanlı, detaylı rapor bloğu oluşturur."""
    lines = []
    source, dest, bandwidth = combo
    lines.append(f"\n=== Deney {case_idx:02d}: S={source}, D={dest}, B={bandwidth:.2f} Mbps ===")
    
    # Algoritmaları ortalama maliyete göre sırala (Kazananı belirlemek için)
    sorted_algos = []
    for name, data in summaries.items():
        cost = data.get("avg_cost")
        if cost is None: cost = float('inf')
        sorted_algos.append((name, data, cost))
    sorted_algos.sort(key=lambda x: x[2])

    for rank, (algo_name, summary, cost_val) in enumerate(sorted_algos, start=1):
        cost_str = f"{summary['avg_cost']:.4f}" if summary["avg_cost"] is not None else "---"
        winner_badge = " [🏆 KAZANAN]" if rank == 1 and summary["avg_cost"] is not None else ""
        
        lines.append(
            f"\n[{algo_name}]{winner_badge} Başarı: {summary['success_count']}/{summary['attempts']} | "
            f"Avg Cost: {cost_str}"
        )
        
        # Süre ve Rota Bilgileri
        if summary["avg_time"] is not None:
            lines.append(
                f"   Süre (sn) -> Ortalama: {summary['avg_time']:.4f}, En iyi: {summary['best_time']:.4f}, En kötü: {summary['worst_time']:.4f}"
            )
        if summary["avg_cost"] is not None:
            lines.append(
                f"   Maliyet -> Ortalama: {summary['avg_cost']:.4f}, Std: {summary['std_cost']:.4f}, "
                f"En iyi: {summary['best_cost']:.4f}, En kötü: {summary['worst_cost']:.4f}"
            )
            lines.append(f"   En iyi rota: {summary['best_path']}")
        
        # Hata ve Detaylı Çalışma Kayıtları
        if summary["failures"]:
            lines.append("   Başarısız denemeler:")
            for fail in summary["failures"]:
                lines.append(f"      · Tekrar {fail['run']}: {fail['reason']}")

        # Tekrarları maliyete göre sıralayıp listeleme
        if algo_name in records:
            valid_runs = [r for r in records[algo_name] if r.success and r.metrics]
            valid_runs.sort(key=lambda r: r.metrics['weighted_cost'])

            for i, rec in enumerate(valid_runs, start=1):
                seed_str = f", Seed={rec.seed}" if rec.seed is not None else ""
                lines.append(
                    f"      -> #{i} (Tekrar {rec.run_id}{seed_str}): "
                    f"Delay={rec.metrics['delay_ms']:.2f} ms | "
                    f"Rel={rec.metrics['reliability']:.5f} | "
                    f"Bottleneck={rec.metrics['bottleneck_mbps']:.2f} Mbps | "
                    f"Maliyet={rec.metrics['weighted_cost']:.4f}"
                )
    return lines

# =====================================================
# 4. ANA PROGRAM (MAIN)
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="BSM307 Rotalama Algoritmaları - Toplu Deney Düzeneği")
    
    # Temel Ayarlar
    parser.add_argument("--repeats", type=int, default=5, help="Her algoritma için tekrar sayısı")
    parser.add_argument("--demands", type=int, default=20, help="Çalıştırılacak talep (demand) sayısı")
    parser.add_argument("--demand-offset", type=int, default=0, help="CSV'de başlanacak satır")
    parser.add_argument("--weights", type=float, nargs=3, default=DEFAULT_WEIGHTS, help="Ağırlıklar: Delay Rel Resource")
    parser.add_argument("--algorithms", nargs="+", default=["ga", "aco", "qlearning"], choices=["ga", "aco", "qlearning"])
    parser.add_argument("--output", type=str, default=None, help="Çıktı rapor dosyası adı")
    parser.add_argument("--demand-file", type=str, default=DEMAND_FILE, help="Talep verisi CSV yolu")
    
    # Algoritma Hiperparametreleri
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
    parser.add_argument("--seed", type=int, default=None, help="Rastgelelik tohumu (Seed)")
    
    args = parser.parse_args()

    # Rastgelelik tohumunu ayarla
    seed_generator = random.Random(args.seed) if args.seed is not None else random.Random()
    weights = normalize_weight_list(args.weights)
    
    # Talepleri Yükle
    combos = load_demands(args.demand_file, args.demands, args.demand_offset)
    if len(combos) < args.demands:
        print(f"⚠️  Demand dosyasında {args.demands} adet kayıt bulunamadı. {len(combos)} adet çalıştırılacak.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"deney_detay_{timestamp}.txt"

    # Rapor Başlığını Oluştur
    overall_report: List[str] = []
    overall_report.append(f"Deney Tarihi: {timestamp}")
    overall_report.append(f"Kullanılan ağırlıklar (normalize): {weights}")
    overall_report.append(f"Algoritmalar: {', '.join(args.algorithms)} | Kayıt: {len(combos)} | Tekrar: {args.repeats}")
    overall_report.append(f"Mod: Yüksek Performans (Parallel Execution)")

    print(f"🚀 Deney başlatılıyor... {len(combos)} kombinasyon, Multiprocessing aktif.")
    
    # Görev listesini hazırla
    tasks = []
    for idx, combo in enumerate(combos, start=1):
        task_seed = seed_generator.randint(0, 2**32 - 1)
        tasks.append((
            idx, combo, args.algorithms, args.repeats, weights, task_seed,
            args.ga_pop, args.ga_generations, args.ga_mutation,
            args.aco_ants, args.aco_iterations, args.aco_alpha, args.aco_beta, args.aco_evap, args.aco_q,
            args.ql_episodes, args.ql_alpha, args.ql_gamma, args.ql_max_steps, 
            args.ql_epsilon_start, args.ql_epsilon_end, args.ql_epsilon_decay
        ))

    overall_summaries: Dict[str, List[int]] = {algo: [] for algo in args.algorithms}
    results_buffer = []

    # Paralel Çalıştırma Başlat
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
                import traceback
                print(f"❌ Hata oluştu Deney {idx}: {repr(e)}")
                traceback.print_exc()

    # Sonuçları sıraya diz
    results_buffer.sort(key=lambda x: x[0])

    readable_names = {"ga": "Genetik Algoritma", "aco": "Karınca Kolonisi", "qlearning": "Q-Learning"}
    
    # Raporu Derle
    for idx, combo, summaries, algo_records in results_buffer:
        for algo in args.algorithms:
            if algo in summaries:
                overall_summaries[algo].append(summaries[algo]["success_count"])
        
        readable_summaries = {readable_names.get(k, k): v for k, v in summaries.items()}
        readable_records = {readable_names.get(k, k): v for k, v in algo_records.items()}
        overall_report.extend(build_report_section(idx, combo, readable_summaries, readable_records))

    # Genel Özet Ekle
    overall_report.append("\n=== Genel Başarı Özeti ===")
    for algo in args.algorithms:
        success_counts = overall_summaries[algo]
        total_cases = len(success_counts)
        fully_successful = sum(1 for c in success_counts if c > 0)
        overall_report.append(
            f"{algo.upper()}: {fully_successful}/{total_cases} kombinasyonda en az bir geçerli rota bulundu."
        )

    # Dosyaya Kaydet
    report_text = "\n".join(overall_report)
    print("\nDeney raporu oluşturuldu. Kaydediliyor...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ Rapor: {output_path}")

if __name__ == "__main__":
    main()