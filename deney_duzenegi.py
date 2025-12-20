import argparse
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ag import G
from genetik_proje import GenetikAlgoritma
from karinca import ACORouting
from q_learning import (
    greedy_path as ql_greedy_path,
    make_reward_fn,
    normalize_weights as ql_normalize_weights,
    q_learning as ql_train,
)


DEFAULT_WEIGHTS = [0.4, 0.4, 0.2]
DEMAND_FILE = "BSM307_317_Guz2025_TermProject_DemandData.csv"
RELIABILITY_SCALE = 100.0
QL_NEIGHBORS = {n: list(G.neighbors(n)) for n in G.nodes()}


@dataclass
class RunRecord:
    run_id: int
    success: bool
    reason: Optional[str]
    duration: float
    path: Optional[List[int]] = None
    metrics: Optional[Dict[str, float]] = None
    raw_score: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


def normalize_weight_list(weights: Sequence[float]) -> List[float]:
    total = sum(weights)
    if total <= 0:
        return list(DEFAULT_WEIGHTS)
    return [w / total for w in weights]


def load_demands(csv_path: str, count: int, offset: int) -> List[Tuple[int, int, float]]:
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df = df[["src", "dst", "demand_mbps"]].dropna()
    df = df.iloc[offset : offset + count]
    combos = []
    for _, row in df.iterrows():
        combos.append((int(row["src"]), int(row["dst"]), float(row["demand_mbps"])))
    return combos


def evaluate_path(graph, path: Optional[Sequence[int]], bandwidth_req: float, weights: Sequence[float]) -> RunRecord:
    if not path or len(path) < 2:
        return RunRecord(run_id=0, success=False, reason="Algoritma geçerli bir rota döndürmedi.", duration=0.0)

    total_delay = 0.0
    log_reliability_cost = 0.0
    resource_cost = 0.0
    bottleneck = float("inf")

    for idx, node in enumerate(path):
        node_data = graph.nodes[node]
        rel = float(node_data.get("reliability", 0.99))
        log_reliability_cost += -math.log(max(rel, 1e-6))
        if idx != 0 and idx != len(path) - 1:
            total_delay += float(node_data.get("processing_delay", 0.0))

    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return RunRecord(
                run_id=0, success=False, reason=f"Rota hatalı kenar içeriyor: ({u}, {v}) grafikte yok.", duration=0.0
            )
        edge = graph.edges[u, v]
        total_delay += float(edge.get("delay", 0.0))
        log_reliability_cost += -math.log(max(edge.get("reliability", 0.99), 1e-6))
        bw = float(edge.get("bandwidth", 1.0))
        resource_cost += 1000.0 / max(bw, 1.0)
        bottleneck = min(bottleneck, bw)

    reliability_value = math.exp(-log_reliability_cost)
    weighted_cost = (
        weights[0] * total_delay + weights[1] * (log_reliability_cost * RELIABILITY_SCALE) + weights[2] * resource_cost
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


def run_genetic_algorithm(
    source: int,
    dest: int,
    bandwidth: float,
    repeats: int,
    weights: Sequence[float],
    pop_size: int,
    generations: int,
    mutation_rate: float,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    for run_idx in range(1, repeats + 1):
        try:
            ga = GenetikAlgoritma(
                G, source, dest, pop_size=pop_size, mutasyon_orani=mutation_rate, nesil=generations, agirliklar=weights
            )
            best_path, raw_score, duration = ga.calistir()
            evaluation = evaluate_path(G, best_path, bandwidth, weights)
            evaluation.run_id = run_idx
            evaluation.duration = duration
            evaluation.raw_score = raw_score
            records.append(evaluation)
        except Exception as exc:
            records.append(
                RunRecord(run_id=run_idx, success=False, reason=f"Genetik algoritma hatası: {exc}", duration=0.0)
            )
    return records


def run_aco(
    source: int,
    dest: int,
    bandwidth: float,
    repeats: int,
    weights: Sequence[float],
    ants: int,
    iterations: int,
    alpha: float,
    beta: float,
    evaporation: float,
    q_value: float,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    for run_idx in range(1, repeats + 1):
        try:
            start = time.perf_counter()
            aco = ACORouting(
                G,
                source,
                dest,
                bandwidth,
                weights,
                n_ants=ants,
                n_iterations=iterations,
                alpha=alpha,
                beta=beta,
                evaporation=evaporation,
                Q=q_value,
            )
            path, fitness, _ = aco.solve()
            duration = time.perf_counter() - start
            evaluation = evaluate_path(G, path, bandwidth, weights)
            evaluation.run_id = run_idx
            evaluation.duration = duration
            evaluation.raw_score = fitness
            records.append(evaluation)
        except Exception as exc:
            records.append(
                RunRecord(run_id=run_idx, success=False, reason=f"ACO çalıştırma hatası: {exc}", duration=0.0)
            )
    return records


def run_q_learning(
    source: int,
    dest: int,
    bandwidth: float,
    repeats: int,
    weights: Sequence[float],
    episodes: int,
    alpha: float,
    gamma: float,
    max_steps: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    q_weights = ql_normalize_weights(weights[0], weights[1], weights[2])
    reward_fn = make_reward_fn(q_weights, demand_mbps=bandwidth)

    for run_idx in range(1, repeats + 1):
        try:
            start = time.perf_counter()
            q_table = ql_train(
                G,
                QL_NEIGHBORS,
                start_node=source,
                goal_node=dest,
                reward_fn=reward_fn,
                episodes=episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay_steps=epsilon_decay_steps,
                max_steps_per_episode=max_steps,
                stochastic_fail=False,
            )
            duration = time.perf_counter() - start
            path = ql_greedy_path(q_table, QL_NEIGHBORS, source, dest)
            evaluation = evaluate_path(G, path, bandwidth, weights)
            evaluation.run_id = run_idx
            evaluation.duration = duration
            records.append(evaluation)
        except Exception as exc:
            records.append(
                RunRecord(run_id=run_idx, success=False, reason=f"Q-learning hatası: {exc}", duration=0.0)
            )
    return records


def build_report_section(
    case_idx: int,
    combo: Tuple[int, int, float],
    summaries: Dict[str, Dict[str, Optional[float]]],
    records: Dict[str, List[RunRecord]],
) -> List[str]:
    lines = []
    source, dest, bandwidth = combo
    lines.append(f"\n=== Deney {case_idx:02d}: S={source}, D={dest}, B={bandwidth:.2f} Mbps ===")
    for algo_name, summary in summaries.items():
        if summary["avg_cost"] is not None:
            lines.append(
                f"\n[{algo_name}] Başarı: {summary['success_count']}/{summary['attempts']} | "
                f"Avg Cost: {summary['avg_cost']:.4f}"
            )
        else:
            lines.append(f"\n[{algo_name}] Başarı: {summary['success_count']}/{summary['attempts']} | Avg Cost: ---")
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
        if summary["failures"]:
            lines.append("   Başarısız denemeler:")
            for fail in summary["failures"]:
                lines.append(f"      · Tekrar {fail['run']}: {fail['reason']}")
        else:
            lines.append("   Başarısız deneme yok.")

        for rec in records[algo_name]:
            if rec.success and rec.metrics:
                lines.append(
                    f"      -> Tekrar {rec.run_id}: Delay={rec.metrics['delay_ms']:.2f} ms | "
                    f"Rel={rec.metrics['reliability']:.5f} | Bottleneck={rec.metrics['bottleneck_mbps']:.2f} Mbps | "
                    f"Maliyet={rec.metrics['weighted_cost']:.4f}"
                )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="BSM307 ağ rotalama algoritmalarını otomatik deney düzeneğinde kıyaslar."
    )
    parser.add_argument("--repeats", type=int, default=5, help="Her kombinasyon için algoritma tekrar sayısı.")
    parser.add_argument("--demands", type=int, default=20, help="Demand dosyasından kaç adet (S,D,B) alınacak.")
    parser.add_argument("--demand-offset", type=int, default=0, help="Demand dosyasında başlanacak satır indexi.")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=DEFAULT_WEIGHTS,
        metavar=("W_DELAY", "W_REL", "W_RES"),
        help="Gecikme / Güvenilirlik / Kaynak ağırlıkları.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ga", "aco", "qlearning"],
        choices=["ga", "aco", "qlearning"],
        help="Çalıştırılacak algoritmalar.",
    )
    parser.add_argument("--output", type=str, default=None, help="Raporun kaydedileceği dosya adı.")
    parser.add_argument("--demand-file", type=str, default=DEMAND_FILE, help="Demand CSV dosyası yolu.")
    parser.add_argument("--ga-pop", type=int, default=100, help="Genetik algoritma popülasyon büyüklüğü.")
    parser.add_argument("--ga-generations", type=int, default=200, help="Genetik algoritma nesil sayısı.")
    parser.add_argument("--ga-mutation", type=float, default=0.1, help="Genetik algoritma mutasyon oranı.")
    parser.add_argument("--aco-ants", type=int, default=25, help="ACO'daki karınca sayısı.")
    parser.add_argument("--aco-iterations", type=int, default=50, help="ACO iterasyon sayısı.")
    parser.add_argument("--aco-alpha", type=float, default=1.0, help="ACO pheromone ağırlığı (alpha).")
    parser.add_argument("--aco-beta", type=float, default=2.0, help="ACO sezgisel bilgi ağırlığı (beta).")
    parser.add_argument("--aco-evap", type=float, default=0.5, help="ACO pheromone buharlaşma katsayısı.")
    parser.add_argument("--aco-q", type=float, default=100.0, help="ACO pheromone güçlendirme sabiti (Q).")
    parser.add_argument("--ql-episodes", type=int, default=2500, help="Q-learning eğitim episode sayısı.")
    parser.add_argument("--ql-alpha", type=float, default=0.15, help="Q-learning öğrenme oranı.")
    parser.add_argument("--ql-gamma", type=float, default=0.95, help="Q-learning gamma değeri.")
    parser.add_argument("--ql-max-steps", type=int, default=200, help="Q-learning bölüm başına maksimum adım.")
    parser.add_argument("--ql-epsilon-start", type=float, default=1.0, help="Q-learning başlangıç epsilon değeri.")
    parser.add_argument("--ql-epsilon-end", type=float, default=0.05, help="Q-learning minimum epsilon değeri.")
    parser.add_argument("--ql-epsilon-decay", type=int, default=2000, help="Q-learning epsilon azalma adım sayısı.")
    parser.add_argument("--seed", type=int, default=None, help="Rastgele algoritmalar için tekrar üretilebilir seed.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    weights = normalize_weight_list(args.weights)
    combos = load_demands(args.demand_file, args.demands, args.demand_offset)
    if len(combos) < args.demands:
        print(
            f"⚠️  Demand dosyasında {args.demands} adet kayıt bulunamadı. {len(combos)} adet kombinasyon çalıştırılacak."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"deney_detay_{timestamp}.txt"

    overall_report: List[str] = []
    overall_report.append(f"Deney Tarihi: {timestamp}")
    overall_report.append(f"Kullanılan ağırlıklar (normalize): {weights}")
    overall_report.append(
        f"Algoritmalar: {', '.join(args.algorithms)} | Demand kayıt sayısı: {len(combos)} | Tekrar sayısı: {args.repeats}"
    )

    overall_summaries: Dict[str, List[int]] = {algo: [] for algo in args.algorithms}
    for idx, combo in enumerate(combos, start=1):
        source, dest, bandwidth = combo
        print(f"[{idx}/{len(combos)}] S={source}, D={dest}, B={bandwidth} Mbps için deney başlıyor...")
        algo_records: Dict[str, List[RunRecord]] = {}
        summaries: Dict[str, Dict[str, Optional[float]]] = {}

        if "ga" in args.algorithms:
            records = run_genetic_algorithm(
                source,
                dest,
                bandwidth,
                args.repeats,
                weights,
                pop_size=args.ga_pop,
                generations=args.ga_generations,
                mutation_rate=args.ga_mutation,
            )
            algo_records["ga"] = records
            summaries["ga"] = summarize_runs(records)
            overall_summaries["ga"].append(summaries["ga"]["success_count"])

        if "aco" in args.algorithms:
            records = run_aco(
                source,
                dest,
                bandwidth,
                args.repeats,
                weights,
                ants=args.aco_ants,
                iterations=args.aco_iterations,
                alpha=args.aco_alpha,
                beta=args.aco_beta,
                evaporation=args.aco_evap,
                q_value=args.aco_q,
            )
            algo_records["aco"] = records
            summaries["aco"] = summarize_runs(records)
            overall_summaries["aco"].append(summaries["aco"]["success_count"])

        if "qlearning" in args.algorithms:
            records = run_q_learning(
                source,
                dest,
                bandwidth,
                args.repeats,
                weights,
                episodes=args.ql_episodes,
                alpha=args.ql_alpha,
                gamma=args.ql_gamma,
                max_steps=args.ql_max_steps,
                epsilon_start=args.ql_epsilon_start,
                epsilon_end=args.ql_epsilon_end,
                epsilon_decay_steps=args.ql_epsilon_decay,
            )
            algo_records["qlearning"] = records
            summaries["qlearning"] = summarize_runs(records)
            overall_summaries["qlearning"].append(summaries["qlearning"]["success_count"])

        readable_names = {"ga": "Genetik Algoritma", "aco": "Karınca Kolonisi", "qlearning": "Q-Learning"}
        readable_summaries = {readable_names[k]: v for k, v in summaries.items()}
        readable_records = {readable_names[k]: v for k, v in algo_records.items()}
        overall_report.extend(build_report_section(idx, combo, readable_summaries, readable_records))

    overall_report.append("\n=== Genel Başarı Özeti ===")
    for algo in args.algorithms:
        success_counts = overall_summaries[algo]
        total_cases = len(success_counts)
        fully_successful = sum(1 for c in success_counts if c > 0)
        overall_report.append(
            f"{algo.upper()}: {fully_successful}/{total_cases} kombinasyonda en az bir geçerli rota bulundu."
        )

    report_text = "\n".join(overall_report)
    print("\nDeney raporu oluşturuldu. Kaydediliyor...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ Rapor: {output_path}")


if __name__ == "__main__":
    main()
