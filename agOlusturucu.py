"""Command-line interface for the Q-learning routing assignment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import networkx as nx

from graph_loader import load_graph
from qlearning_env import extract_greedy_path, path_metrics, train_q_learning

NODE_FILE = "BSM307_317_Guz2025_TermProject_NodeData.csv"
EDGE_FILE = "BSM307_317_Guz2025_TermProject_EdgeData.csv"


def select_source_destination(graph: nx.Graph, fixed_pair: Optional[str]) -> tuple[int, int]:
    if fixed_pair:
        src, dst = map(int, fixed_pair.split(","))
        return src, dst

    nodes = list(graph.nodes())
    s, d = random.sample(nodes, 2)
    while not nx.has_path(graph, s, d):
        s, d = random.sample(nodes, 2)
    return s, d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q-learning tabanlı yol optimizasyonu")
    parser.add_argument(
        "--mode",
        choices=["delay", "reliability", "efficiency"],
        default="delay",
        help="Optimizasyon hedefi",
    )
    parser.add_argument("--episodes", type=int, default=3000, help="Eğitim bölüm sayısı")
    parser.add_argument("--alpha", type=float, default=0.1, help="Öğrenme oranı")
    parser.add_argument("--gamma", type=float, default=0.95, help="İskonto faktörü")
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Sabit S,D çifti (örn. 12,87). Boş bırakılırsa rastgele seçilir",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    node_file = args.data_dir / NODE_FILE
    edge_file = args.data_dir / EDGE_FILE
    graph = load_graph(node_file, edge_file, verbose=True)

    source, destination = select_source_destination(graph, args.pair)
    print(f"Seçilen kaynak-hedef: {source} -> {destination}")
    print(f"Optimizasyon modu: {args.mode}")

    q_table, rewards = train_q_learning(
        graph,
        mode=args.mode,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        max_steps=args.max_steps,
        fixed_pair=(source, destination),
    )

    final_path = extract_greedy_path(q_table, graph, source, destination)
    if not final_path:
        print("Q-tablosu ile geçerli yol bulunamadı.")
        return

    metrics = path_metrics(graph, final_path)
    print("Bulunan yol:", final_path)
    print(
        "Gecikme={:.2f} ms, Güvenilirlik={:.6f}, Bottleneck BW={:.2f} Mbps".format(
            metrics["total_delay_ms"], metrics["total_reliability"], metrics["bottleneck_bw_Mbps"]
        )
    )
    print(
        "Son 10 bölüm ödülleri:",
        [round(r, 2) for r in rewards[-10:]],
    )


if __name__ == "__main__":
    main()
