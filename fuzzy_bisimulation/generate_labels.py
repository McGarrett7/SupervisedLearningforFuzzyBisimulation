"""
Dataset and Ground-Truth Label Generation.

Implements the offline semantic label generation stage of Algorithm 1: samples seed
entities, extracts their local subgraphs with local_sampling.py, computes ground-truth
fuzzy bisimulation values with exact_bisimulation.py, and exports the labeled entity
pairs (train/val/test) to the data/ directory.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

from .exact_bisimulation import SEMANTICS, compute_exact_fuzzy_bisimulation
from .fuzzy_kg import load_fuzzy_kg
from .local_sampling import LocalNeighborhoodSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def generate_dataset(
    data_dir: Path,
    output_dir: Path,
    num_pairs: int = 50000,
    pairs_per_subgraph: int = 100,
    num_hops: int = 2,
    max_nodes: int = 64,
    semantics: str = "godel",
    alpha: float = 0.6,
    beta: float = 0.4,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    """
    Orchestrates fuzzification, local subgraph sampling, exact bisimulation ground-truth
    computation, and dataset splitting (train/val/test).

    Every pair (s_i, s_j) is labeled with y_ij = B*_{G_k}(s_i, s_j), the greatest fixed
    point computed on the local subgraph G_k it was drawn from. All pairs of one subgraph
    go to the same partition to prevent leakage between the splits.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and fuzzifying knowledge graph from {data_dir}...")
    graph = load_fuzzy_kg(data_dir, alpha=alpha, beta=beta)
    print(f"  {graph.num_entities} entities, {graph.num_relations} relations, {graph.num_edges} fuzzy triples")

    sampler = LocalNeighborhoodSampler(graph, num_hops=num_hops, max_nodes=max_nodes)
    connected = np.unique(np.concatenate([graph.heads, graph.tails]))
    seeds = rng.permutation(connected)

    print(f"Generating {num_pairs} labeled pairs into {output_dir}...")
    pairs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    subgraph_ids: List[np.ndarray] = []
    subgraph_sizes: List[int] = []
    num_generated = 0
    exact_time = 0.0

    for seed_entity in seeds:
        if num_generated >= num_pairs:
            break
        subgraph = sampler.sample(int(seed_entity))
        size = subgraph.num_nodes
        if size < 2:
            continue

        start = time.perf_counter()
        bisimulation = compute_exact_fuzzy_bisimulation(
            size, subgraph.heads, subgraph.relations, subgraph.tails, subgraph.weights, semantics=semantics
        )
        exact_time += time.perf_counter() - start

        # Ordered pairs (i, j) with i != j, drawn without replacement.
        num_selected = min(pairs_per_subgraph, size * (size - 1), num_pairs - num_generated)
        flat = rng.choice(size * (size - 1), size=num_selected, replace=False)
        first, second = flat // (size - 1), flat % (size - 1)
        second = second + (second >= first)

        pairs.append(np.stack([subgraph.nodes[first], subgraph.nodes[second]], axis=1))
        labels.append(bisimulation[first, second])
        subgraph_ids.append(np.full(num_selected, len(subgraph_sizes), dtype=np.int64))
        subgraph_sizes.append(size)
        num_generated += num_selected
        if len(subgraph_sizes) % 100 == 0:
            print(f"  {len(subgraph_sizes)} subgraphs, {num_generated} pairs")

    all_pairs = torch.from_numpy(np.concatenate(pairs))
    all_labels = torch.from_numpy(np.concatenate(labels).astype(np.float32))
    all_subgraph_ids = torch.from_numpy(np.concatenate(subgraph_ids))

    # Seeds were drawn in random order, so consecutive blocks of subgraphs form the splits.
    num_subgraphs = len(subgraph_sizes)
    train_end = int(round(split[0] * num_subgraphs))
    val_end = train_end + int(round(split[1] * num_subgraphs))
    bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, num_subgraphs)}

    split_sizes: Dict[str, int] = {}
    for name in SPLITS:
        low, high = bounds[name]
        mask = (all_subgraph_ids >= low) & (all_subgraph_ids < high)
        split_sizes[name] = int(mask.sum())
        torch.save(
            {"pairs": all_pairs[mask], "labels": all_labels[mask], "subgraph_ids": all_subgraph_ids[mask]},
            output_dir / f"{name}.pt",
        )

    torch.save(
        {
            "num_entities": graph.num_entities,
            "num_relations": graph.num_relations,
            "edge_index": torch.from_numpy(np.stack([graph.heads, graph.tails])),
            "edge_type": torch.from_numpy(graph.relations),
            "edge_weight": torch.from_numpy(graph.weights),
        },
        output_dir / "graph.pt",
    )

    meta = {
        "dataset": data_dir.name,
        "num_entities": graph.num_entities,
        "num_relations": graph.num_relations,
        "num_triples": graph.num_edges,
        "num_pairs": num_generated,
        "num_subgraphs": num_subgraphs,
        "mean_subgraph_size": float(np.mean(subgraph_sizes)),
        "split_sizes": split_sizes,
        "label_mean": float(all_labels.mean()),
        "label_std": float(all_labels.std()),
        "exact_bisimulation_seconds": exact_time,
        "config": {
            "pairs_per_subgraph": pairs_per_subgraph,
            "num_hops": num_hops,
            "max_nodes": max_nodes,
            "semantics": semantics,
            "alpha": alpha,
            "beta": beta,
            "split": list(split),
            "seed": seed,
        },
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"  splits: {split_sizes}, exact bisimulation time: {exact_time:.1f}s")
    print("Dataset generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled dataset for fuzzy bisimulation.")
    parser.add_argument("--dataset", type=str, default="WN18RR", help="Dataset name (sub-directory of data/).")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory with the raw triples (default: data/<dataset>).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated dataset (default: data/<dataset>/processed).")
    parser.add_argument("--num_pairs", type=int, default=50000, help="Number of labeled pairs to generate.")
    parser.add_argument("--pairs_per_subgraph", type=int, default=100, help="Pairs sampled from each local subgraph.")
    parser.add_argument("--num_hops", type=int, default=2, help="Hop radius h of the local subgraphs.")
    parser.add_argument("--max_nodes", type=int, default=64, help="Maximum number of entities m per local subgraph.")
    parser.add_argument("--semantics", type=str, default="godel", choices=sorted(SEMANTICS), help="Fuzzy semantics of the bisimulation operator.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Fuzzification weight of the relation frequency.")
    parser.add_argument("--beta", type=float, default=0.4, help="Fuzzification weight of the local connectivity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / args.dataset
    generate_dataset(
        data_dir=data_dir,
        output_dir=Path(args.output_dir) if args.output_dir else data_dir / "processed",
        num_pairs=args.num_pairs,
        pairs_per_subgraph=args.pairs_per_subgraph,
        num_hops=args.num_hops,
        max_nodes=args.max_nodes,
        semantics=args.semantics,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
    )
