"""
Evaluation and Metrics Benchmark Module.

Loads trained checkpoints and compares predictions against exact ground-truth values.
Outputs quantitative metrics (MAE, MSE, RMSE, Pearson / Spearman / Kendall-Tau
correlation), the inference time, and a Common Neighbors baseline, and exports
evaluation summaries to results/ directory.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.stats
import torch

from .model import FuzzyBisimNet
from .train import load_graph, predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Measures the numerical agreement between predicted and exact similarity values."""
    error = predictions - labels
    return {
        "mae": float(np.abs(error).mean()),
        "mse": float((error ** 2).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "pearson": float(scipy.stats.pearsonr(predictions, labels)[0]),
        "spearman": float(scipy.stats.spearmanr(predictions, labels)[0]),
        "kendall_tau": float(scipy.stats.kendalltau(predictions, labels)[0]),
    }


def common_neighbors_scores(edge_index: np.ndarray, num_entities: int, pairs: np.ndarray) -> np.ndarray:
    """
    Structural heuristic baseline: shared neighbors of the two entities.

    The paper does not state how the count is mapped to [0, 1]; here it is divided by
    the size of the larger of the two (undirected) neighborhoods.
    """
    rows = np.concatenate([edge_index[0], edge_index[1]])
    cols = np.concatenate([edge_index[1], edge_index[0]])
    adjacency = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(num_entities, num_entities))
    adjacency.data[:] = 1.0

    first, second = adjacency[pairs[:, 0]], adjacency[pairs[:, 1]]
    common = np.asarray(first.multiply(second).sum(axis=1)).ravel()
    largest = np.maximum(np.asarray(first.sum(axis=1)).ravel(), np.asarray(second.sum(axis=1)).ravel())
    return common / np.maximum(largest, 1.0)


def plot_predictions(predictions: np.ndarray, labels: np.ndarray, path: Path, max_points: int = 5000) -> None:
    """Saves a scatter plot of predicted against exact fuzzy bisimulation values."""
    if predictions.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(predictions.shape[0], max_points, replace=False)
        predictions, labels = predictions[keep], labels[keep]

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color="#898781", linewidth=1, linestyle="--", zorder=1)
    ax.scatter(labels, predictions, s=14, color="#2a78d6", alpha=0.25, linewidths=0, zorder=2)
    ax.text(0.97, 0.93, "exact = predicted", color="#52514e", fontsize=8, ha="right", va="top", rotation=45)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("Exact fuzzy bisimulation", color="#52514e")
    ax.set_ylabel("Predicted similarity", color="#52514e")
    ax.set_title("Predicted vs. exact fuzzy bisimulation (test pairs)", color="#0b0b0b", fontsize=10, loc="left")
    ax.grid(color="#898781", alpha=0.2, linewidth=0.5)
    ax.tick_params(colors="#898781", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#898781")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def evaluate(
    checkpoint_path: Path,
    test_data_path: Path,
    results_dir: Path,
    device: Optional[str] = None,
) -> None:
    """
    Evaluates the model on test pairs and logs metrics to results directory.
    """
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=run_device)
    model = FuzzyBisimNet(**checkpoint["model_config"]).to(run_device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"Evaluating on test dataset from {test_data_path}...")
    data = load_graph(test_data_path.parent / "graph.pt", run_device)
    test_set = torch.load(test_data_path)
    pairs = test_set["pairs"].to(run_device)
    labels = test_set["labels"].numpy()

    # Inference time covers the GNN forward pass and the scoring of all test pairs.
    if run_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = predict(model, data, pairs)
    if run_device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start
    predictions = predictions.cpu().numpy()

    start = time.perf_counter()
    baseline = common_neighbors_scores(data.edge_index.cpu().numpy(), data.num_nodes, test_set["pairs"].numpy())
    baseline_time = time.perf_counter() - start

    report = {
        "num_test_pairs": int(labels.shape[0]),
        "checkpoint_epoch": checkpoint["epoch"],
        "model_config": checkpoint["model_config"],
        "proposed": {**compute_metrics(predictions, labels), "inference_seconds": inference_time},
        "common_neighbors": {**compute_metrics(baseline, labels), "inference_seconds": baseline_time},
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savetxt(
        results_dir / "predictions.csv",
        np.column_stack([test_set["pairs"].numpy(), labels, predictions]),
        fmt=["%d", "%d", "%.6f", "%.6f"],
        delimiter=",",
        header="entity_i,entity_j,exact,predicted",
        comments="",
    )
    plot_predictions(predictions, labels, results_dir / "predictions.png")

    for name in ("proposed", "common_neighbors"):
        metrics = report[name]
        print(
            f"  {name:17s} MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
            f"Pearson r: {metrics['pearson']:.4f} | Time: {metrics['inference_seconds']:.4f}s"
        )
    print(f"Evaluation complete. Reports written to {results_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fuzzy bisimulation approximation model.")
    parser.add_argument("--dataset", type=str, default="WN18RR", help="Dataset name (sub-directory of data/).")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint (default: checkpoints/<dataset>/best_model.pt).")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to test dataset (default: data/<dataset>/processed/test.pt).")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save evaluation results (default: results/<dataset>).")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available).")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else PROJECT_ROOT / "checkpoints" / args.dataset / "best_model.pt",
        test_data_path=Path(args.test_data_path) if args.test_data_path else PROJECT_ROOT / "data" / args.dataset / "processed" / "test.pt",
        results_dir=Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results" / args.dataset,
        device=args.device,
    )
