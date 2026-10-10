# Huấn luyện có giám sát (Algorithm 1): Huber loss, Adam, dừng sớm theo MAE validation.

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple, Optional
import torch
import torch.nn as nn

from .fuzzy_kg import structural_node_features
from .model import FuzzyBisimNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Đồ thị mờ ở dạng tensor: đặc trưng nút, cạnh truyền tin (tail -> head) và trọng số mờ của cạnh.
class GraphData(NamedTuple):
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor


# Nạp đồ thị mờ do generate_labels.py xuất ra, kèm đặc trưng nút.
def load_graph(graph_path: Path, device: torch.device) -> GraphData:
    graph = torch.load(graph_path)
    heads, tails = graph["edge_index"].numpy()
    x = torch.from_numpy(
        structural_node_features(
            graph["num_entities"],
            graph["num_relations"],
            heads,
            graph["edge_type"].numpy(),
            tails,
            graph["edge_weight"].numpy(),
        )
    )
    # Đảo chiều cạnh để mỗi thực thể gom thông tin từ các nút kế tiếp, đúng với chiều chuyển đi ra của toán tử F.
    edge_index = graph["edge_index"].flip(0)
    return GraphData(x.to(device), edge_index.to(device), graph["edge_weight"].to(device))


# Chấm điểm các cặp thực thể bằng một lần lan truyền GNN trên toàn đồ thị.
@torch.no_grad()
def predict(
    model: FuzzyBisimNet,
    data: GraphData,
    pairs: torch.Tensor,
    batch_size: int = 65536,
) -> torch.Tensor:
    model.eval()
    h = model.encode(data.x, data.edge_index, data.edge_weight)
    return torch.cat([model.score(h, pairs[i:i + batch_size]) for i in range(0, pairs.shape[0], batch_size)])


# Huấn luyện mô hình, theo dõi MAE validation và lưu checkpoint tốt nhất.
def train(
    data_dir: Path,
    checkpoint_dir: Path,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_gnn_layers: int = 3,
    num_mlp_layers: int = 3,
    huber_delta: float = 1.0,
    patience: int = 10,
    use_gnn: bool = True,
    use_diff: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
) -> None:
    torch.manual_seed(seed)
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    data = load_graph(data_dir / "graph.pt", run_device)
    train_set = torch.load(data_dir / "train.pt")
    val_set = torch.load(data_dir / "val.pt")
    train_pairs, train_labels = train_set["pairs"].to(run_device), train_set["labels"].to(run_device)
    val_pairs, val_labels = val_set["pairs"].to(run_device), val_set["labels"].to(run_device)
    print(f"  {train_pairs.shape[0]} training pairs, {val_pairs.shape[0]} validation pairs, device: {run_device}")

    model = FuzzyBisimNet(
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        num_mlp_layers=num_mlp_layers,
        use_gnn=use_gnn,
        use_diff=use_diff,
    ).to(run_device)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {epochs} epochs (batch size: {batch_size}, lr: {lr})...")
    history = []
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    num_train = train_pairs.shape[0]

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        permutation = torch.randperm(num_train, device=run_device)
        total_loss = 0.0
        for i in range(0, num_train, batch_size):
            batch = permutation[i:i + batch_size]
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight, train_pairs[batch])
            loss = criterion(out, train_labels[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]

        train_loss = total_loss / num_train
        val_mae = (predict(model, data, val_pairs) - val_labels).abs().mean().item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | {time.perf_counter() - start:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "model_config": model.config, "epoch": epoch, "val_mae": val_mae},
                checkpoint_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best Val MAE: {best_val_mae:.4f}).")
                break

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training finished. Checkpoints saved to {checkpoint_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train supervised model for fuzzy bisimulation approximation.")
    parser.add_argument("--dataset", type=str, default="WN18RR", help="Dataset name (sub-directory of data/).")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to processed data directory (default: data/<dataset>/processed).")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save model checkpoints (default: checkpoints/<dataset>).")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GCN layers.")
    parser.add_argument("--num_mlp_layers", type=int, default=3, help="Number of layers of the similarity head.")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Threshold of the Huber loss.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience on validation MAE.")
    parser.add_argument("--no_gnn", action="store_true", help="Ablation: disable neighborhood aggregation.")
    parser.add_argument("--no_diff", action="store_true", help="Ablation: drop the |h_i - h_j| pair feature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available).")
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / args.dataset / "processed",
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else PROJECT_ROOT / "checkpoints" / args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_mlp_layers=args.num_mlp_layers,
        huber_delta=args.huber_delta,
        patience=args.patience,
        use_gnn=not args.no_gnn,
        use_diff=not args.no_diff,
        seed=args.seed,
        device=args.device,
    )
