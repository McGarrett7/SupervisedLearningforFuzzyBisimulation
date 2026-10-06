# Mô hình dự đoán độ tương đồng theo cặp:
# f(s_i, s_j) = Sigmoid(MLP([h_i || h_j || |h_i - h_j|])), h = GCNConv(G) với trọng số mờ làm trọng số cạnh

from typing import Any, Dict
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Bộ mã hoá GNN và đầu dự đoán theo cặp. use_gnn=False thay GCNConv bằng lớp tuyến tính,
# use_diff=False bỏ |h_i - h_j| (dùng cho ablation)
class FuzzyBisimNet(nn.Module):

    # Khởi tạo các lớp mã hoá và đầu MLP theo cấu hình.
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_mlp_layers: int = 3,
        use_gnn: bool = True,
        use_diff: bool = True,
    ):
        super().__init__()
        self.config: Dict[str, Any] = dict(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_mlp_layers=num_mlp_layers,
            use_gnn=use_gnn,
            use_diff=use_diff,
        )
        self.use_gnn = use_gnn
        self.use_diff = use_diff

        dims = [in_dim] + [hidden_dim] * num_gnn_layers
        layer = GCNConv if use_gnn else nn.Linear
        self.encoder = nn.ModuleList(layer(dims[i], dims[i + 1]) for i in range(num_gnn_layers))

        head = []
        head_in = hidden_dim * (3 if use_diff else 2)
        for _ in range(num_mlp_layers - 1):
            head += [nn.Linear(head_in, hidden_dim), nn.ReLU()]
            head_in = hidden_dim
        head.append(nn.Linear(head_in, 1))
        self.mlp = nn.Sequential(*head)

    # Ánh xạ mỗi thực thể thành biểu diễn cấu trúc h_i
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h, edge_index, edge_weight) if self.use_gnn else layer(h)
            if i < len(self.encoder) - 1:
                h = torch.relu(h)
        return h

    # Dự đoán độ tương đồng của các cặp (P x 2) từ biểu diễn h
    def score(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        h_i, h_j = h[pairs[:, 0]], h[pairs[:, 1]]
        parts = [h_i, h_j, torch.abs(h_i - h_j)] if self.use_diff else [h_i, h_j]
        return torch.sigmoid(self.mlp(torch.cat(parts, dim=-1))).squeeze(-1)

    # Mã hoá đồ thị rồi chấm điểm các cặp.
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        return self.score(self.encode(x, edge_index, edge_weight), pairs)


if __name__ == "__main__":
    print("Model module initialized.")
