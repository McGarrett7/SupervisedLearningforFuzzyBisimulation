"""
Neural Similarity Model Module.

Defines the GNN-based pairwise predictor of Sections 4.4-4.6 of the paper:

    f(s_i, s_j) = Sigmoid( MLP( [ h_i || h_j || |h_i - h_j| ] ) ),   h = GNN(G)

The encoder stacks torch_geometric GCNConv layers that use the fuzzy weights as edge
weights.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class FuzzyBisimNet(nn.Module):
    """
    GNN encoder followed by a metric-aware pairwise similarity head.

    use_gnn=False replaces the graph convolutions by linear layers (no neighborhood
    aggregation) and use_diff=False drops |h_i - h_j|; both exist for the ablation
    study (Section 5.6).
    """

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

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Maps every entity to its structural representation h_i (Eq. 3-4)."""
        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h, edge_index, edge_weight) if self.use_gnn else layer(h)
            if i < len(self.encoder) - 1:
                h = torch.relu(h)
        return h

    def score(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Predicts the similarity of the entity pairs (P x 2) from the representations h (Eq. 5-7)."""
        h_i, h_j = h[pairs[:, 0]], h[pairs[:, 1]]
        parts = [h_i, h_j, torch.abs(h_i - h_j)] if self.use_diff else [h_i, h_j]
        return torch.sigmoid(self.mlp(torch.cat(parts, dim=-1))).squeeze(-1)

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
