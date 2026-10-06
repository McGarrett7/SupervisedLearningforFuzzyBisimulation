# Lấy mẫu đồ thị con cục bộ h-hop G_k = N_h(s_k) quanh thực thể hạt giống

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .fuzzy_kg import FuzzyKnowledgeGraph


@dataclass
class LocalSubgraph:
    # Đồ thị con cảm sinh G_k; nodes là id toàn cục (hạt giống đứng đầu)
    # heads/tails là chỉ số cục bộ trong nodes

    seed: int
    nodes: np.ndarray
    heads: np.ndarray
    relations: np.ndarray
    tails: np.ndarray
    weights: np.ndarray

    # Số nút của đồ thị con
    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])


# Gom các vị trí theo khoá để tra nhanh danh sách cạnh của từng nút
def _group_by(keys: np.ndarray, num_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    indptr = np.zeros(num_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=num_groups), out=indptr[1:])
    return indptr, order


# Bộ lấy mẫu lân cận h-hop quanh thực thể hạt giống bằng BFS
class LocalNeighborhoodSampler:
    # Dựng sẵn chỉ mục cạnh kề (cả hai chiều) và cạnh đi ra của từng nút
    def __init__(self, graph: FuzzyKnowledgeGraph, num_hops: int = 2, max_nodes: int = 64):
        self.graph = graph
        self.num_hops = num_hops
        self.max_nodes = max_nodes

        # Mở rộng lân cận theo cả hai chiều cạnh
        self._neighbors = np.concatenate([graph.tails, graph.heads])
        self._neighbor_weights = np.concatenate([graph.weights, graph.weights])
        self._adj_indptr, self._adj_order = _group_by(
            np.concatenate([graph.heads, graph.tails]), graph.num_entities
        )
        self._out_indptr, self._out_order = _group_by(graph.heads, graph.num_entities)
        self._local_index = np.full(graph.num_entities, -1, dtype=np.int64)

    # Lấy các cạnh gắn với một tập nút
    def _incident(self, nodes: np.ndarray, indptr: np.ndarray, order: np.ndarray) -> np.ndarray:
        return np.concatenate([order[indptr[v]:indptr[v + 1]] for v in nodes])

    # Trích đồ thị con quanh hạt giống, khi vượt số nút tối đa thì giữ các nút nối mạnh nhất về lớp trước
    def sample(self, seed: int) -> LocalSubgraph:
        selected = [np.array([seed], dtype=np.int64)]
        self._local_index[seed] = 0
        num_selected = 1
        frontier = selected[0]

        for _ in range(self.num_hops):
            budget = self.max_nodes - num_selected
            if budget <= 0 or frontier.size == 0:
                break
            incident = self._incident(frontier, self._adj_indptr, self._adj_order)
            neighbors = self._neighbors[incident]
            unseen = self._local_index[neighbors] < 0
            candidates, inverse = np.unique(neighbors[unseen], return_inverse=True)
            if candidates.size > budget:
                score = np.bincount(inverse, weights=self._neighbor_weights[incident][unseen])
                candidates = candidates[np.argsort(-score, kind="stable")[:budget]]

            self._local_index[candidates] = num_selected + np.arange(candidates.size)
            num_selected += candidates.size
            selected.append(candidates)
            frontier = candidates

        nodes = np.concatenate(selected)
        edges = self._incident(nodes, self._out_indptr, self._out_order)
        edges = edges[self._local_index[self.graph.tails[edges]] >= 0]
        subgraph = LocalSubgraph(
            seed=int(seed),
            nodes=nodes,
            heads=self._local_index[self.graph.heads[edges]],
            relations=self.graph.relations[edges],
            tails=self._local_index[self.graph.tails[edges]],
            weights=self.graph.weights[edges],
        )
        self._local_index[nodes] = -1
        return subgraph


if __name__ == "__main__":
    print("Local Sampling module initialized.")
