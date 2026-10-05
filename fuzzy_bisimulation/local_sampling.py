"""
Local Sampling and Subgraph Extraction Module.

Extracts h-hop local subgraphs G_k = N_h(s_k) around seed entities (Sections 4.2 and 5.2
of the paper) so that exact fuzzy bisimulation only has to be computed on graphs of
bounded size instead of the whole Cartesian product S x S.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .fuzzy_kg import FuzzyKnowledgeGraph


@dataclass
class LocalSubgraph:
    """
    Induced local subgraph G_k = (S_k, R_k, w_k).

    nodes holds the global entity ids of S_k (the seed comes first); heads and tails are
    local indices into nodes, relations keeps the global relation ids.
    """

    seed: int
    nodes: np.ndarray
    heads: np.ndarray
    relations: np.ndarray
    tails: np.ndarray
    weights: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])


def _group_by(keys: np.ndarray, num_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (indptr, order) so that order[indptr[v]:indptr[v + 1]] lists the positions with key v."""
    order = np.argsort(keys, kind="stable")
    indptr = np.zeros(num_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=num_groups), out=indptr[1:])
    return indptr, order


class LocalNeighborhoodSampler:
    """
    Samples h-hop neighborhoods around seed entities using breadth-first search.
    """

    def __init__(self, graph: FuzzyKnowledgeGraph, num_hops: int = 2, max_nodes: int = 64):
        self.graph = graph
        self.num_hops = num_hops
        self.max_nodes = max_nodes

        # Neighborhoods are expanded along both edge directions.
        self._neighbors = np.concatenate([graph.tails, graph.heads])
        self._neighbor_weights = np.concatenate([graph.weights, graph.weights])
        self._adj_indptr, self._adj_order = _group_by(
            np.concatenate([graph.heads, graph.tails]), graph.num_entities
        )
        self._out_indptr, self._out_order = _group_by(graph.heads, graph.num_entities)
        self._local_index = np.full(graph.num_entities, -1, dtype=np.int64)

    def _incident(self, nodes: np.ndarray, indptr: np.ndarray, order: np.ndarray) -> np.ndarray:
        return np.concatenate([order[indptr[v]:indptr[v + 1]] for v in nodes])

    def sample(self, seed: int) -> LocalSubgraph:
        """
        Extracts the subgraph induced by the entities within num_hops of the seed.

        When a BFS frontier does not fit in the remaining node budget, the boundary
        entities with the largest total fuzzy weight towards the previous frontier are
        kept, which favours strongly and densely connected neighbors.

        Args:
            seed: Global id of the seed entity s_k.

        Returns:
            LocalSubgraph: Induced subgraph with at most max_nodes entities.
        """
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
