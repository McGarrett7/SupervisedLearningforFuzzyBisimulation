"""
Fuzzy Knowledge Graph Loading and Fuzzification Module.

Loads a benchmark knowledge graph (FB15k-237, WN18RR, YAGO3-10) stored as
tab-separated ``head<TAB>relation<TAB>tail`` triples and converts it into a fuzzy
knowledge graph G = (S, R, w) by assigning a fuzzy weight to every observed triple
(Section 5.1 of the paper):

    w(s, r, s') = Sigmoid(alpha * Freq(r) + beta * Deg(s, s')),   alpha = 0.6, beta = 0.4
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

TRIPLE_FILES = ("train.txt", "valid.txt", "test.txt")


@dataclass
class FuzzyKnowledgeGraph:
    """
    Fuzzy knowledge graph G = (S, R, w) stored as a weighted edge list.

    Edge e encodes the fuzzy fact w(heads[e], relations[e], tails[e]) = weights[e].
    """

    num_entities: int
    num_relations: int
    heads: np.ndarray
    relations: np.ndarray
    tails: np.ndarray
    weights: np.ndarray
    entity_names: List[str]
    relation_names: List[str]

    @property
    def num_edges(self) -> int:
        return int(self.heads.shape[0])


def _read_dict(path: Path) -> Optional[List[str]]:
    """
    Reads an ``id<TAB>name`` vocabulary file.

    Returns None when the file is missing or malformed, in which case the vocabulary
    is rebuilt from the triples instead.
    """
    if not path.exists():
        return None
    names: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 2 or parts[0] != str(len(names)):
                return None
            names.append(parts[1])
    return names or None


def compute_fuzzy_weights(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    num_entities: int,
    num_relations: int,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> np.ndarray:
    """
    Assigns a fuzzy weight to each triple from relation frequency and local connectivity.

    The paper does not fix how Freq and Deg are scaled, so both are normalised to [0, 1]:
    Freq(r) is the triple count of r divided by the largest relation count, and
    Deg(s, s') is the mean of the log-scaled total degrees of the two incident entities.

    Returns:
        np.ndarray: Fuzzy weights in (0, 1), one per triple.
    """
    rel_count = np.bincount(relations, minlength=num_relations).astype(np.float64)
    freq = rel_count / max(rel_count.max(), 1.0)

    degree = np.bincount(heads, minlength=num_entities) + np.bincount(tails, minlength=num_entities)
    log_degree = np.log1p(degree) / max(np.log1p(degree.max()), 1e-12)
    connectivity = 0.5 * (log_degree[heads] + log_degree[tails])

    logits = alpha * freq[relations] + beta * connectivity
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def load_fuzzy_kg(dataset_dir: Path, alpha: float = 0.6, beta: float = 0.4) -> FuzzyKnowledgeGraph:
    """
    Loads the triples found in dataset_dir and fuzzifies them.

    Args:
        dataset_dir: Directory containing train.txt (and optionally valid.txt, test.txt,
            entities.dict, relations.dict).
        alpha: Weight of the relation-frequency term.
        beta: Weight of the local-connectivity term.

    Returns:
        FuzzyKnowledgeGraph: The fuzzified graph with duplicate triples removed.
    """
    entity_names = _read_dict(dataset_dir / "entities.dict") or []
    relation_names = _read_dict(dataset_dir / "relations.dict") or []
    entity_ids: Dict[str, int] = {name: i for i, name in enumerate(entity_names)}
    relation_ids: Dict[str, int] = {name: i for i, name in enumerate(relation_names)}

    triples = []
    for file_name in TRIPLE_FILES:
        path = dataset_dir / file_name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                head, relation, tail = parts
                triples.append((
                    entity_ids.setdefault(head, len(entity_ids)),
                    relation_ids.setdefault(relation, len(relation_ids)),
                    entity_ids.setdefault(tail, len(entity_ids)),
                ))
    if not triples:
        raise FileNotFoundError(f"No triples found in {dataset_dir} (expected {TRIPLE_FILES[0]}).")

    array = np.unique(np.asarray(triples, dtype=np.int64), axis=0)
    heads, relations, tails = array[:, 0], array[:, 1], array[:, 2]
    num_entities, num_relations = len(entity_ids), len(relation_ids)

    return FuzzyKnowledgeGraph(
        num_entities=num_entities,
        num_relations=num_relations,
        heads=heads,
        relations=relations,
        tails=tails,
        weights=compute_fuzzy_weights(heads, relations, tails, num_entities, num_relations, alpha, beta),
        entity_names=list(entity_ids),
        relation_names=list(relation_ids),
    )


def structural_node_features(
    num_entities: int,
    num_relations: int,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Builds the initial entity representations h^(0) from the fuzzy transition structure.

    Each entity is described by the strongest outgoing and incoming fuzzy weight per
    relation type, followed by its log-scaled out- and in-degree.

    Returns:
        np.ndarray: Feature matrix of shape (num_entities, 2 * num_relations + 2).
    """
    features = np.zeros((num_entities, 2 * num_relations + 2), dtype=np.float32)
    np.maximum.at(features, (heads, relations), weights)
    np.maximum.at(features, (tails, num_relations + relations), weights)

    out_degree = np.bincount(heads, minlength=num_entities)
    in_degree = np.bincount(tails, minlength=num_entities)
    scale = max(np.log1p(max(out_degree.max(), in_degree.max())), 1e-12)
    features[:, -2] = np.log1p(out_degree) / scale
    features[:, -1] = np.log1p(in_degree) / scale
    return features


if __name__ == "__main__":
    print("Fuzzy Knowledge Graph module initialized.")
