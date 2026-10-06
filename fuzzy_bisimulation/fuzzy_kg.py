# Nạp đồ thị tri thức và fuzzy hoá các bộ ba (Mục 5.1 của bài báo).

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

TRIPLE_FILES = ("train.txt", "valid.txt", "test.txt")


# Đồ thị tri thức mờ G = (S, R, w) lưu dạng danh sách cạnh có trọng số.
@dataclass
class FuzzyKnowledgeGraph:
    num_entities: int
    num_relations: int
    heads: np.ndarray
    relations: np.ndarray
    tails: np.ndarray
    weights: np.ndarray
    entity_names: List[str]
    relation_names: List[str]

    # Số bộ ba mờ của đồ thị.
    @property
    def num_edges(self) -> int:
        return int(self.heads.shape[0])


# Đọc file từ vựng `id<TAB>tên`, bỏ qua nếu file thiếu hoặc sai định dạng.
def _read_dict(path: Path) -> Optional[List[str]]:
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


# Gán trọng số mờ cho mỗi bộ ba: Sigmoid(alpha * Freq(r) + beta * Deg(s, s')).
def compute_fuzzy_weights(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    num_entities: int,
    num_relations: int,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> np.ndarray:
    rel_count = np.bincount(relations, minlength=num_relations).astype(np.float64)
    freq = rel_count / max(rel_count.max(), 1.0)

    degree = np.bincount(heads, minlength=num_entities) + np.bincount(tails, minlength=num_entities)
    log_degree = np.log1p(degree) / max(np.log1p(degree.max()), 1e-12)
    connectivity = 0.5 * (log_degree[heads] + log_degree[tails])

    logits = alpha * freq[relations] + beta * connectivity
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


# Đọc các bộ ba của một bộ dữ liệu, bỏ trùng lặp và fuzzy hoá.
def load_fuzzy_kg(dataset_dir: Path, alpha: float = 0.6, beta: float = 0.4) -> FuzzyKnowledgeGraph:
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


# Tạo đặc trưng ban đầu của nút từ trọng số mờ theo từng quan hệ và bậc của nút.
def structural_node_features(
    num_entities: int,
    num_relations: int,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
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
