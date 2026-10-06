# Tính fuzzy bisimulation chính xác B* = gfp(F) bằng lặp điểm bất động (Mục 3.2 của bài báo).

from typing import Callable, Dict, Tuple
import numpy as np

Operator = Callable[[np.ndarray, np.ndarray], np.ndarray]


# Phép kéo theo Gödel.
def _godel_implication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a <= b, 1.0, b)


# T-norm Łukasiewicz.
def _lukasiewicz_tnorm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a + b - 1.0, 0.0)


# Phép kéo theo Łukasiewicz.
def _lukasiewicz_implication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(1.0 - a + b, 1.0)


# Phép kéo theo của t-norm tích.
def _product_implication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a <= b, 1.0, b / np.maximum(a, 1e-12))


# Mỗi ngữ nghĩa mờ là một cặp (t-norm, phép kéo theo).
SEMANTICS: Dict[str, Tuple[Operator, Operator]] = {
    "godel": (np.minimum, _godel_implication),
    "lukasiewicz": (_lukasiewicz_tnorm, _lukasiewicz_implication),
    "product": (np.multiply, _product_implication),
}


# Tính thành phần một chiều của toán tử F cho mọi cặp trạng thái, chỉ duyệt các cạnh có thật.
def _directional_component(
    relation: np.ndarray,
    num_states: int,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    weights: np.ndarray,
    tnorm: Operator,
    implication: Operator,
) -> np.ndarray:
    result = np.ones((num_states, num_states), dtype=np.float64)
    for r in np.unique(relations):
        mask = relations == r
        h, t, w = heads[mask], tails[mask], weights[mask]

        # matched[t, s'] = max_{t'} ( w(t, r, t') (x) B(s', t') )
        matched = np.zeros((num_states, num_states), dtype=np.float64)
        np.maximum.at(matched, h, tnorm(w[:, None], relation[:, t].T))

        # result[s, t] = min_{s'} ( w(s, r, s') -> matched[t, s'] )
        np.minimum.at(result, h, implication(w[:, None], matched[:, t].T))
    return result


# Áp dụng toán tử F lên quan hệ mờ B: lấy min của chiều thuận và chiều ngược.
def fuzzy_bisimulation_operator(
    relation: np.ndarray,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    weights: np.ndarray,
    semantics: str = "godel",
) -> np.ndarray:
    tnorm, implication = SEMANTICS[semantics]
    num_states = relation.shape[0]
    forward = _directional_component(relation, num_states, heads, relations, tails, weights, tnorm, implication)
    backward = _directional_component(relation.T, num_states, heads, relations, tails, weights, tnorm, implication)
    return np.minimum(forward, backward.T)


# Lặp B_{k+1} = F(B_k) từ quan hệ toàn 1 cho tới khi hội tụ về điểm bất động lớn nhất.
def compute_exact_fuzzy_bisimulation(
    num_states: int,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    weights: np.ndarray,
    semantics: str = "godel",
    tolerance: float = 1e-5,
    max_iter: int = 1000,
) -> np.ndarray:
    if semantics not in SEMANTICS:
        raise ValueError(f"Unknown semantics '{semantics}'; choose from {sorted(SEMANTICS)}.")
    heads = np.asarray(heads, dtype=np.int64)
    relations = np.asarray(relations, dtype=np.int64)
    tails = np.asarray(tails, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    relation = np.ones((num_states, num_states), dtype=np.float64)
    for _ in range(max_iter):
        prev_relation = relation
        relation = fuzzy_bisimulation_operator(prev_relation, heads, relations, tails, weights, semantics)

        diff = np.max(np.abs(relation - prev_relation)) if num_states else 0.0
        if diff < tolerance:
            break

    return relation.astype(np.float32)


if __name__ == "__main__":
    # Trạng thái 0 và 1 có cạnh r với trọng số khác nhau tới cùng một nút.
    demo = compute_exact_fuzzy_bisimulation(
        num_states=3,
        heads=np.array([0, 1]),
        relations=np.array([0, 0]),
        tails=np.array([2, 2]),
        weights=np.array([0.9, 0.6]),
    )
    print(demo)
