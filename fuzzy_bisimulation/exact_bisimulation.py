"""
Exact Fuzzy Bisimulation Computation Module.

Computes the ground-truth fuzzy bisimulation relation B* = gfp(F) between all pairs of
entities of a fuzzy knowledge graph G = (S, R, w) via fixed-point iteration (Section 3.2
of the paper). The directional component of the operator is

    F(B)(s, t) = min_{r in R} min_{s' in S} [ w(s, r, s') -> max_{t' in S} ( w(t, r, t') (x) B(s', t') ) ]

and F takes the minimum of this component and its reverse-direction counterpart, where
(x) is a t-norm and -> its residuum.
"""

from typing import Callable, Dict, Tuple
import numpy as np

Operator = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _godel_implication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a <= b, 1.0, b)


def _lukasiewicz_tnorm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a + b - 1.0, 0.0)


def _lukasiewicz_implication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(1.0 - a + b, 1.0)


def _product_implication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a <= b, 1.0, b / np.maximum(a, 1e-12))


# Each fuzzy semantics is a (t-norm, residuum) pair.
SEMANTICS: Dict[str, Tuple[Operator, Operator]] = {
    "godel": (np.minimum, _godel_implication),
    "lukasiewicz": (_lukasiewicz_tnorm, _lukasiewicz_implication),
    "product": (np.multiply, _product_implication),
}


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
    """
    Evaluates min_r min_{s'} [ w(s, r, s') -> max_{t'} ( w(t, r, t') (x) B(s', t') ) ] for all (s, t).

    Only existing transitions are visited: a missing transition has weight 0, which
    contributes 0 to the inner supremum and 1 to the outer infimum.
    """
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


def fuzzy_bisimulation_operator(
    relation: np.ndarray,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    weights: np.ndarray,
    semantics: str = "godel",
) -> np.ndarray:
    """
    Applies the fuzzy bisimulation operator F to a fuzzy relation B (N x N).

    Returns:
        np.ndarray: F(B), the minimum of the forward and the reverse-direction component.
    """
    tnorm, implication = SEMANTICS[semantics]
    num_states = relation.shape[0]
    forward = _directional_component(relation, num_states, heads, relations, tails, weights, tnorm, implication)
    backward = _directional_component(relation.T, num_states, heads, relations, tails, weights, tnorm, implication)
    return np.minimum(forward, backward.T)


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
    """
    Computes exact fuzzy bisimulation degrees between all pairs of states.

    Iterates B_{k+1} = F(B_k) from the full relation B_0 = 1 (the top element, so the
    limit is the greatest fixed point) until max |B_{k+1} - B_k| < tolerance.

    Args:
        num_states: Number of states N; states are indexed 0..N-1.
        heads: Source state of each fuzzy transition.
        relations: Relation type of each fuzzy transition.
        tails: Target state of each fuzzy transition.
        weights: Fuzzy degree w(s, r, s') of each transition, in [0, 1].
        semantics: Fuzzy semantics, one of "godel", "lukasiewicz", "product".
        tolerance: Convergence threshold for fixed-point iteration.
        max_iter: Maximum number of iterations.

    Returns:
        np.ndarray: Pairwise fuzzy bisimulation matrix (N x N) with values in [0, 1].
    """
    if semantics not in SEMANTICS:
        raise ValueError(f"Unknown semantics '{semantics}'; choose from {sorted(SEMANTICS)}.")
    heads = np.asarray(heads, dtype=np.int64)
    relations = np.asarray(relations, dtype=np.int64)
    tails = np.asarray(tails, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    relation = np.ones((num_states, num_states), dtype=np.float64)
    for step in range(max_iter):
        prev_relation = relation
        relation = fuzzy_bisimulation_operator(prev_relation, heads, relations, tails, weights, semantics)

        diff = np.max(np.abs(relation - prev_relation)) if num_states else 0.0
        if diff < tolerance:
            break

    return relation.astype(np.float32)


if __name__ == "__main__":
    # States 0 and 1 have r-transitions of different strength into the same sink.
    demo = compute_exact_fuzzy_bisimulation(
        num_states=3,
        heads=np.array([0, 1]),
        relations=np.array([0, 0]),
        tails=np.array([2, 2]),
        weights=np.array([0.9, 0.6]),
    )
    print(demo)
