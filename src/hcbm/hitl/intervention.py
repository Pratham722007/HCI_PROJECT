"""Concept intervention: replace a fraction of predicted concepts at one level with ground truth (the "expert").

Selection per window: random, or uncertainty-ordered (most uncertain concepts corrected first). Concept uncertainty
= closeness to the concept's decision boundary: L1 nearest tertile edge (+-0.43 z), L2 the 0.5 midpoint, L3 zero.
"""
import numpy as np

from hcbm.concepts.level1 import TERTILE_Z

BOUNDARY = {"l1": (-TERTILE_Z, TERTILE_Z), "l2": (0.5,), "l3": (0.0,)}


def uncertainty(level, pred):
    d = np.min(np.stack([np.abs(pred - b) for b in BOUNDARY[level]]), 0)
    return -d                                              # larger = more uncertain


def intervention_mask(level, pred, truth_ok, frac, order, rng):
    """(n, k) bool: which concepts to replace. Only concepts with a valid truth value are eligible."""
    n, k = pred.shape
    score = rng.random((n, k)) if order == "random" else uncertainty(level, pred) + 1e-9 * rng.random((n, k))
    score = np.where(truth_ok, score, -np.inf)
    n_fix = np.round(frac * truth_ok.sum(1)).astype(int)
    rank = np.argsort(np.argsort(-score, 1), 1)            # 0 = first to correct
    return (rank < n_fix[:, None]) & truth_ok
