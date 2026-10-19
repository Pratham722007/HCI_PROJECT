"""Inter-rater agreement and multiple-comparison helpers."""
import numpy as np


def krippendorff_interval(M):
    """M (raters, items) with NaN for missing. Interval-metric alpha."""
    M = np.asarray(M, float)
    units = [M[:, j][~np.isnan(M[:, j])] for j in range(M.shape[1])]
    units = [u for u in units if len(u) >= 2]
    vals = np.concatenate(units); n = len(vals)
    Do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    De = ((vals[:, None] - vals[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - Do / De) if De > 0 else float("nan")


def fleiss_kappa(M, cats=(1, 2, 3, 4, 5)):
    """M (raters, items), complete items only."""
    M = np.asarray(M, float); M = M[:, ~np.isnan(M).any(0)]
    if M.shape[1] == 0:
        return float("nan")
    n = M.shape[0]
    counts = np.stack([(M == c).sum(0) for c in cats], 1)                 # (items, cats)
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p, dtype=float)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)
