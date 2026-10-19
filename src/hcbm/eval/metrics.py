"""Metrics with subject-level resampling (windows of one subject are never treated as independent)."""
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1(y, p, labels=None):
    return float(f1_score(y, p, average="macro", labels=labels, zero_division=0))


def _subject_rows(subjects):
    subs = np.unique(subjects)
    return subs, [np.flatnonzero(subjects == s) for s in subs]


def subject_bootstrap(fn, subjects, *arrays, n_boot=1000, seed=42):
    """Point estimate + 95% CI of fn(*arrays) resampling SUBJECTS with replacement."""
    subs, rows = _subject_rows(np.asarray(subjects))
    rng = np.random.default_rng(seed)
    point = fn(*arrays)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(subs), len(subs))])
        stats.append(fn(*(a[idx] for a in arrays)))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return dict(est=float(point), lo=float(lo), hi=float(hi))


def paired_bootstrap(fn, subjects, y, pa, pb, n_boot=2000, seed=42):
    """Difference fn(y, pa) - fn(y, pb) with subject-bootstrap CI and two-sided bootstrap p-value."""
    subs, rows = _subject_rows(np.asarray(subjects))
    rng = np.random.default_rng(seed)
    d0 = fn(y, pa) - fn(y, pb)
    ds = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(subs), len(subs))])
        ds.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    ds = np.asarray(ds)
    p = float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(d0), lo=float(lo), hi=float(hi), p=p)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return dict(spearman=float("nan"), pearson=float("nan"))
    return dict(spearman=float(spearmanr(a, b)[0]), pearson=float(pearsonr(a, b)[0]))


def spearman(a, b):
    return corr(a, b)["spearman"]


def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = pred == y
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def risk_coverage(conf, correct):
    """Sort by confidence (high first); risk = error rate among retained. Returns coverage, risk, AURC."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    err = 1.0 - np.asarray(correct, float)[order]
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk, float(risk.mean())
