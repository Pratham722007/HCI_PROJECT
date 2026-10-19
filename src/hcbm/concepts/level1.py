"""Level 1 neurophysiological concepts: Welch band power per channel (theta, alpha, gamma) + frontal alpha asymmetry.

Computed from the preprocessed 12-ch windows only (never from model outputs); identical code for every dataset.
Concept order: [band0_ch0 .. band0_ch11, band1_ch0 .., band2_ch11, FAA] -> 12 x 3 + 1 = 37.
"""
import numpy as np
from scipy.signal import welch

BANDS = ("theta", "alpha", "gamma")
TERTILE_Z = 0.4307                       # +-z of the standard-normal tertile edges (bin boundaries for uncertainty)


def concept_names(ch_names, bands=BANDS):
    return [f"{b}_{c}" for b in bands for c in ch_names] + ["FAA"]


def band_log_power(X, fs, band_edges, bands=BANDS):
    """X (n, C, T) -> log band power (n, len(bands) * C) using Welch (1 s Hann segments, 50% overlap)."""
    f, pxx = welch(X, fs=fs, nperseg=int(fs), noverlap=int(fs) // 2, axis=-1)      # (n, C, F)
    df = f[1] - f[0]
    out = []
    for b in bands:
        lo, hi = band_edges[b]
        m = (f >= lo) & (f < hi)
        out.append(np.log(pxx[..., m].sum(-1) * df + 1e-12))
    return np.concatenate(out, axis=1)


def faa(logp, ch_names, left="F3", right="F4", bands=BANDS):
    """ln(alpha right) - ln(alpha left), from the log-power matrix returned by band_log_power."""
    C = len(ch_names)
    a = bands.index("alpha") * C
    return logp[:, a + ch_names.index(right)] - logp[:, a + ch_names.index(left)]


def zscore_by_subject(V, subjects, mask=None):
    """Per-subject, per-column z-score; statistics from rows where mask is True (e.g. non-rejected windows)."""
    Z = np.empty_like(V, dtype=np.float64)
    mask = np.ones(len(V), bool) if mask is None else mask
    for s in np.unique(subjects):
        r = subjects == s
        use = r & mask if (r & mask).any() else r
        mu, sd = V[use].mean(0), V[use].std(0)
        Z[r] = (V[r] - mu) / (sd + 1e-8)
    return Z


def tertile_bins(Z, subjects, mask=None):
    """Per-subject tertile bin (0 low, 1 mid, 2 high) of each column."""
    B = np.empty(Z.shape, dtype=np.int8)
    mask = np.ones(len(Z), bool) if mask is None else mask
    for s in np.unique(subjects):
        r = subjects == s
        use = r & mask if (r & mask).any() else r
        q1, q2 = np.quantile(Z[use], [1 / 3, 2 / 3], axis=0)
        B[r] = (Z[r] > q1).astype(np.int8) + (Z[r] > q2).astype(np.int8)
    return B


def compute_level1(X, subjects, ch_names, fs, band_edges, faa_pair=("F3", "F4"), mask=None):
    """Return dict(raw (n,37) log power + FAA, z (n,37) per-subject z, bins (n,37) tertiles)."""
    lp = band_log_power(X, fs, band_edges)
    raw = np.concatenate([lp, faa(lp, list(ch_names), *faa_pair)[:, None]], axis=1)
    z = zscore_by_subject(raw, subjects, mask)
    return dict(raw=raw, z=z, bins=tertile_bins(z, subjects, mask))
