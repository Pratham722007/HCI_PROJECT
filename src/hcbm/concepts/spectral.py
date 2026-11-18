"""Hand-made spectral features on a canonical channel layout (D28): log band power and relative power (theta, alpha, beta,
gamma) per channel + left-right asymmetry for 13 homologous pairs. Missing channels give NaN, which become 0 after the
per-subject z-score (= "typical"), so every dataset yields the same feature vector length."""
import numpy as np
from scipy.signal import welch

PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("FC1", "FC2"), ("FC5", "FC6"), ("C3", "C4"), ("T7", "T8"),
         ("CP1", "CP2"), ("CP5", "CP6"), ("P3", "P4"), ("P7", "P8"), ("PO3", "PO4"), ("O1", "O2")]


def bands(cfg):
    return {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": tuple(cfg["band_edges"]["gamma"])}


def feature_names(canonical, cfg):
    b = list(bands(cfg))
    return ([f"logp_{x}_{c}" for x in b for c in canonical] + [f"rel_{x}_{c}" for x in b for c in canonical] +
            [f"asym_{x}_{l}-{r}" for x in b for l, r in PAIRS])


def features(X, ch_names, canonical, fs, cfg):
    """X (n, C, T) with channels `ch_names` -> (n, F) on the canonical layout; NaN where a channel is missing."""
    f, p = welch(X, fs=fs, nperseg=int(fs), noverlap=int(fs) // 2, axis=-1)
    bp = np.stack([p[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in bands(cfg).values()], 1)     # (n, 4, C)
    logp, rel = np.log(bp + 1e-12), np.log(bp / bp.sum(1, keepdims=True) + 1e-12)
    n, nb = len(X), bp.shape[1]
    L, R = np.full((n, nb, len(canonical)), np.nan), np.full((n, nb, len(canonical)), np.nan)
    pos = {c: i for i, c in enumerate(ch_names)}
    for j, c in enumerate(canonical):
        if c in pos:
            L[:, :, j], R[:, :, j] = logp[:, :, pos[c]], rel[:, :, pos[c]]
    ci = {c: j for j, c in enumerate(canonical)}
    A = np.stack([L[:, :, ci[r]] - L[:, :, ci[l]] for l, r in PAIRS], 2)                         # (n, 4, 13)
    return np.concatenate([L.reshape(n, -1), R.reshape(n, -1), A.reshape(n, -1)], 1)


def zscore_subject_nan(V, subjects, mask):
    """Per-subject z-score ignoring NaN; NaN (missing channel) -> 0."""
    Z = np.zeros_like(V)
    for s in np.unique(subjects):
        r = subjects == s
        use = r & mask if (r & mask).any() else r
        mu, sd = np.nanmean(V[use], 0), np.nanstd(V[use], 0)
        Z[r] = (V[r] - mu) / (sd + 1e-8)
    return np.nan_to_num(Z, nan=0.0)
