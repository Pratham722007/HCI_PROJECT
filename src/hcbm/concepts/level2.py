"""Level 2 affective concepts: valence, arousal (real self-ratings), dominance (real on DENS/DEAP/DREAMER, pseudo on FACED).

All three are rescaled to 0-1 per dataset so 0.5 is the locked binarisation midpoint on every corpus
(FACED 3.5/7, DENS and DEAP 5 on 1-9, DREAMER 3 on 1-5; D4).
"""
import numpy as np

SCALES = {"faced": (0.0, 7.0), "dens": (1.0, 9.0), "deap": (1.0, 9.0), "dreamer": (1.0, 5.0)}
DIMS = ("valence", "arousal", "dominance")


def rescale(x, dataset):
    lo, hi = SCALES[dataset]
    return (np.asarray(x, float) - lo) / (hi - lo)


def rater_reliability(valence, subjects, trials, norm_subjects):
    """Per subject: Pearson r between their per-clip valence and the per-clip mean of the OTHER norm subjects.

    norm_subjects: subjects allowed to define the clip norm (FACED train/val only, never the test group).
    Returns {subject: r}.
    """
    subs = np.unique(subjects)
    clips = np.unique(trials)
    M = np.full((len(subs), len(clips)), np.nan)
    for i, s in enumerate(subs):
        for j, c in enumerate(clips):
            m = (subjects == s) & (trials == c)
            if m.any():
                M[i, j] = valence[m][0]
    is_norm = np.isin(subs, list(norm_subjects))
    out = {}
    for i, s in enumerate(subs):
        others = is_norm.copy(); others[i] = False
        out[str(s)] = float(np.corrcoef(M[i], np.nanmean(M[others], 0))[0, 1])
    return out


def build_level2(dataset, labels, fine=None, dominance_prototype=None):
    """labels: dict with valence_raw, arousal_raw[, dominance_raw]. Returns (values (n,3), mask (n,3) bool, pseudo flag)."""
    n = len(labels["valence_raw"])
    V = np.full((n, 3), np.nan)
    M = np.zeros((n, 3), bool)
    V[:, 0] = rescale(labels["valence_raw"], dataset)
    V[:, 1] = rescale(labels["arousal_raw"], dataset)
    M[:, :2] = True
    if "dominance_raw" in labels:
        V[:, 2] = rescale(labels["dominance_raw"], dataset); M[:, 2] = True
    elif fine is not None and dominance_prototype is not None:
        V[:, 2] = np.asarray(dominance_prototype)[fine]; M[:, 2] = True            # pseudo-label (D3)
    return V, M
