"""Level 3 appraisal concepts: category-level pseudo-labels from configs/appraisal_profiles.yaml (always pseudo-labels)."""
import numpy as np

DIMS = ("novelty", "goal_conduciveness", "coping")


def profile_table(profiles, fine_classes, section="faced"):
    """(n_fine, 3) appraisal matrix and (n_fine,) dominance prototypes, rows in `fine_classes` order."""
    rows = profiles[section]
    A = np.array([[rows[c][d] for d in DIMS] for c in fine_classes], dtype=float)
    dom = np.array([rows[c]["dominance_prototype"] for c in fine_classes], dtype=float)
    return A, dom


def build_level3(fine, table):
    """fine (n,) int category index, -1 = unknown -> (n,3) values (NaN where unknown)."""
    fine = np.asarray(fine)
    out = np.full((len(fine), 3), np.nan)
    ok = fine >= 0
    out[ok] = table[fine[ok]]
    return out
