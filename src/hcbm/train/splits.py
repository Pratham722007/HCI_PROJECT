"""Subject-wise splits: fixed held-out test group + K folds over the remaining subjects."""
import numpy as np


def make_splits(subjects, seed=42, test_ratio=0.20, n_folds=5):
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subs)
    n_test = int(round(len(subs) * test_ratio))
    test, rest = sorted(perm[:n_test].tolist()), perm[n_test:]
    folds = [sorted(f.tolist()) for f in np.array_split(rest, n_folds)]       # each fold = validation subjects
    return dict(seed=seed, test=test, folds=folds)


def assert_disjoint(splits):
    seen = set(splits["test"])
    for f in splits["folds"]:
        assert not seen & set(f), "subject appears in two partitions"
        seen |= set(f)
