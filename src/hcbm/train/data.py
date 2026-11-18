"""Load row-aligned embeddings + concept targets for one dataset."""
import json
import os

import h5py
import numpy as np


def emb_path(root, dataset, tag):
    return os.path.join(root, f"data/embeddings/{dataset}_{tag}.h5")


def load(root, dataset, tag, targets="targets.h5", extra=None, subj_norm=False, smooth=0):
    """extra: optional "<file>:<key>" in data/processed/<dataset>/ (e.g. "spectral_full.h5:feats") appended to E (D28).
    subj_norm: z-score the embedding per subject (statistics over that subject's non-rejected windows; unsupervised,
      uses no labels) (D29). smooth: centred moving average of E over k neighbouring windows of the same (subject, trial)
      clip (D29). Order: subj_norm on the embedding -> append extra -> smooth.
    Return dict: E (n, D) float32, subject (str), trial, window_idx, reject, l1, l1_bins, l2, l2_mask, l2_train_mask, l3,
    emotion (-1 = none), plus meta and ckpt_sha256 ("" = placeholder encoder)."""
    ep = emb_path(root, dataset, tag)
    tp = os.path.join(root, f"data/processed/{dataset}/{targets}")
    with h5py.File(ep, "r") as g, h5py.File(tp, "r") as t:
        n = g["emb"].shape[0]
        assert int(g.attrs["done"]) == n, f"{ep}: embedding incomplete {g.attrs['done']}/{n}"
        assert t["subject_id"].shape[0] == n, f"{dataset}: embedding/targets row mismatch"
        for k in ("subject_id", "trial_id", "window_idx"):
            assert (g[k][:] == t[k][:]).all(), f"{dataset}: {k} not aligned"
        D = dict(E=g["emb"][:].reshape(n, -1).astype(np.float32), subject=t["subject_id"][:].astype(str),
                 trial=t["trial_id"][:], window_idx=t["window_idx"][:], reject=t["reject"][:], l1=t["l1_z"][:].astype(np.float32),
                 l1_bins=t["l1_bins"][:], l2=t["l2"][:].astype(np.float32), l2_mask=t["l2_mask"][:],
                 l3=t["l3"][:].astype(np.float32), emotion=t["emotion"][:].astype(int),
                 meta=json.loads(t.attrs["meta"]), concept_names=json.loads(t.attrs["concept_names"]),
                 ckpt_sha256=str(g.attrs["ckpt_sha256"]), scale=float(g.attrs["scale"]))
        D["l2_train_mask"] = t["l2_train_mask"][:] if "l2_train_mask" in t else D["l2_mask"]
        for k in ("fine", "rater_r"):
            if k in t:
                D[k] = t[k][:]
    if subj_norm:
        D["E"] = subject_zscore(D["E"], D["subject"], ~D["reject"])
    if extra:
        fn, key = extra.split(":")
        with h5py.File(os.path.join(root, f"data/processed/{dataset}/{fn}"), "r") as x:
            assert (x["subject_id"][:].astype(str) == D["subject"]).all(), f"{dataset}: {fn} rows not aligned"
            D["E"] = np.concatenate([D["E"], x[key][:].astype(np.float32)], 1)
    if smooth and smooth > 1:
        D["E"] = smooth_within_clip(D["E"], D["subject"], D["trial"], D["window_idx"], smooth)
    return D


def subject_zscore(E, subjects, mask):
    out = np.empty_like(E)
    for s in np.unique(subjects):
        r = subjects == s
        use = r & mask if (r & mask).any() else r
        out[r] = (E[r] - E[use].mean(0)) / (E[use].std(0) + 1e-6)
    return out


def smooth_within_clip(E, subjects, trials, widx, k):
    """Centred moving average over k windows inside each (subject, trial) clip, edges averaged over available windows."""
    out = np.empty_like(E)
    key = np.char.add(subjects.astype(str), "|" + np.asarray(trials).astype(str))
    for u in np.unique(key):
        r = np.flatnonzero(key == u)
        r = r[np.argsort(widx[r], kind="stable")]
        c = np.cumsum(np.vstack([np.zeros((1, E.shape[1]), E.dtype), E[r]]), 0)
        h = k // 2
        lo = np.clip(np.arange(len(r)) - h, 0, len(r)); hi = np.clip(np.arange(len(r)) + h + 1, 0, len(r))
        out[r] = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out
