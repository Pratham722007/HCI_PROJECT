"""Load row-aligned embeddings + concept targets for one dataset."""
import json
import os

import h5py
import numpy as np


def emb_path(root, dataset, tag):
    return os.path.join(root, f"data/embeddings/{dataset}_{tag}.h5")


def load(root, dataset, tag):
    """Return dict: E (n, D) float32, subject (str), trial, reject, l1, l1_bins, l2, l2_mask, l2_train_mask, l3,
    emotion (-1 = none), plus meta and ckpt_sha256 ("" = placeholder encoder)."""
    ep = emb_path(root, dataset, tag)
    tp = os.path.join(root, f"data/processed/{dataset}/targets.h5")
    with h5py.File(ep, "r") as g, h5py.File(tp, "r") as t:
        n = g["emb"].shape[0]
        assert int(g.attrs["done"]) == n, f"{ep}: embedding incomplete {g.attrs['done']}/{n}"
        assert t["subject_id"].shape[0] == n, f"{dataset}: embedding/targets row mismatch"
        for k in ("subject_id", "trial_id", "window_idx"):
            assert (g[k][:] == t[k][:]).all(), f"{dataset}: {k} not aligned"
        D = dict(E=g["emb"][:].reshape(n, -1).astype(np.float32), subject=t["subject_id"][:].astype(str),
                 trial=t["trial_id"][:], reject=t["reject"][:], l1=t["l1_z"][:].astype(np.float32),
                 l1_bins=t["l1_bins"][:], l2=t["l2"][:].astype(np.float32), l2_mask=t["l2_mask"][:],
                 l3=t["l3"][:].astype(np.float32), emotion=t["emotion"][:].astype(int),
                 meta=json.loads(t.attrs["meta"]), concept_names=json.loads(t.attrs["concept_names"]),
                 ckpt_sha256=str(g.attrs["ckpt_sha256"]), scale=float(g.attrs["scale"]))
        D["l2_train_mask"] = t["l2_train_mask"][:] if "l2_train_mask" in t else D["l2_mask"]
        for k in ("fine", "rater_r"):
            if k in t:
                D[k] = t[k][:]
    return D
