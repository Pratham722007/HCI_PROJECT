"""Unified HDF5 schema writer/validator.

Datasets: X [n, C, T] float32; ch_names, fs, subject_id (str), trial_id (int), window_idx (int),
reject (bool, True = artifact), labels/<name> arrays (n,) or (n,k). meta = JSON provenance attr.
"""

import json
import h5py
import numpy as np

REQUIRED = ["X", "ch_names", "subject_id", "trial_id", "window_idx", "reject"]


def write_h5(path, X, ch_names, fs, subject_id, trial_id, window_idx, reject, labels, meta):
    n = X.shape[0]
    for name, a in [("subject_id", subject_id), ("trial_id", trial_id), ("window_idx", window_idx), ("reject", reject)]:
        assert len(a) == n, f"{name} length {len(a)} != {n}"
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.asarray(X, dtype="float32"), compression="gzip", compression_opts=1, chunks=(1,) + X.shape[1:])
        f.create_dataset("ch_names", data=np.array(ch_names, dtype="S"))
        f.create_dataset("subject_id", data=np.array(subject_id, dtype="S"))
        f.create_dataset("trial_id", data=np.asarray(trial_id, dtype="int32"))
        f.create_dataset("window_idx", data=np.asarray(window_idx, dtype="int32"))
        f.create_dataset("reject", data=np.asarray(reject, dtype="bool"))
        f.attrs["fs"] = fs
        g = f.create_group("labels")
        for k, v in labels.items():
            v = np.asarray(v)
            g.create_dataset(k, data=v.astype("S") if v.dtype.kind in "US" else v)
        f.attrs["meta"] = json.dumps(meta)


def validate_h5(path, expect_channels=None, expect_samples=None):
    """Raise AssertionError on schema violations; return summary dict."""
    with h5py.File(path, "r") as f:
        for k in REQUIRED:
            assert k in f, f"missing {k}"
        n, c, t = f["X"].shape
        assert f["X"].dtype == np.float32
        for k in REQUIRED[2:]:
            assert f[k].shape[0] == n, f"{k} row mismatch"
        assert len(f["ch_names"]) == c
        assert "fs" in f.attrs and "meta" in f.attrs
        for k in f["labels"]:
            assert f["labels"][k].shape[0] == n, f"label {k} row mismatch"
        if expect_channels is not None:
            assert [s.decode() for s in f["ch_names"][:]] == list(expect_channels), "channel order mismatch"
        if expect_samples is not None:
            assert t == expect_samples, f"samples {t} != {expect_samples}"
        # finite check in chunks
        for i in range(0, n, 2048):
            assert np.isfinite(f["X"][i:i + 2048]).all(), "non-finite values in X"
        return dict(n=n, channels=c, samples=t, n_reject=int(f["reject"][:].sum()),
                    n_subjects=len(set(f["subject_id"][:].tolist())))
