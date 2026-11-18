"""Rev 3c (D28): spectral features -> data/processed/<ds>/spectral_full.h5, row-aligned with windows_12ch.h5.

Run: .venv/bin/python -E -P scripts/build_spectral.py [faced dens deap dreamer]
  feats   from windows_full.h5 (30 ch) when it exists, else windows_12ch.h5 (DREAMER)
  feats12 from windows_12ch.h5 only (the 12-channel view, used to train robustness to DREAMER's montage)
Both on the canonical 30-channel layout (missing channels = 0 after per-subject z-score; statistics from non-rejected rows).
"""
import json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.concepts import spectral  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
CAN = cfg["full_channels"]


def compute(path, sub, keep):
    with h5py.File(path, "r") as f:
        ch = [c.decode() for c in f["ch_names"][:]]
        V = np.concatenate([spectral.features(f["X"][i:i + 2048], ch, CAN, cfg["fs"], cfg) for i in range(0, f["X"].shape[0], 2048)])
    return spectral.zscore_subject_nan(V, sub, keep).astype(np.float32), ch


for ds in sys.argv[1:] or [cfg["primary_dataset"]] + cfg["cross_corpus_datasets"]:
    p12, pf = (os.path.join(ROOT, f"data/processed/{ds}/windows_{m}.h5") for m in ("12ch", "full"))
    with h5py.File(p12, "r") as f:
        sub = f["subject_id"][:].astype(str); keep = ~f["reject"][:]; ids = {k: f[k][:] for k in ("subject_id", "trial_id", "window_idx")}
    f12, _ = compute(p12, sub, keep)
    ff, chf = compute(pf, sub, keep) if os.path.exists(pf) else (f12, "12ch")
    with h5py.File(os.path.join(ROOT, f"data/processed/{ds}/spectral_full.h5"), "w") as g:
        for k, v in ids.items():
            g.create_dataset(k, data=v)
        g.create_dataset("feats", data=ff); g.create_dataset("feats12", data=f12)
        g.attrs["feature_names"] = json.dumps(spectral.feature_names(CAN, cfg))
        g.attrs["meta"] = json.dumps(dict(dataset=ds, feats_source="windows_full.h5" if os.path.exists(pf) else "windows_12ch.h5"))
    print(ds, ff.shape, "missing-channel share (feats):", round(float((ff == 0).mean()), 3), flush=True)
