"""Write data/processed/<dataset>/splits.json (hard rule: subject-wise, fixed before any training).

Run: .venv/bin/python -E -P scripts/make_splits.py [dataset]   (default: protocol primary_dataset)
"""
import json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.train.splits import make_splits, assert_disjoint  # noqa: E402
from hcbm.utils.config import load_yaml_config  # noqa: E402
cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
ds = sys.argv[1] if len(sys.argv) > 1 else cfg["primary_dataset"]
with h5py.File(os.path.join(ROOT, f"data/processed/{ds}/windows_12ch.h5")) as f:
    subs = f["subject_id"][:].astype(str)
sp = make_splits(subs, cfg["split_seeds"]["default"], cfg["split_seeds"]["held_out_test_ratio"], cfg["split_seeds"]["n_folds"])
assert_disjoint(sp)
sp["dataset"] = ds
json.dump(sp, open(os.path.join(ROOT, f"data/processed/{ds}/splits.json"), "w"), indent=1)
cnt = dict(zip(*np.unique(subs, return_counts=True)))
print(ds, "test", len(sp["test"]), "subjects /", int(sum(cnt[s] for s in sp["test"])), "windows; folds",
      [(len(x), int(sum(cnt[s] for s in x))) for x in sp["folds"]])
