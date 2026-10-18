"""Write data/processed/faced/splits.json (hard rule: subject-wise, fixed before any training)."""
import json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.train.splits import make_splits, assert_disjoint  # noqa: E402
from hcbm.utils.config import load_yaml_config  # noqa: E402
cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
with h5py.File(os.path.join(ROOT, "data/processed/faced/windows_12ch.h5")) as f:
    subs = f["subject_id"][:].astype(str)
sp = make_splits(subs, cfg["split_seeds"]["default"], cfg["split_seeds"]["held_out_test_ratio"], cfg["split_seeds"]["n_folds"])
assert_disjoint(sp)
json.dump(sp, open(os.path.join(ROOT, "data/processed/faced/splits.json"), "w"), indent=1)
print("test", len(sp["test"]), "folds", [len(x) for x in sp["folds"]])
