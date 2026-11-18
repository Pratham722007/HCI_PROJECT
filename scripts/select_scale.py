"""Choose the EEGPT input multiplier (docs/EEGPT_REQUIREMENTS.md, open item) on train/val subjects of the primary dataset only (rev 3 = DENS).

Run: .venv/bin/python -E -P scripts/select_scale.py [--device cuda] [--n-subjects 30]
For each candidate scale, embeds all windows of a fixed random subset of non-test subjects and scores a subject-wise
5-fold logistic probe (coarse-class macro-F1). Writes reports/scale_selection_<dataset>.json; run_all.sh reads `best`.
The test group and DEAP/DREAMER are never touched.
"""
import argparse, json, os, sys
import h5py, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.models.eegpt_encoder import EEGPTEncoder  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--n-subjects", type=int, default=30)
ap.add_argument("--scales", default="0.01,0.1,1,10"); ap.add_argument("--random-weights", action="store_true")
ap.add_argument("--montage", default="12ch", choices=["12ch", "full"])
ap.add_argument("--pca", type=int, default=0, help="PCA dims inside the probe (speed); 0 = none")
a = ap.parse_args()
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml")); DS = cfg["primary_dataset"]
sp = json.load(open(os.path.join(ROOT, f"data/processed/{DS}/splits.json")))
pool = sorted(s for f in sp["folds"] for s in f)
subs = sorted(np.random.default_rng(42).choice(pool, min(a.n_subjects, len(pool)), replace=False).tolist())
with h5py.File(os.path.join(ROOT, f"data/processed/{DS}/windows_{a.montage}.h5"), "r") as f:
    sid = f["subject_id"][:].astype(str)
    rows = np.flatnonzero(np.isin(sid, subs) & ~f["reject"][:])
    X = f["X"][rows[0]:rows[-1] + 1][rows - rows[0]]
    g = sid[rows]; ch = [c.decode() for c in f["ch_names"][:]]
with h5py.File(os.path.join(ROOT, f"data/processed/{DS}/windows_12ch.h5"), "r") as f12:   # labels live in the 12-ch file
    assert (f12["subject_id"][:].astype(str) == sid).all(), "windows_full rows differ from windows_12ch"
    y = f12["labels"]["coarse"][:][rows]
res = {}
for s in map(float, a.scales.split(",")):
    enc = EEGPTEncoder(os.path.join(ROOT, paths["eegpt"]["repo"]), os.path.join(ROOT, paths["eegpt"]["checkpoint"]), ch, 1024, s,
                       a.device, load_weights=not a.random_weights)
    E = np.concatenate([enc(X[i:i + 128]).cpu().numpy().reshape(-1, 2048) for i in range(0, len(X), 128)])
    f1s = []
    for tr, va in GroupKFold(5).split(E, y, g):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        steps = [StandardScaler()] + ([PCA(a.pca, random_state=0)] if a.pca else []) + [LogisticRegression(max_iter=300, C=0.1)]
        p = make_pipeline(*steps).fit(E[tr], y[tr]).predict(E[va])
        f1s.append(f1_score(y[va], p, average="macro"))
    res[s] = float(np.mean(f1s)); print(f"scale {s}: macro-F1 {res[s]:.4f}", flush=True)
best = max(res, key=res.get)
json.dump(dict(dataset=DS, montage=a.montage, pca=a.pca, subjects=subs, scores=res, best=best, ckpt_sha256=enc.ckpt_sha256 or ""),
          open(os.path.join(ROOT, "reports", f"scale_selection_{DS}{'_full' if a.montage == 'full' else ''}.json"), "w"), indent=1)
print("best", best)
