"""Is there emotion signal in the EEG at all? Compare feature sets with the same subject-wise logistic probe.

Run: .venv/bin/python -E -P scripts/feature_check.py [datasets...]   (default: primary dataset + faced if built)
     -> reports/feature_check.md + .json
Feature sets (all from the processed 12-ch windows; train/val subjects only, the test group is never read):
  eegpt      frozen EEGPT embedding (run tag dens12), standardised + PCA 256
  band37     Level-1 concepts: log theta/alpha/gamma power x 12 ch + frontal alpha asymmetry, per-subject z-score
  spectral   log power theta/alpha/beta/gamma x 12 ch, relative power, 5 hemispheric asymmetry pairs x 4 bands,
             per-subject z-score (116 features)
  eegpt+spectral  both concatenated
Probe: StandardScaler + LogisticRegression (C=0.1, class-balanced), 5-fold GroupKFold over subjects; macro-F1 mean +- std.
"""
import json, os, sys, time
import h5py, numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.concepts import level1  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": tuple(cfg["band_edges"]["gamma"])}
PAIRS = [("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"), ("T7", "T8"), ("P7", "P8")]
EMB_TAG = "dens12"
DS = sys.argv[1:] or [cfg["primary_dataset"]] + (["faced"] if os.path.exists(os.path.join(ROOT, "data/embeddings/faced_dens12.h5")) else [])


def spectral(X, ch, fs):
    f, p = welch(X, fs=fs, nperseg=fs, noverlap=fs // 2, axis=-1)                 # (n, C, F)
    bp = np.stack([p[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS.values()], 1)   # (n, 4, C)
    logp = np.log(bp + 1e-12)
    rel = np.log(bp / bp.sum(1, keepdims=True) + 1e-12)
    asym = np.stack([logp[:, :, ch.index(r)] - logp[:, :, ch.index(l)] for l, r in PAIRS], 2)  # (n, 4, 5)
    return np.concatenate([logp.reshape(len(X), -1), rel.reshape(len(X), -1), asym.reshape(len(X), -1)], 1)


def probe(Xf, y, g, pca=False):
    steps = [StandardScaler()] + ([PCA(256, random_state=0)] if pca else []) + \
            [LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")]
    f1 = []
    for tr, va in GroupKFold(5).split(Xf, y, g):
        p = make_pipeline(*steps).fit(Xf[tr], y[tr]).predict(Xf[va])
        f1.append(f1_score(y[va], p, average="macro"))
    return float(np.mean(f1)), float(np.std(f1))


res, lines = {}, ["# Feature check: is there emotion signal in the 12-channel EEG?\n",
                  "Same subject-wise 5-fold logistic probe on every feature set; train/val subjects only (test group untouched). "
                  "Macro-F1 mean +- std over folds; class-balanced weights.\n"]
for ds in DS:
    t0 = time.time()
    sp = json.load(open(os.path.join(ROOT, f"data/processed/{ds}/splits.json")))
    with h5py.File(os.path.join(ROOT, f"data/processed/{ds}/windows_12ch.h5"), "r") as f:
        sub = f["subject_id"][:].astype(str); keep = ~f["reject"][:] & ~np.isin(sub, sp["test"])
        idx = np.flatnonzero(keep); ch = [c.decode() for c in f["ch_names"][:]]
        lab = {k: f["labels"][k][:][idx] for k in ("coarse", "valence_raw", "arousal_raw")}
        S = np.concatenate([spectral(f["X"][idx[i:i + 2048]], ch, cfg["fs"]) for i in range(0, len(idx), 2048)])
        B = np.concatenate([level1.band_log_power(f["X"][idx[i:i + 2048]], cfg["fs"], cfg["band_edges"]) for i in range(0, len(idx), 2048)])
    g = sub[idx]
    B = np.concatenate([B, level1.faa(B, ch)[:, None]], 1)
    S, B = level1.zscore_by_subject(S, g), level1.zscore_by_subject(B, g)
    with h5py.File(os.path.join(ROOT, f"data/embeddings/{ds}_{EMB_TAG}.h5"), "r") as e:
        E = e["emb"][:].reshape(e["emb"].shape[0], -1)[idx]
    thr = cfg["rating_thresholds"][ds]
    tasks = {"emotion": (lab["coarse"].astype(int), 1 / len(np.unique(lab["coarse"])), np.ones(len(idx), bool)),
             "valence": ((lab["valence_raw"] > thr["valence"]).astype(int), 0.5, lab["valence_raw"] != thr["valence"]),
             "arousal": ((lab["arousal_raw"] > thr["arousal"]).astype(int), 0.5, lab["arousal_raw"] != thr["arousal"])}
    feats = {"eegpt": (E, True), "band37": (B, False), "spectral": (S, False), "eegpt+spectral": (np.concatenate([E, S], 1), True)}
    res[ds] = {}
    lines += [f"## {ds.upper()} ({len(set(g))} train/val subjects, {len(idx)} windows)\n",
              "| features | dims | " + " | ".join(f"{t} (chance {c:.3f})" for t, (_, c, _) in tasks.items()) + " |",
              "|---|---|" + "---|" * len(tasks)]
    for fn, (Xf, pca) in feats.items():
        cells = []
        for t, (y, _, m) in tasks.items():
            mu, sd = probe(Xf[m], y[m], g[m], pca)
            res[ds].setdefault(fn, {})[t] = [mu, sd]; cells.append(f"{mu:.3f} +- {sd:.3f}")
        lines.append(f"| {fn} | {Xf.shape[1]} | " + " | ".join(cells) + " |")
        print(ds, fn, res[ds][fn], flush=True)
    lines.append(f"\n({time.time() - t0:.0f}s)\n")
lines.append("If hand-made spectral features beat EEGPT, the signal exists in the EEG and the frozen 12-channel EEGPT embedding "
             "is the bottleneck; if nothing beats chance, the labels/windows themselves carry little decodable signal.\n")
open(os.path.join(ROOT, "reports/feature_check.md"), "w").write("\n".join(lines))
json.dump(res, open(os.path.join(ROOT, "reports/feature_check.json"), "w"), indent=1)
print("wrote reports/feature_check.md")
