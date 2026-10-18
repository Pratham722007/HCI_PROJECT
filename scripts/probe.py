"""Phase 3.4 / Gate 3: logistic-regression probe on frozen FACED embeddings (train/val subjects only; test group never read).

Run: .venv/bin/python -E -P scripts/probe.py data/embeddings/faced_12ch.h5 -> reports/probe_<name>.json
Targets: binary valence (>3.5), binary arousal (>3.5), coarse 7-class. Metrics: accuracy + macro-F1 vs majority baseline,
mean +- std over the 5 subject-wise folds. Features: 2048-d (4x512) standardised.
"""
import json, os, sys
import h5py, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emb_path = sys.argv[1]
sp = json.load(open(os.path.join(ROOT, "data/processed/faced/splits.json")))
with h5py.File(emb_path) as g, h5py.File(os.path.join(ROOT, "data/processed/faced/windows_12ch.h5")) as w:
    n = g["emb"].shape[0]; done = int(g.attrs["done"]); assert done == n, f"embedding incomplete {done}/{n}"
    E = g["emb"][:].reshape(n, -1); sub = w["subject_id"][:n].astype(str); rej = w["reject"][:n]
    L = {k: w["labels"][k][:n] for k in ("valence_raw", "arousal_raw", "coarse")}
    assert (w["subject_id"][:n] == g["subject_id"][:]).all(), "rows not aligned"
    scale = float(g.attrs["scale"])
test = set(sp["test"]); keep = ~rej & ~np.isin(sub, list(test))
T = {"valence": (L["valence_raw"] > 3.5).astype(int), "arousal": (L["arousal_raw"] > 3.5).astype(int), "coarse7": L["coarse"].astype(int)}
res = {}
for name, y in T.items():
    accs, f1s, base = [], [], []
    for fold in sp["folds"]:
        va = keep & np.isin(sub, fold); tr = keep & ~np.isin(sub, fold)
        if va.sum() == 0 or tr.sum() == 0: continue
        sc = StandardScaler().fit(E[tr]); clf = LogisticRegression(max_iter=300, C=0.1).fit(sc.transform(E[tr]), y[tr])
        p = clf.predict(sc.transform(E[va]))
        accs.append((p == y[va]).mean()); f1s.append(f1_score(y[va], p, average="macro"))
        base.append((y[va] == np.bincount(y[tr]).argmax()).mean())
    res[name] = dict(acc=[float(np.mean(accs)), float(np.std(accs))], macro_f1=[float(np.mean(f1s)), float(np.std(f1s))],
                     majority_acc=float(np.mean(base)), chance_f1=1.0 / len(np.unique(y)), folds=len(accs))
    print(name, res[name], flush=True)
out = os.path.join(ROOT, "reports", f"probe_{os.path.basename(emb_path).replace('.h5', '')}.json")
json.dump(dict(embedding=emb_path, scale=scale, results=res), open(out, "w"), indent=1)
