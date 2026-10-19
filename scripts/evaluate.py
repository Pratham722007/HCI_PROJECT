"""Phases 5-7 reports from saved predictions: in-corpus (primary dataset, rev 3 = DENS), Sim 1 (zero-shot DEAP/DREAMER),
ablation.

Run: .venv/bin/python -E -P scripts/evaluate.py --tag dens12   -> reports/<tag>/{in_corpus,sim1,sim_ablation}.md + .csv
Main numbers use the 5-fold ensemble (mean of fold-model outputs) with subject-bootstrap 95% CIs; fold mean +- std
is given alongside. Nothing here selects anything: all thresholds were fixed before (0.5 on the 0-1 rating scale).
V/A from the emotion head ("via emotion") = class probabilities x per-class mean train-subject valence/arousal, so the
flat baseline gets V/A predictions in the same way as the CBM.
"""
import argparse, csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.eval import metrics as M  # noqa: E402
from hcbm.models.confidence import softmax  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="dens12"); ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()
man = runs.manifest(ROOT, a.tag)
out_dir = os.path.join(ROOT, "reports", a.tag); os.makedirs(out_dir, exist_ok=True)
PRIMARY = man.get("primary", "faced"); NAME = PRIMARY.upper()
sp = json.load(open(os.path.join(ROOT, f"data/processed/{PRIMARY}/splits.json")))
D = {ds: tdata.load(ROOT, ds, a.tag) for ds in [PRIMARY] + man.get("cross_corpus", [])}
F = D[PRIMARY]
COARSE = F["meta"]["coarse_classes"]
variants = [v for v in runs.ALL_VARIANTS if runs.load_preds(ROOT, a.tag, v)]
P = {v: runs.load_preds(ROOT, a.tag, v) for v in variants}
LC = list(range(len(COARSE)))
train_rows = np.flatnonzero(~F["reject"] & ~np.isin(F["subject"], sp["test"]))
proto = np.stack([np.nanmean(F["l2"][train_rows][F["emotion"][train_rows] == c, :2], 0) for c in LC])   # (classes, 2) V/A
bad_raters = set(man.get("unreliable_raters") or [])


def boot(fn, subj, *arr):
    return M.subject_bootstrap(fn, subj, *arr, n_boot=a.n_boot)


def fmt(r):
    return f"{r['est']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def binary(true01, pred):
    m = np.abs(true01 - 0.5) > 1e-9                       # exact midpoint ties dropped (D4)
    return m, (true01 > 0.5).astype(int), (pred > 0.5).astype(int)


def ensemble(v, ds, key):
    return np.mean([p[ds][key] for p in P[v]], 0)


def probs(v, ds):
    return np.mean([softmax(p[ds]["logits"]) for p in P[v]], 0)


def fold_stat(v, ds, fn):
    s = [fn(p[ds]) for p in P[v]]
    return f"{np.mean(s):.3f} +- {np.std(s):.3f}"


rows_csv = []
banner = runs.placeholder_banner(man)

# ---------------- in-corpus (Gate 5) ----------------
tr = P[variants[0]][0]["test"]["rows"]
yt, st = F["emotion"][tr], F["subject"][tr]
lines = [f"# In-corpus {NAME} results (tag `{a.tag}`)\n", banner,
         f"Held-out test group: {len(set(st))} subjects, {len(tr)} non-rejected windows. {len(COARSE)}-class coarse emotion ({', '.join(COARSE)}). "
         "Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.\n",
         "| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |",
         "|---|---|---|---|---|---|"]
for v in variants:
    pr = probs(v, "test").argmax(1)
    s = json.load(open(os.path.join(runs.run_dir(ROOT, a.tag, v), "summary.json")))
    cv = np.mean([f["val_macro_f1"] for f in s["folds"]])
    r_acc, r_f1 = boot(M.acc, st, yt, pr), boot(lambda y, p: M.macro_f1(y, p, LC), st, yt, pr)
    lines.append(f"| {v} | {fmt(r_acc)} | {fmt(r_f1)} | {fold_stat(v, 'test', lambda d: M.macro_f1(yt, d['logits'].argmax(1), LC))} | {cv:.3f} | {s['n_params']:,} |")
    rows_csv.append(dict(section="in_corpus", model=v, dataset="test", metric="macro_f1", **r_f1))
    rows_csv.append(dict(section="in_corpus", model=v, dataset="test", metric="acc", **r_acc))
lines.append(f"\nChance macro-F1 ~ {1 / len(LC):.3f}; majority-class accuracy {np.bincount(yt).max() / len(yt):.3f}.\n")
if "flat" in variants and "level123" in variants:
    d = M.paired_bootstrap(lambda y, p: M.macro_f1(y, p, LC), st, yt, probs("level123", "test").argmax(1), probs("flat", "test").argmax(1))
    lines.append(f"**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], p = {d['p']:.3f}. "
                 f"{'Non-inferior (CI lower bound > -0.02).' if d['lo'] > -0.02 else 'Non-inferiority NOT shown (CI lower bound <= -0.02).'}\n")

lines += [f"## Concept accuracy on {NAME} test (CBM variants, ensemble)\n",
          "| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 dominance bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |", "|---|---|---|---|---|---|---|"]
rel = ~np.isin(st, list(bad_raters))
for v in variants:
    if v == "flat":
        continue
    l1, l2, l3 = (ensemble(v, "test", k) for k in ("l1", "l2", "l3"))
    r1 = np.nanmean([M.corr(l1[:, j], F["l1"][tr, j])["pearson"] for j in range(l1.shape[1])])
    va = []
    for d_ in (0, 1, 2):
        m, y, p = binary(F["l2"][tr, d_], l2[:, d_]); m &= F["l2_mask"][tr, d_]
        va.append(M.acc(y[m], p[m]) if m.any() else float("nan"))
    m, y, p = binary(F["l2"][tr, 0], l2[:, 0]); m = m & rel
    lines.append(f"| {v} | {r1:.3f} | {va[0]:.3f} | {va[1]:.3f} | {va[2]:.3f} | {M.acc(y[m], p[m]):.3f} | {np.nanmean(np.abs(l3 - F['l3'][tr])):.3f} |")
lines.append(f"\nUnreliable raters in the test group (D11, FACED only; V/A loss masked in training; included above unless stated): "
             f"{sorted(bad_raters & set(sp['test'])) or 'none'}. Dominance: "
             f"{'FACED pseudo-label (D3)' if PRIMARY == 'faced' else 'real self-rating'}.\n")
open(os.path.join(out_dir, "in_corpus.md"), "w").write("\n".join(lines))

# ---------------- Sim 1: zero-shot cross-corpus ----------------
def cross_rows(ds):
    X = D[ds]; ok = ~X["reject"]
    return X, ok


sim, notbetter = [], []
for ds in man.get("cross_corpus", []):
    X, ok = cross_rows(ds)
    subj = X["subject"]
    sim += [f"\n## {ds.upper()} (zero-shot, {len(set(subj))} subjects, {ok.sum()} non-rejected windows)\n",
            "| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |",
            "|---|---|---|---|---|---|---|---|"]
    preds = {}
    for v in variants:
        pv = probs(v, ds) @ proto                                   # via emotion: (n, 2)
        readouts = [("via emotion", pv)]
        if v != "flat":
            readouts.insert(0, ("L2 concept", ensemble(v, ds, "l2")[:, :2]))
        for name, pr in readouts:
            preds[(v, name)] = pr
            cells = []
            for d_ in (0, 1):
                m, y, p = binary(X["l2"][:, d_], pr[:, d_]); m &= ok
                r = boot(M.acc, subj[m], y[m], p[m])
                cells += [fmt(r), f"{M.macro_f1(y[m], p[m]):.3f}"]
                rows_csv.append(dict(section="sim1", model=v, dataset=ds, metric=f"{'VA'[d_]}_acc_{name}", **r))
            for d_ in (0, 1):
                r = boot(M.spearman, subj[ok], X["l2"][ok, d_], pr[ok, d_]); cells.append(fmt(r))
                rows_csv.append(dict(section="sim1", model=v, dataset=ds, metric=f"{'VA'[d_]}_spearman_{name}", **r))
            sim.append(f"| {v} | {name} | " + " | ".join([cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]]) + " |")
    if ds == "deap":
        sim += ["\nDEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):\n", "| model | readout | quadrant acc [95% CI] |", "|---|---|---|"]
        mv, yv, _ = binary(X["l2"][:, 0], X["l2"][:, 0]); ma, ya, _ = binary(X["l2"][:, 1], X["l2"][:, 1])
        m = mv & ma & ok; q = 2 * yv + ya
        for (v, name), pr in preds.items():
            qp = 2 * (pr[:, 0] > 0.5) + (pr[:, 1] > 0.5)
            sim.append(f"| {v} | {name} | {fmt(boot(M.acc, subj[m], q[m], qp[m]))} |")
    if ds == "dreamer":
        m = ok & (X["emotion"] >= 0)
        sim += [f"\nDREAMER coarse-emotion accuracy (**secondary**, clip-level target emotion -> {len(COARSE)} classes, unmapped clips excluded):\n",
                "| model | acc [95% CI] | macro-F1 |", "|---|---|---|"]
        for v in variants:
            p = probs(v, ds).argmax(1)
            present = sorted(set(X["emotion"][m]))
            sim.append(f"| {v} | {fmt(boot(M.acc, subj[m], X['emotion'][m], p[m]))} | {M.macro_f1(X['emotion'][m], p[m], present):.3f} |")
    if "flat" in variants and "level123" in variants:
        sim += ["\nPaired subject bootstrap, CBM(level123) minus flat (accuracy):\n", "| comparison | dim | diff [95% CI] | p |", "|---|---|---|---|"]
        for name in ("L2 concept", "via emotion"):
            for d_ in (0, 1):
                m, y, _ = binary(X["l2"][:, d_], X["l2"][:, d_]); m &= ok
                pa = (preds[("level123", name)][:, d_] > 0.5).astype(int); pb = (preds[("flat", "via emotion")][:, d_] > 0.5).astype(int)
                r = M.paired_bootstrap(M.acc, subj[m], y[m], pa[m], pb[m])
                sim.append(f"| CBM {name} vs flat | {'valence' if d_ == 0 else 'arousal'} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.3f} |")
                if r["diff"] <= 0 or r["lo"] <= 0:
                    notbetter.append(f"{ds} {'valence' if d_ == 0 else 'arousal'} accuracy (CBM {name}: {r['diff']:+.3f}, CI [{r['lo']:+.3f}, {r['hi']:+.3f}])")
    if (D[ds]["l2_mask"][:, 2]).any() and "level123" in variants:
        r = boot(M.spearman, subj[ok], X["l2"][ok, 2], ensemble("level123", ds, "l2")[ok, 2])
        sim.append(f"\nPredicted dominance (trained on {'FACED pseudo-labels, exploratory (D3)' if PRIMARY == 'faced' else NAME + ' self-ratings'}) "
                   f"vs real {ds.upper()} dominance, Spearman {fmt(r)}.\n")

head = [f"# Simulation 1: zero-shot cross-corpus generalisation (tag `{a.tag}`)\n", banner,
        f"Models trained on {NAME} only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. "
        "Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.\n",
        "**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** " +
        ("; ".join(notbetter) if notbetter else "none") + "\n"]
open(os.path.join(out_dir, "sim1.md"), "w").write("\n".join(head + sim))

# ---------------- ablation ----------------
ab = [f"# Ablation (tag `{a.tag}`)\n", banner,
      f"Same splits/seed/hyperparameters for every variant. {NAME} = test macro-F1 (ensemble). Cross-corpus = valence/arousal "
      "accuracy via the emotion head (comparable across all variants, incl. flat).\n",
      f"| variant | {NAME} test macro-F1 | " + " | ".join(f"{ds} V acc | {ds} A acc" for ds in man.get("cross_corpus", [])) + " |",
      "|---|---|" + "---|---|" * len(man.get("cross_corpus", []))]
for v in variants:
    cells = [f"{M.macro_f1(yt, probs(v, 'test').argmax(1), LC):.3f}"]
    for ds in man.get("cross_corpus", []):
        X, ok = cross_rows(ds); pv = probs(v, ds) @ proto
        for d_ in (0, 1):
            m, y, p = binary(X["l2"][:, d_], pv[:, d_]); m &= ok; cells.append(f"{M.acc(y[m], p[m]):.3f}")
    ab.append(f"| {v} | " + " | ".join(cells) + " |")
ab.append("\n`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the "
          "embedding trunk) are labelled variants, not the main model.\n")
open(os.path.join(out_dir, "sim_ablation.md"), "w").write("\n".join(ab))

with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["section", "model", "dataset", "metric", "est", "lo", "hi"]); w.writeheader(); w.writerows(rows_csv)
print("wrote", out_dir)
