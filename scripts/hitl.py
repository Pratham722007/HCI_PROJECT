"""Phase 9: human-in-the-loop experiments -> reports/<tag>/hitl.md + figures.

Run: .venv/bin/python -E -P scripts/hitl.py --tag 12ch
1. Intervention curves (main CBM, level123): replace 0-100% of one level's predicted concepts with ground truth, random
   vs uncertainty-ordered; FACED test (L1 real band power, L2 real ratings, L3 PSEUDO-expert) and DREAMER (L1, L2 real;
   coarse emotion is secondary). Ensemble of the 5 fold models; CI = subject bootstrap.
2. Escalation: risk-coverage, AURC vs random escalation, error rate escalated vs retained at the threshold fixed on
   FACED validation, ECE before/after temperature scaling.
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.train.trainer import predict  # noqa: E402
from hcbm.models.confidence import softmax  # noqa: E402
from hcbm.hitl.intervention import intervention_mask  # noqa: E402
from hcbm.eval import metrics as M, plots  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="12ch"); ap.add_argument("--variant", default="level123")
ap.add_argument("--n-boot", type=int, default=500)
a = ap.parse_args()
mcfg = load_yaml_config(os.path.join(ROOT, "configs/model.yaml"))
man = runs.manifest(ROOT, a.tag)
out_dir = os.path.join(ROOT, "reports", a.tag); os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
sp = json.load(open(os.path.join(ROOT, "data/processed/faced/splits.json")))
F = tdata.load(ROOT, "faced", a.tag)
DR = tdata.load(ROOT, "dreamer", a.tag) if "dreamer" in man.get("cross_corpus", []) else None
folds = [runs.load_fold(p, mcfg) for p in runs.fold_paths(ROOT, a.tag, a.variant)]
FRACS = np.round(np.linspace(0, 1, 11), 2)
L7 = list(range(7))


def curve(D, rows, level, order):
    E = D["E"][rows]; truth = {"l1": D["l1"], "l2": D["l2"], "l3": D["l3"]}[level][rows]
    ok = ~np.isnan(truth) & (D["l2_mask"][rows] if level == "l2" else True)
    y, subj = D["emotion"][rows], D["subject"][rows]
    res = []
    for frac in FRACS:
        P = 0
        for k, (m, ck) in enumerate(folds):
            base = predict(m, E)[level]
            mask = intervention_mask(level, base, ok, frac, order, np.random.default_rng(100 + k))
            P = P + softmax(predict(m, E, {level: (np.nan_to_num(truth), mask)})["logits"], ck["temperature"])
        r = M.subject_bootstrap(M.acc, subj, y, P.argmax(1), n_boot=a.n_boot)
        res.append(dict(frac=float(frac), **r, macro_f1=M.macro_f1(y, P.argmax(1), sorted(set(y)))))
    return res


lines = [f"# Human-in-the-loop experiments (tag `{a.tag}`, model `{a.variant}`)\n", runs.placeholder_banner(man)]
test_rows = np.flatnonzero(~F["reject"] & np.isin(F["subject"], sp["test"]))
sets = [("FACED test", F, test_rows, ("l1", "l2", "l3"))]
if DR is not None:
    sets.append(("DREAMER (secondary labels)", DR, np.flatnonzero(~DR["reject"] & (DR["emotion"] >= 0)), ("l1", "l2")))
curves = {}
lines += ["## 1. Concept intervention\n",
          "Emotion accuracy after correcting a fraction of one level's concepts with ground truth (L3 = **pseudo-expert**: "
          "category appraisal profiles, so correcting L3 leaks the category by construction).\n"]
for name, D, rows, levels in sets:
    lines += [f"### {name}\n", "| level | order | 0% | 30% | 50% | 70% | 100% | monotonic? |", "|---|---|---|---|---|---|---|---|"]
    for lv in levels:
        for order in ("random", "uncertainty"):
            c = curve(D, rows, lv, order); curves[(name, lv, order)] = c
            pick = {round(r["frac"], 1): r for r in c}
            mono = all(c[i + 1]["est"] >= c[i]["est"] - 0.005 for i in range(len(c) - 1))
            lines.append(f"| {lv.upper()} | {order} | " + " | ".join(f"{pick[f]['est']:.3f}" for f in (0.0, 0.3, 0.5, 0.7, 1.0)) + f" | {'yes' if mono else 'NO'} |")
    lines.append("")

fig, ax = plots.axes(3)
for i, lv in enumerate(("l1", "l2", "l3")):
    for j, order in enumerate(("random", "uncertainty")):
        c = curves[("FACED test", lv, order)]
        x = [r["frac"] * 100 for r in c]; y = [r["est"] for r in c]
        ax[i].plot(x, y, color=plots.SERIES[j], lw=2, marker="o", ms=4, label=order)
        ax[i].fill_between(x, [r["lo"] for r in c], [r["hi"] for r in c], color=plots.SERIES[j], alpha=0.12, lw=0)
    ax[i].set_title(f"Correct {lv.upper()}" + (" (pseudo-expert)" if lv == "l3" else ""), fontsize=10)
    ax[i].set_xlabel("% concepts corrected"); ax[i].set_ylabel("emotion accuracy" if i == 0 else "")
ax[0].legend(frameon=False, fontsize=9)
plots.save(fig, os.path.join(out_dir, "figures/intervention_faced.png"))
lines.append("![intervention](figures/intervention_faced.png)\n")

# ---------------- escalation ----------------
lines += ["## 2. Confidence escalation\n",
          f"Threshold per fold = confidence keeping {mcfg['training']['escalation_coverage']:.0%} of that fold's FACED validation windows. "
          "Values = mean +- std over the 5 fold models. Random escalation AURC = overall error rate.\n",
          "| model | set | coverage at threshold | error retained | error escalated | AURC | random AURC | ECE raw | ECE temp-scaled |",
          "|---|---|---|---|---|---|---|---|---|"]
evalsets = [("FACED val (CV)", F, None), ("FACED test", F, test_rows)]
if DR is not None:
    evalsets.append(("DREAMER (secondary)", DR, np.flatnonzero(~DR["reject"] & (DR["emotion"] >= 0))))
rc_curves = {}
h5 = []
for v in ("flat", a.variant):
    fl = [runs.load_fold(p, mcfg) for p in runs.fold_paths(ROOT, a.tag, v)]
    for sname, D, rows0 in evalsets:
        st = []
        for k, (m, ck) in enumerate(fl):
            rows = np.flatnonzero(~F["reject"] & np.isin(F["subject"], sp["folds"][k])) if rows0 is None else rows0
            lg = predict(m, D["E"][rows])["logits"]; y = D["emotion"][rows]
            p_raw, p_t = softmax(lg), softmax(lg, ck["temperature"])
            conf = p_t.max(1); correct = p_t.argmax(1) == y
            cov, risk, aurc = M.risk_coverage(conf, correct)
            keep = conf >= ck["threshold"]
            st.append([keep.mean(), 1 - correct[keep].mean() if keep.any() else np.nan,
                       1 - correct[~keep].mean() if (~keep).any() else np.nan, aurc, 1 - correct.mean(),
                       M.ece(p_raw, y), M.ece(p_t, y)])
            if sname == "FACED test" and k == 0:
                rc_curves[v] = (cov, risk)
        s = np.array(st, float); mu, sd = np.nanmean(s, 0), np.nanstd(s, 0)
        lines.append(f"| {v} | {sname} | " + " | ".join(f"{m_:.3f} +- {s_:.3f}" for m_, s_ in zip(mu, sd)) + " |")
        h5.append((v, sname, mu[2] > mu[1], mu[3] < mu[4]))
lines.append("\n**H5 check** (error escalated > error retained, and AURC < random):\n")
lines += [f"- {v}, {s}: escalated-error higher: **{'yes' if e else 'NO'}**; AURC below random: **{'yes' if r else 'NO'}**" for v, s, e, r in h5]

fig, ax = plots.axes(1)
for j, (v, (cov, risk)) in enumerate(rc_curves.items()):
    ax[0].plot(cov * 100, risk, color=plots.SERIES[j], lw=2, label=v)
    ax[0].axhline(risk[-1], color=plots.SERIES[j], lw=1, ls="--", alpha=0.6)
ax[0].set_xlabel("coverage (% windows kept, most confident first)"); ax[0].set_ylabel("error rate among kept")
ax[0].set_title("Risk-coverage, FACED test (fold 0)\ndashed = random escalation", fontsize=10)
ax[0].legend(frameon=False, fontsize=9)
plots.save(fig, os.path.join(out_dir, "figures/risk_coverage_faced.png"))
lines.append("\n![risk-coverage](figures/risk_coverage_faced.png)\n")
open(os.path.join(out_dir, "hitl.md"), "w").write("\n".join(lines))
json.dump({f"{n}|{lv}|{o}": c for (n, lv, o), c in curves.items()}, open(os.path.join(out_dir, "hitl_curves.json"), "w"))
print("wrote", os.path.join(out_dir, "hitl.md"))
