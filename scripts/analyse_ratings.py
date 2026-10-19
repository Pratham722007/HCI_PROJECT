"""Phase 10.6: unblind and analyse human ratings -> reports/<tag>/human_eval.md.

Run: .venv/bin/python -E -P scripts/analyse_ratings.py --tag 12ch reports/<tag>/human_eval/ratings_*.csv
     --synthetic: dry run with 3 fake raters (writes human_eval_DRYRUN.md; tests the full report path).
Per criterion: CBM vs flat mean +- item-bootstrap CI, Wilcoxon signed-rank on per-item mean difference, rank-biserial
effect size, Benjamini-Hochberg FDR across criteria; agreement = Krippendorff's alpha (interval) and Fleiss' kappa;
split by CBM-correct vs wrong prediction.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.eval.agreement import krippendorff_interval, fleiss_kappa, bh  # noqa: E402
CRITERIA = ["faithfulness", "plausibility", "clarity", "actionability"]


def synthetic(items, key, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for r in range(3):
        for it in items:
            for s in ("A", "B"):
                base = 3.4 if key[str(it["id"])][s] == "cbm" else 3.0
                for c in CRITERIA:
                    rows.append(dict(rater_id=f"fake{r}", item_id=it["id"], side=s, criterion=c,
                                     score=int(np.clip(np.round(base + rng.normal(0, 0.9)), 1, 5))))
    return pd.DataFrame(rows)


ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="*"); ap.add_argument("--tag", default="12ch")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()
he = os.path.join(ROOT, "reports", a.tag, "human_eval")
K = json.load(open(os.path.join(he, "key_PRIVATE.json")))["key"]
items = json.load(open(os.path.join(he, "items.json")))
df = synthetic(items, K) if a.synthetic else pd.concat([pd.read_csv(f) for f in (a.files or glob.glob(os.path.join(he, "ratings_*.csv")))])
df = df[df.criterion.isin(CRITERIA)].copy()
df["score"] = df.score.astype(float)
df["system"] = [K[str(int(i))][s] for i, s in zip(df.item_id, df.side)]
df["cbm_correct"] = [K[str(int(i))]["cbm_correct"] for i in df.item_id]
raters = sorted(df.rater_id.unique())
rng = np.random.default_rng(42)

lines = [f"# Human evaluation (tag `{a.tag}`){' - DRY RUN, SYNTHETIC RATERS' if a.synthetic else ''}\n",
         f"{len(raters)} raters, {df.item_id.nunique()} items. A/B balance: "
         f"{np.mean([K[str(i)]['A'] == 'cbm' for i in df.item_id.unique()]):.2f} of items had CBM as A.\n",
         "| criterion | CBM mean [95% CI] | flat mean [95% CI] | Wilcoxon p | FDR q | rank-biserial r | Krippendorff alpha | Fleiss kappa |",
         "|---|---|---|---|---|---|---|---|"]
res = []
for c in CRITERIA:
    d = df[df.criterion == c]
    per = d.groupby(["item_id", "system"]).score.mean().unstack()                     # mean over raters
    diff = (per["cbm"] - per["flat"]).dropna()
    bs = {s: np.percentile([per[s].sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(1000)], [2.5, 97.5]) for s in ("cbm", "flat")}
    nz = diff[diff != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    rb = float((np.sum(np.sign(nz) * pd.Series(np.abs(nz)).rank().values)) / pd.Series(np.abs(nz)).rank().sum()) if len(nz) else 0.0
    wide = d.assign(unit=d.item_id.astype(str) + d.side).pivot_table(index="rater_id", columns="unit", values="score")
    res.append(dict(c=c, cbm=per["cbm"].mean(), flat=per["flat"].mean(), bs=bs, p=p, rb=rb,
                    alpha=krippendorff_interval(wide.values), kappa=fleiss_kappa(wide.values)))
for r, q in zip(res, bh([r["p"] for r in res])):
    lines.append(f"| {r['c']} | {r['cbm']:.2f} [{r['bs']['cbm'][0]:.2f}, {r['bs']['cbm'][1]:.2f}] | {r['flat']:.2f} [{r['bs']['flat'][0]:.2f}, "
                 f"{r['bs']['flat'][1]:.2f}] | {r['p']:.4f} | {q:.4f} | {r['rb']:+.2f} | {r['alpha']:.2f} | {r['kappa']:.2f} |")
lines += ["\n## By CBM prediction correctness (mean over raters and items)\n", "| criterion | CBM correct: CBM / flat | CBM wrong: CBM / flat |", "|---|---|---|"]
for c in CRITERIA:
    d = df[df.criterion == c]; cells = []
    for ok in (True, False):
        g = d[d.cbm_correct == ok].groupby("system").score.mean()
        cells.append(f"{g.get('cbm', np.nan):.2f} / {g.get('flat', np.nan):.2f}")
    lines.append(f"| {c} | {cells[0]} | {cells[1]} |")
name = "human_eval_DRYRUN.md" if a.synthetic else "human_eval.md"
open(os.path.join(ROOT, "reports", a.tag, name), "w").write("\n".join(lines))
print("wrote", name)
