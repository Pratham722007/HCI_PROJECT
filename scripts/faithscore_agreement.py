"""Phase 8.6: agreement between the automatic FaithScore judges and your manual labels (20 pairs).

Fill the `human_faithful` column (1 = explanation changed faithfully, 0 = not) in
reports/<tag>/faithscore_manual_labels.csv, then run:
  .venv/bin/python -E -P scripts/faithscore_agreement.py --tag 12ch   -> appends to reports/<tag>/faithscore.md
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="12ch"); a = ap.parse_args()
d = pd.read_csv(os.path.join(ROOT, "reports", a.tag, "faithscore_manual_labels.csv")).dropna(subset=["human_faithful"])
assert len(d), "no manual labels filled in yet"
lines = [f"\n## Judge reliability vs manual labels (n = {len(d)})\n", "| judge | agreement | Cohen's kappa |", "|---|---|---|"]
for j in ("judge_a", "judge_b"):
    lines.append(f"| {j} | {np.mean(d[j] == d.human_faithful):.2f} | {cohen_kappa_score(d[j], d.human_faithful):.2f} |")
open(os.path.join(ROOT, "reports", a.tag, "faithscore.md"), "a").write("\n".join(lines) + "\n")
print("\n".join(lines))
