"""Results-progression chart for the progress report. Run with the project env: .venv/bin/python -E -P docs/report/make_chart.py"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.eval import plots  # noqa: E402
runs = ["12 channels\n(faced12)", "30 channels +\nsubject norm\n(faced30)", "+ spectral features\n+ smoothing\n(faced30sx)"]
cbm, flat = [0.197, 0.228, 0.295], [0.227, 0.266, 0.331]          # reports/<tag>/in_corpus.md, test macro-F1 (ensemble)
fig, ax = plots.axes(1, w=6.4, h=3.4); a = ax[0]
x, w = np.arange(3), 0.34
b1 = a.bar(x - w / 2 - 0.01, cbm, w, color=plots.SERIES[0], label="Main CBM (level123)")
b2 = a.bar(x + w / 2 + 0.01, flat, w, color=plots.SERIES[1], label="Flat baseline")
for bars in (b1, b2):
    for r in bars:
        a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.006, f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, color=plots.TEXT)
a.axhline(1 / 7, color=plots.MUTED, lw=1.2, ls="--"); a.text(-0.62, 1 / 7 + 0.012, "chance 0.143", ha="left", fontsize=8, color=plots.TEXT2)
a.set_xlim(-0.65, 2.65); a.set_xticks(x); a.set_xticklabels(runs, fontsize=8.5); a.set_ylim(0, 0.38); a.set_ylabel("Test macro-F1 (7 classes)")
a.legend(frameon=False, fontsize=8.5, loc="upper left")
plots.save(fig, os.path.join(ROOT, "docs/report/progression.png"))
