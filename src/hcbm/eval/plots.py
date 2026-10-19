"""Shared matplotlib style for report figures (reference categorical palette, fixed slot order; thin marks)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]   # slots 1,2,3,7,8 (avoid adjacent low-contrast pairs)
MUTED = "#8a8985"
TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e0"; SURFACE = "#fcfcfb"


def axes(n=1, w=5.0, h=3.6):
    fig, ax = plt.subplots(1, n, figsize=(w * n, h), squeeze=False, facecolor=SURFACE)
    for a in ax[0]:
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=TEXT2, labelsize=9)
        a.xaxis.label.set_color(TEXT2); a.yaxis.label.set_color(TEXT2); a.title.set_color(TEXT)
    return fig, ax[0]


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)
