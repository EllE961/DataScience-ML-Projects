"""Redraw docs/assets/progress.png from the tables in README.md.

Run from the repo root:  python scripts/make_figures.py

Each "## <category>" section in the README holds a table whose last column is "done" or "open".
The script counts both per category and draws one stacked bar per category.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")


def read_progress():
    """Return [(category, done, open), ...] in README order."""
    counts, heading = {}, None
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## "):
                heading = line[3:]
                continue
            row = re.match(r"^\|\s*\d+\s*\|.*\|\s*(done|open)\s*\|$", line)
            if row and heading:
                d, o = counts.get(heading, (0, 0))
                counts[heading] = (d + 1, o) if row.group(1) == "done" else (d, o + 1)
    return [(name, d, o) for name, (d, o) in counts.items()]


def main():
    cats = read_progress()
    total = sum(d + o for _, d, o in cats)
    done = sum(d for _, d, _ in cats)
    print(f"{done} of {total} done")
    for name, d, o in cats:
        print(f"  {name:50s} {d:2d} / {d + o:2d}")

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    names = [c[0] for c in cats]
    y = range(len(cats))
    ax.barh(y, [c[1] for c in cats], color=EMBER, height=0.62, label="done")
    ax.barh(y, [c[2] for c in cats], left=[c[1] for c in cats], color="#2b2825", height=0.62, label="open")
    for i, (_, d, o) in enumerate(cats):
        ax.text(d + o + 0.4, i, f"{d} / {d + o}", va="center", color=INK2 if d else INK3, fontsize=9.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=INK2, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(d + o for _, d, o in cats) + 4)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("projects")
    ax.set_title(f"Progress by category: {done} of {total} projects done", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 0.995), ncol=2)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "progress.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
