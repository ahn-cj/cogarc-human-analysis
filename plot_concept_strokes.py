"""
Figure 1 (concept): what a chunk is, and that chunk granularity is a reliable trait.

Top row — the SAME problem drawn by a lumper (few large strokes) and a splitter
(many small clicks); each participant's cells are coloured by which stroke painted
them, so a lumper shows a few large blocks and a splitter shows many small ones.
Bottom — split-half reliability of cells-per-stroke across participants.

Usage:  python plot_concept_strokes.py --task d9f24cd1 --lumper ei0ij8ms --splitter xgvto4mg
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr, rankdata

from human_targets import human_targets
from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR

TYPED_DIR = os.path.join(DEFAULT_DATA_ROOT, os.path.dirname(_EXP2_EDIT_DIR),
                         "Edit sequences typed")
ARC = ['#2B2B2B', '#248ADA', '#C71010', '#1FC719', '#F7DE28',
       '#878494', '#F954F2', '#EE6000', '#6B23A9', '#8B5A28']
ARC_CMAP = mcolors.ListedColormap(ARC)


def solution_grid(task):
    t = human_targets(task)
    for lbl, g in zip(t["labels"], t["grids"]):
        if lbl == "Success":
            return np.array(g)
    return None


def load_strokes(task, subj):
    """Return ordered edits [(row,col,color)] and a per-edit stroke index.
    Grid convention: row = raw x, col = raw y (verified by IoU with the solution)."""
    path = os.path.join(TYPED_DIR, task + ".json", f"subj_{subj}_trial_{task}.json.csv")
    edits, stroke_id, sid = [], [], 0
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r["action"] != "edit":
                continue
            row, col, c = int(float(r["x"])), int(float(r["y"])), int(float(r["color"]))
            if edits and r.get("type") == "click":
                sid += 1
            elif not edits:
                sid = 0
            edits.append((row, col, c))
            stroke_id.append(sid)
    return edits, stroke_id


def draw_grid(ax, grid, title, edgecolor="#cccccc"):
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#888888")
    ax.set_title(title, fontsize=11)


def draw_strokes(ax, shape, edits, stroke_id, title):
    """Colour each painted cell by its (last) stroke index, qualitative palette."""
    H, W = shape
    stroke_of = -np.ones((H, W), dtype=int)
    for (y, x, _), sid in zip(edits, stroke_id):
        if 0 <= y < H and 0 <= x < W:
            stroke_of[y, x] = sid
    n_str = max(stroke_id) + 1 if stroke_id else 0
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    rng = np.random.default_rng(3)
    palette = base[rng.permutation(20)]
    img = np.ones((H, W, 3))
    for y in range(H):
        for x in range(W):
            s = stroke_of[y, x]
            if s >= 0:
                img[y, x] = palette[s % 20][:3]
    ax.imshow(img, interpolation="none")
    # grid lines
    for k in range(W + 1):
        ax.axvline(k - 0.5, color="#dddddd", lw=0.5)
    for k in range(H + 1):
        ax.axhline(k - 0.5, color="#dddddd", lw=0.5)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def split_half(seed=0):
    per = pd.read_csv("prior_analysis/stroke_chunking_per_st.csv")
    rng = np.random.default_rng(seed)
    a, b = [], []
    for sid, g in per.groupby("subject_id"):
        if len(g) < 8:
            continue
        idx = rng.permutation(len(g)); h = len(g) // 2
        a.append(g.iloc[idx[:h]]["cells_per_stroke"].mean())
        b.append(g.iloc[idx[h:]]["cells_per_stroke"].mean())
    return np.array(a), np.array(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="d9f24cd1")
    ap.add_argument("--lumper", default="j2dvablb")
    ap.add_argument("--splitter", default="a6jpzpek")
    ap.add_argument("--out", default="prior_analysis/concept_strokes_figure.png")
    args = ap.parse_args()

    sol = solution_grid(args.task)
    le, ls = load_strokes(args.task, args.lumper)
    se, ss = load_strokes(args.task, args.splitter)
    shape = sol.shape

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.5, wspace=0.18,
                          left=0.07, right=0.97, top=0.80, bottom=0.07)

    draw_grid(fig.add_subplot(gs[0, 0]), sol, "Target solution")
    draw_strokes(fig.add_subplot(gs[0, 1]), shape, le, ls,
                 f"Lumper — {max(ls)+1} strokes\n({len(le)/(max(ls)+1):.0f} cells per stroke)")
    draw_strokes(fig.add_subplot(gs[0, 2]), shape, se, ss,
                 f"Splitter — {max(ss)+1} strokes\n({len(se)/(max(ss)+1):.1f} cells per stroke)")

    fig.text(0.07, 0.905, "A. The chunking unit: the same problem, drawn coarsely vs finely  "
             "(each colour = one continuous stroke)", fontsize=12, fontweight="bold")

    # reliability scatter
    ax = fig.add_subplot(gs[1, :])
    a, b = split_half()
    ax.scatter(a, b, s=22, alpha=0.5, color="#2166ac", edgecolors="white", linewidths=0.3)
    lim = [0, np.percentile(np.r_[a, b], 99) * 1.05]
    ax.plot(lim, lim, color="black", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    rho = spearmanr(a, b).statistic
    ax.text(0.03, 0.95, f"split-half Spearman ρ = {rho:.2f}\n(n = {len(a)} participants)",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_xlabel("cells per stroke — random half A", fontsize=10)
    ax.set_ylabel("half B", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.07, 0.40, "B. Chunk granularity (cells per stroke) is a reliable individual trait",
             fontsize=12, fontweight="bold")

    fig.suptitle("Chunk granularity: the lumper–splitter dimension and its reliability",
                 fontsize=13.5, x=0.07, ha="left", y=0.965)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}  (lumper {max(ls)+1} strokes, splitter {max(ss)+1} strokes)")


if __name__ == "__main__":
    main()
