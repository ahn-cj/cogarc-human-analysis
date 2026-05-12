"""
Figure: chunking style × top-solver performance and timing.

Three panels:

    A. Top-solver vs rest — violin + strip plots for each reliable chunk
       feature, with Mann-Whitney p and Cliff's delta annotated.

    B. Chunk-feature × timing correlation heatmap (Spearman ρ).
       Rows = chunk features; columns = deliberation time, RT between edits.

    C. Scatter: mean chunk size vs median RT between edits, coloured by
       accuracy (continuous), with top-solver boundary marked.

Reads from prior_analysis/ (outputs of chunking_top_solvers.py).
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ── labels / style ────────────────────────────────────────────────────────────

FEAT_LABELS = {
    "size": "chunk size\n(edits)",
    "n_chunks_total": "n chunks per\ntrajectory",
    "n_cells": "n cells per\nchunk",
    "is_connected": "frac 4-connected\nchunks",
    "fill_ratio": "fill ratio\n(compactness)",
    "color_homogeneity": "color\nhomogeneity",
    "bbox_area": "bbox area",
    "nn_chain_rate": "draw-order\nadjacency",
    "success_iou_best": "success-IoU\nper chunk",
}

TIMING_LABELS = {
    "deliberation_time_median": "deliberation\ntime (ms)",
    "mean_rt_median": "mean RT\nbetween edits (ms)",
}

COLOR_TOP = "#2166ac"
COLOR_REST = "#d6604d"

# Features with split-half reliability ρ > 0.2 (from individual differences analysis)
RELIABLE_FEATURES = ["size", "n_chunks_total", "n_cells", "is_connected", "fill_ratio"]


def _strip_jitter(n: int, pos: float, width: float = 0.08, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pos + rng.uniform(-width, width, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/chunking_top_solvers_figure.png")
    args = ap.parse_args()

    merged = pd.read_csv(os.path.join(args.prior_dir,
                                      "chunking_top_solvers_merged.csv"))
    top_stats = pd.read_csv(os.path.join(args.prior_dir,
                                         "chunking_top_solvers_stats.csv"))
    timing_corr = pd.read_csv(os.path.join(args.prior_dir,
                                            "chunking_timing_correlations.csv"))

    top_stats_idx = top_stats.set_index("feature")

    avail_feats = [f for f in RELIABLE_FEATURES if f in merged.columns]
    avail_timing = [c for c in ["deliberation_time_median", "mean_rt_median"]
                    if c in merged.columns]

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.8, 0.7, 1.0],
                  wspace=0.38, left=0.05, right=0.98, top=0.88, bottom=0.14)

    # ── Panel A: violins top vs rest ──────────────────────────────────────────
    axA = fig.add_subplot(gs[0, 0])

    n_feats = len(avail_feats)
    ys = np.arange(n_feats)
    top_mask = merged["top_solver"].astype(bool)
    rest_mask = ~top_mask

    violin_width = 0.32
    offset = 0.22

    for i, feat in enumerate(avail_feats):
        for j, (mask, color, pos_off) in enumerate([
            (top_mask, COLOR_TOP, -offset),
            (rest_mask, COLOR_REST, +offset),
        ]):
            vals = merged.loc[mask, feat].dropna().values
            if len(vals) < 5:
                continue
            vp = axA.violinplot(
                vals,
                positions=[i + pos_off],
                widths=violin_width,
                showmedians=True,
                showextrema=False,
                vert=False,
            )
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)

            xs = _strip_jitter(len(vals), i + pos_off, width=0.06, seed=j)
            axA.scatter(vals, xs, color=color, alpha=0.35, s=8, zorder=3)

    # Annotate Cliff's delta and p-value after all violins are drawn.
    # Expand xlim by 40 % on the right so annotations don't bleed into Panel B.
    axA.autoscale(axis="x")
    x_lo, x_hi = axA.get_xlim()
    x_span = x_hi - x_lo
    axA.set_xlim(x_lo, x_hi + 0.40 * x_span)
    x_ann = x_hi + 0.02 * x_span  # just past the data edge
    for i, feat in enumerate(avail_feats):
        if feat in top_stats_idx.index:
            row = top_stats_idx.loc[feat]
            d = row["cliffs_delta"]
            p = row["p_mannwhitney"]
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<.001"
            axA.text(x_ann, i, f"δ={d:+.2f}{sig}  {p_str}",
                     va="center", ha="left", fontsize=7.5, color="#555555")

    axA.set_yticks(ys)
    axA.set_yticklabels([FEAT_LABELS.get(f, f).replace("\n", " ")
                         for f in avail_feats], fontsize=9.5)
    axA.set_xlabel("feature value (per-subject mean)", fontsize=9.5)
    axA.set_title("A. Chunk features: top solvers vs rest",
                  loc="left", fontsize=12)
    axA.grid(axis="x", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=COLOR_TOP, lw=4, alpha=0.7,
               label=f"top solvers (n={top_mask.sum()}, ≥95 % acc)"),
        Line2D([0], [0], color=COLOR_REST, lw=4, alpha=0.7,
               label=f"rest (n={rest_mask.sum()})"),
    ]
    axA.legend(handles=legend_handles, loc="lower right", fontsize=9)

    # ── Panel B: chunk × timing heatmap ──────────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])

    mat = timing_corr.pivot(index="chunk_feature",
                             columns="timing_feature",
                             values="rho").reindex(
        index=avail_feats, columns=avail_timing)
    p_mat = timing_corr.pivot(index="chunk_feature",
                               columns="timing_feature",
                               values="p_value").reindex(
        index=avail_feats, columns=avail_timing)

    im = axB.imshow(mat.values, cmap="RdBu_r", vmin=-0.35, vmax=0.35,
                    aspect="auto")
    axB.set_xticks(range(len(avail_timing)))
    axB.set_xticklabels([TIMING_LABELS.get(t, t) for t in avail_timing],
                        fontsize=9)
    axB.set_yticks(range(len(avail_feats)))
    axB.set_yticklabels([FEAT_LABELS.get(f, f).replace("\n", " ")
                         for f in avail_feats], fontsize=9)

    for i, feat in enumerate(avail_feats):
        for j, tf in enumerate(avail_timing):
            v = mat.values[i, j]
            p = p_mat.values[i, j]
            if np.isnan(v):
                continue
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            label = f"{v:.2f}{sig}"
            axB.text(j, i, label, ha="center", va="center", fontsize=9,
                     color="white" if abs(v) > 0.2 else "black")

    axB.set_title("B. Chunk style × timing\n(Spearman ρ, *p<.05 **p<.01)",
                  loc="left", fontsize=12)
    cbar = fig.colorbar(im, ax=axB, shrink=0.7, pad=0.04)
    cbar.ax.set_ylabel("Spearman ρ", fontsize=8.5)

    # ── Panel C: scatter chunk size vs RT, coloured by accuracy ──────────────
    axC = fig.add_subplot(gs[0, 2])

    sub = merged[["size", "mean_rt_median", "accuracy", "top_solver"]].dropna()
    sc = axC.scatter(
        sub["size"],
        sub["mean_rt_median"],
        c=sub["accuracy"],
        cmap="RdYlGn",
        vmin=0.5, vmax=1.0,
        s=40,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )
    # Ring top solvers
    top_sub = sub[sub["top_solver"]]
    axC.scatter(
        top_sub["size"],
        top_sub["mean_rt_median"],
        s=90, facecolors="none",
        edgecolors=COLOR_TOP, linewidths=1.5, zorder=4,
        label="top solver",
    )

    from scipy.stats import spearmanr
    ok = sub[["size", "mean_rt_median"]].notna().all(axis=1)
    rho, p = spearmanr(sub.loc[ok, "size"], sub.loc[ok, "mean_rt_median"])
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    axC.set_xlabel("mean chunk size (edits per chunk)", fontsize=9.5)
    axC.set_ylabel("median RT between edits (ms)", fontsize=9.5)
    axC.set_title(
        f"C. Chunk size vs edit pace\n(Spearman ρ={rho:.2f}, {p_str})",
        loc="left", fontsize=12,
    )
    axC.legend(fontsize=9)
    axC.grid(alpha=0.25)
    cbar2 = fig.colorbar(sc, ax=axC, shrink=0.7, pad=0.04)
    cbar2.ax.set_ylabel("accuracy", fontsize=8.5)

    n_top = int(top_mask.sum())
    n_rest = int(rest_mask.sum())
    n_total = len(merged)
    fig.suptitle(
        f"Chunking style × top-solver performance  "
        f"(n={n_total} subjects: {n_top} top, {n_rest} rest)",
        fontsize=13, y=0.96,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
