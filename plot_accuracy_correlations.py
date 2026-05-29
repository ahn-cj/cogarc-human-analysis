"""
Figure (§3 lead): continuous association between chunking / pacing measures
and problem-solving accuracy.

This is the continuous replacement for the top-solver dichotomy.  Instead of
splitting subjects at an arbitrary accuracy threshold, it treats accuracy as the
graded variable it is and asks how each behavioral measure tracks it.

Layout:
    Left  : forest plot — Spearman ρ (with bootstrap 95% CI) of every measure
            against accuracy, ranked by |ρ|.  Pacing/chunking core measures
            highlighted; significant effects filled, non-significant hollow.
    Right : 2×2 scatter panels for the four headline measures, each with a
            LOWESS trend and the Spearman ρ [CI], p annotated.

Reads from prior_analysis/:
    chunking_top_solvers_merged.csv        (per-subject features + accuracy)
    chunking_accuracy_correlations.csv     (ρ, CI, p per feature)
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
from statsmodels.nonparametric.smoothers_lowess import lowess


# ── style ───────────────────────────────────────────────────────────────────

COLOR_SIG     = "#2166ac"   # significant
COLOR_NS      = "#9aa0a6"   # non-significant
COLOR_CORE    = "#b2182b"   # core pacing/chunking constructs
COLOR_TREND   = "#b2182b"

# Core constructs of the chapter (pacing + chunking).
CORE_FEATURES = {"mean_rt_median", "size", "n_chunks_total", "n_cells"}

# Readable labels.
LABELS = {
    "mean_rt_median":           "edit pace (median inter-edit RT)",
    "deliberation_time_median": "deliberation time",
    "size":                     "mean chunk size",
    "n_cells":                  "cells per chunk",
    "n_chunks_total":           "number of chunks",
    "is_connected":             "chunk connectedness",
    "fill_ratio":               "fill ratio",
    "color_homogeneity":        "color homogeneity",
    "bbox_area":                "bounding-box area",
    "nn_chain_rate":            "nearest-neighbor chain rate",
    "success_iou_best":         "success IoU (best)",
}

# Four headline scatter panels: pacing, chunking, chunk extent, chunk quality.
HEADLINE = ["mean_rt_median", "size", "n_cells", "success_iou_best"]


def _fmt_p(p: float) -> str:
    return "p<.001" if p < 0.001 else f"p={p:.3f}"


def _scatter_panel(ax, x, y, label, rho, ci_lo, ci_hi, p):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ax.scatter(x, y, s=28, alpha=0.55, color="#4393c3",
               edgecolors="white", linewidths=0.4, zorder=3)
    # LOWESS trend
    if len(x) >= 10:
        sm = lowess(y, x, frac=0.7, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color=COLOR_TREND, lw=2.2, zorder=4)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("accuracy", fontsize=10)
    ax.grid(alpha=0.2)
    ax.text(
        0.04, 0.05,
        f"ρ = {rho:+.2f}  [{ci_lo:+.2f}, {ci_hi:+.2f}]\n{_fmt_p(p)}",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/accuracy_correlations_figure.png")
    args = ap.parse_args()

    merged = pd.read_csv(
        os.path.join(args.prior_dir, "chunking_top_solvers_merged.csv"))
    corr = pd.read_csv(
        os.path.join(args.prior_dir, "chunking_accuracy_correlations.csv"))

    n = int(merged["accuracy"].notna().sum())

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1.0, 1.0],
                  wspace=0.32, hspace=0.42,
                  left=0.27, right=0.97, top=0.88, bottom=0.10)

    # ── Left: forest plot ──────────────────────────────────────────────────────
    axF = fig.add_subplot(gs[:, 0])
    # ascending by rho so the most positive sits at the top
    cf = corr.sort_values("rho", ascending=True).reset_index(drop=True)
    ypos = np.arange(len(cf))
    for i, row in cf.iterrows():
        sig = row["p_value"] < 0.05
        is_core = row["feature"] in CORE_FEATURES
        color = COLOR_SIG if sig else COLOR_NS
        ax_face = color if sig else "white"
        axF.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                 color=color, lw=2.0, zorder=2)
        axF.scatter([row["rho"]], [i], s=70, facecolor=ax_face,
                    edgecolor=color, linewidths=1.6, zorder=3)
        if is_core:
            axF.scatter([row["rho"]], [i], s=200, facecolor="none",
                        edgecolor=COLOR_CORE, linewidths=1.4, zorder=4)
    axF.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axF.set_yticks(ypos)
    axF.set_yticklabels([LABELS.get(f, f) for f in cf["feature"]], fontsize=9.5)
    axF.set_xlabel("Spearman ρ with accuracy  (95% bootstrap CI)", fontsize=10)
    axF.set_title("A. Continuous associations with accuracy\n"
                  "(ranked; bootstrap 95% CI)", loc="left", fontsize=12)
    axF.grid(axis="x", alpha=0.2)
    legend_handles = [
        Line2D([0], [0], marker="o", color=COLOR_SIG, lw=0,
               markerfacecolor=COLOR_SIG, markersize=8, label="p < .05"),
        Line2D([0], [0], marker="o", color=COLOR_NS, lw=0,
               markerfacecolor="white", markeredgecolor=COLOR_NS,
               markersize=8, label="n.s."),
        Line2D([0], [0], marker="o", color=COLOR_CORE, lw=0,
               markerfacecolor="none", markeredgecolor=COLOR_CORE,
               markersize=11, label="core pacing/chunking"),
    ]
    axF.legend(handles=legend_handles, fontsize=8.5, loc="lower right",
               frameon=True)

    # ── Right: 2×2 scatter panels ──────────────────────────────────────────────
    corr_idx = corr.set_index("feature")
    panel_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    panel_letters = ["B", "C", "D", "E"]
    for ax, feat, letter in zip(panel_axes, HEADLINE, panel_letters):
        r = corr_idx.loc[feat]
        _scatter_panel(
            ax,
            merged[feat].values, merged["accuracy"].values,
            LABELS.get(feat, feat),
            float(r["rho"]), float(r["ci_lo"]), float(r["ci_hi"]),
            float(r["p_value"]),
        )
        ax.set_title(f"{letter}. {LABELS.get(feat, feat)}",
                     loc="left", fontsize=11)

    fig.suptitle(
        f"Section 3: chunking and pacing measures track problem-solving accuracy "
        f"(continuous; n={n} subjects)",
        fontsize=13, y=0.95,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
