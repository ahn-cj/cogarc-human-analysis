"""
Figure: Spearman ρ of each measure with accuracy, uncontrolled vs after
regression controls for task difficulty, trajectory length, and (for pacing)
n_examples / example-view-time.

Continuous replacement for the earlier top-solver Cliff's δ version: accuracy
is treated as a continuous variable, so the robustness check is whether each
measure's rank correlation with accuracy survives residualising the feature on
its confounds.

Reads:
    prior_analysis/controlled_accuracy_correlations.csv
        (feature, rho_raw, ci_lo_raw, ci_hi_raw, p_raw,
                  rho_ctrl, ci_lo_ctrl, ci_hi_ctrl, p_ctrl)
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FEATURE_ORDER = [
    # Pacing first (visually grouped at the top of the figure)
    "deliberation_time", "mean_rt_between_edits",
    # Chunking
    "size", "n_cells", "n_chunks_total",
    "is_connected", "fill_ratio",
    "color_homogeneity",
    "bbox_area", "nn_chain_rate", "success_iou_best",
]

FEATURE_LABELS = {
    "deliberation_time":     "deliberation time",
    "mean_rt_between_edits": "inter-edit RT (edit pace)",
    "size":                  "chunk size",
    "n_cells":               "unique cells per chunk",
    "n_chunks_total":        "n chunks per trajectory",
    "is_connected":          "frac 4-connected chunks",
    "fill_ratio":            "fill ratio (compactness)",
    "color_homogeneity":     "color homogeneity",
    "bbox_area":             "bbox area",
    "nn_chain_rate":         "draw-order adjacency",
    "success_iou_best":      "success-IoU per chunk",
}

RELIABLE = {"deliberation_time", "mean_rt_between_edits",
            "size", "n_cells", "n_chunks_total",
            "is_connected", "fill_ratio"}

COLOR_RAW  = "#999999"
COLOR_CTRL = "#2166ac"


def _sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/controls_comparison_figure.png")
    args = ap.parse_args()

    corr = pd.read_csv(os.path.join(args.prior_dir,
                                    "controlled_accuracy_correlations.csv"))
    idx = corr.set_index("feature")

    feats = [f for f in FEATURE_ORDER if f in idx.index]
    y = np.arange(len(feats))
    off = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    for cond, ys, color, lo_c, hi_c, rho_c, p_c, lbl in [
        ("raw",  y - off, COLOR_RAW,  "ci_lo_raw",  "ci_hi_raw",  "rho_raw",  "p_raw",  "uncontrolled"),
        ("ctrl", y + off, COLOR_CTRL, "ci_lo_ctrl", "ci_hi_ctrl", "rho_ctrl", "p_ctrl", "controlled (difficulty + log-trajectory + n_examples / view-time)"),
    ]:
        rho = np.array([idx.loc[f, rho_c] for f in feats])
        lo  = np.array([idx.loc[f, lo_c]  for f in feats])
        hi  = np.array([idx.loc[f, hi_c]  for f in feats])
        p   = np.array([idx.loc[f, p_c]   for f in feats])
        ax.errorbar(rho, ys, xerr=[rho - lo, hi - rho], fmt="o",
                    color=color, ecolor=color, elinewidth=1.6, capsize=3,
                    markersize=6, label=lbl, zorder=3)
        for r, yy, hival, pv in zip(rho, ys, hi, p):
            sig = _sig_marker(pv)
            if sig:
                ax.text(hival + 0.015, yy, sig, va="center", ha="left",
                        fontsize=9, color=color, fontweight="bold")

    ax.axvline(0, color="black", lw=0.7, zorder=2)
    ax.axvspan(-0.10, 0.10, color="#f5f5f5", zorder=1, label="negligible (|ρ| < .10)")

    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in feats], fontsize=10)
    for tick, f in zip(ax.get_yticklabels(), feats):
        if f in RELIABLE:
            tick.set_fontweight("bold")

    ax.set_xlabel("Spearman ρ with accuracy  (95% bootstrap CI)", fontsize=11)
    ax.set_xlim(-0.6, 0.6)
    ax.invert_yaxis()
    ax.set_title(
        "Continuous accuracy correlations survive regression controls\n"
        "(deliberation time and n-chunks effects emerge once their confounds are removed)\n"
        "(***p<.001  **p<.01  *p<.05)",
        loc="left", fontsize=11.5,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
