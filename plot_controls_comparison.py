"""
Figure: Cliff's δ for top-solver vs rest, uncontrolled vs after regression
controls for task difficulty and trajectory length.

Reads:
    prior_analysis/chunking_top_solvers_stats.csv      (uncontrolled)
    prior_analysis/controlled_top_solver_stats.csv     (after controls)

Plots paired horizontal bars (one feature per row) so the visual contrast
between the two conditions is immediate.  Significance markers are added
next to each bar.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_ORDER = [
    "size", "n_cells", "n_chunks_total",
    "is_connected", "fill_ratio",
    "color_homogeneity",
    "bbox_area", "nn_chain_rate", "success_iou_best",
]

FEATURE_LABELS = {
    "size":              "chunk size",
    "n_cells":           "unique cells per chunk",
    "n_chunks_total":    "n chunks per trajectory",
    "is_connected":      "frac 4-connected chunks",
    "fill_ratio":        "fill ratio (compactness)",
    "color_homogeneity": "color homogeneity",
    "bbox_area":         "bbox area",
    "nn_chain_rate":     "draw-order adjacency",
    "success_iou_best":  "success-IoU per chunk",
}

# Reliable features (split-half ρ > 0.2) get highlighted
RELIABLE = {"size", "n_cells", "n_chunks_total", "is_connected", "fill_ratio"}


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

    raw = pd.read_csv(os.path.join(args.prior_dir,
                                    "chunking_top_solvers_stats.csv"))
    ctrl = pd.read_csv(os.path.join(args.prior_dir,
                                     "controlled_top_solver_stats.csv"))

    raw_idx  = raw.set_index("feature")
    ctrl_idx = ctrl.set_index("feature")

    feats = [f for f in FEATURE_ORDER if f in raw_idx.index and f in ctrl_idx.index]
    y = np.arange(len(feats))
    h = 0.36

    d_raw  = np.array([raw_idx.loc[f, "cliffs_delta"]  for f in feats])
    p_raw  = np.array([raw_idx.loc[f, "p_mannwhitney"] for f in feats])
    d_ctrl = np.array([ctrl_idx.loc[f, "cliffs_delta"]  for f in feats])
    p_ctrl = np.array([ctrl_idx.loc[f, "p_mannwhitney"] for f in feats])

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_raw  = ax.barh(y - h/2, d_raw,  h,
                        color="#999999", alpha=0.85,
                        label="uncontrolled", zorder=3)
    bars_ctrl = ax.barh(y + h/2, d_ctrl, h,
                        color="#2166ac", alpha=0.85,
                        label="controlled (difficulty + log trajectory length)",
                        zorder=3)

    # Annotate δ and significance at the end of each bar
    x_pad = 0.012
    for bar, d, p in zip(bars_raw, d_raw, p_raw):
        sig = _sig_marker(p)
        ax.text(d + (x_pad if d >= 0 else -x_pad), bar.get_y() + bar.get_height()/2,
                f"{d:+.2f}{sig}",
                ha="left" if d >= 0 else "right",
                va="center", fontsize=8.5, color="#444444")
    for bar, d, p in zip(bars_ctrl, d_ctrl, p_ctrl):
        sig = _sig_marker(p)
        ax.text(d + (x_pad if d >= 0 else -x_pad), bar.get_y() + bar.get_height()/2,
                f"{d:+.2f}{sig}",
                ha="left" if d >= 0 else "right",
                va="center", fontsize=8.5, color="#1f4f7f", fontweight="bold")

    ax.set_yticks(y)
    # Bold the reliable feature labels
    labels = []
    for f in feats:
        lbl = FEATURE_LABELS.get(f, f)
        if f in RELIABLE:
            lbl = f"{lbl}"
        labels.append(lbl)
    ax.set_yticklabels(labels, fontsize=10)
    for tick, f in zip(ax.get_yticklabels(), feats):
        if f in RELIABLE:
            tick.set_fontweight("bold")

    ax.axvline(0, color="black", lw=0.7, zorder=2)
    ax.axvspan(-0.147, 0.147, color="#f5f5f5", zorder=1,
               label="negligible (|δ| < .15)")

    ax.set_xlabel("Cliff's δ  (top solvers vs rest)", fontsize=11)
    ax.set_xlim(-0.55, 0.55)
    ax.invert_yaxis()  # most-affected feature at top
    ax.set_title(
        "Top-solver chunking effects survive controls for task difficulty and trajectory length\n"
        "(bold feature names = split-half ρ > 0.2;   ***p<.001  **p<.01  *p<.05)",
        loc="left", fontsize=11.5,
    )
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
