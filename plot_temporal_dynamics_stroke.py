"""
Figure 7 (stroke measures): within-experiment temporal dynamics of chunk
granularity and pacing.

Row 1 — within-subject early-vs-late stability (each point = one subject).
Row 2 — quartile trajectories, top solvers vs rest (mean ± SE of per-subject means).

Reads prior_analysis/temporal_dynamics_stroke_{early_late,stability,quartiles}.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR_TOP, COLOR_REST, COLOR_OTHER = "#2166ac", "#d6604d", "#808080"
MEASURES = [
    ("cells_per_stroke",  "Cells per stroke (granularity)", "cells"),
    ("rt",                "Inter-edit RT (execution pace)", "log s"),
    ("deliberation_time", "Deliberation time (planning pace)", "log s"),
]
QORDER = ["Q1", "Q2", "Q3", "Q4"]
QLAB = ["Q1\n(1-19)", "Q2\n(20-37)", "Q3\n(38-56)", "Q4\n(57-75)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out", default="prior_analysis/temporal_dynamics_figure.png")
    args = ap.parse_args()
    el = pd.read_csv(f"{args.prior_dir}/temporal_dynamics_stroke_early_late.csv")
    stab = pd.read_csv(f"{args.prior_dir}/temporal_dynamics_stroke_stability.csv").set_index("outcome")
    quart = pd.read_csv(f"{args.prior_dir}/temporal_dynamics_stroke_quartiles.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5),
                             gridspec_kw=dict(hspace=0.42, wspace=0.32, left=0.07,
                                              right=0.97, top=0.86, bottom=0.08))

    for ax, (col, title, unit) in zip(axes[0], MEASURES):
        sub = el[el["outcome"] == col].dropna(subset=["early", "late"])
        ax.scatter(sub["early"], sub["late"], s=25, alpha=0.55, color=COLOR_OTHER,
                   edgecolors="white", linewidths=0.4)
        allv = np.concatenate([sub["early"].values, sub["late"].values])
        lo, hi = np.percentile(allv, [1, 99]); pad = 0.04 * (hi - lo)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color="black", ls="--", lw=0.7, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        rho = float(stab.loc[col, "spearman_rho"]); n = int(stab.loc[col, "n_subjects"])
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}\nn = {n}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92))
        ax.set_xlabel(f"early half ({unit})", fontsize=10)
        ax.set_ylabel(f"late half ({unit})", fontsize=10)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        ax.grid(alpha=0.2); ax.set_aspect("equal", adjustable="box")
    fig.text(0.025, 0.905, "A. Within-subject early-vs-late stability  "
             "(rank order is stable for all three measures)",
             fontsize=12.5, fontweight="bold")

    for ax, (col, title, unit) in zip(axes[1], MEASURES):
        sub = quart[quart["outcome"] == col]
        x = np.arange(len(QORDER))
        for grp, color, marker in [("top", COLOR_TOP, "o"), ("rest", COLOR_REST, "s")]:
            gs = sub[sub["group"] == grp].set_index("quartile").reindex(QORDER)
            nsub = int(gs["n_subjects"].dropna().iloc[0])
            ax.errorbar(x, gs["mean"], yerr=gs["se"], color=color, marker=marker,
                        lw=2, ms=7, capsize=4, label=f"{grp} (n={nsub})", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(QLAB, fontsize=9)
        ax.set_xlabel("Trial-order quartile", fontsize=10)
        ax.set_ylabel(f"{title} ({unit})", fontsize=10)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="best")
    fig.text(0.025, 0.49, "B. Quartile trajectories  (top solvers vs rest; "
             "the granularity advantage is present from Q1)",
             fontsize=12.5, fontweight="bold")

    fig.suptitle("Within-experiment temporal dynamics: granularity and pacing both "
                 "improve with practice,\nbut the top-solver granularity advantage is "
                 "present from the first quartile  (n = 189, 33 top solvers)",
                 fontsize=12.5, y=0.975)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
