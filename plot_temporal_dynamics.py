"""
Figure: within-experiment temporal dynamics of pacing and chunking style.

Two rows × three columns:

    Row 1 (early vs late):  scatter of each subject's first-half mean
                            against their second-half mean, with Spearman ρ
                            annotated.  Diagonal line = perfect stability.

    Row 2 (quartile trajectories):  per-quartile mean ± SE for top solvers
                                     vs rest.

Reads from prior_analysis/ (outputs of temporal_dynamics.py).
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


COLOR_TOP   = "#2166ac"
COLOR_REST  = "#d6604d"
COLOR_OTHER = "#808080"

MEASURES = [
    ("rt",                "log",   "Inter-edit RT",          "log(s)"),
    ("deliberation_time", "log",   "Deliberation time",      "log(s)"),
    ("num_chunks",        "linear","n chunks per trajectory","count"),
]
QUARTILE_ORDER = ["Q1 (1-19)", "Q2 (20-37)", "Q3 (38-56)", "Q4 (57-75)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/temporal_dynamics_figure.png")
    args = ap.parse_args()

    el    = pd.read_csv(os.path.join(args.prior_dir, "temporal_dynamics_early_late.csv"))
    stab  = pd.read_csv(os.path.join(args.prior_dir, "temporal_dynamics_stability.csv"))
    quart = pd.read_csv(os.path.join(args.prior_dir, "temporal_dynamics_quartiles.csv"))
    stab_idx = stab.set_index("outcome")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5),
                             gridspec_kw=dict(hspace=0.42, wspace=0.32,
                                              left=0.07, right=0.97,
                                              top=0.92, bottom=0.08))

    # ── Row 1: early vs late scatter ──────────────────────────────────────────
    for ax, (col, kind, title, unit) in zip(axes[0], MEASURES):
        sub = el[el["outcome"] == col].dropna(subset=["early", "late"])
        ax.scatter(sub["early"], sub["late"],
                   s=25, alpha=0.55, color=COLOR_OTHER,
                   edgecolors="white", linewidths=0.4)

        # Identity line
        all_vals = np.concatenate([sub["early"].values, sub["late"].values])
        lo, hi = np.percentile(all_vals, [1, 99])
        pad = 0.04 * (hi - lo)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color="black", ls="--", lw=0.7, alpha=0.5,
                label="identity (no change)")
        ax.set_xlim(lims); ax.set_ylim(lims)

        rho = float(stab_idx.loc[col, "spearman_rho"])
        n   = int(stab_idx.loc[col, "n_subjects"])
        ax.text(0.04, 0.96,
                f"Spearman ρ = {rho:.2f}\nn = {n} subjects",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.92))

        ax.set_xlabel(f"early half ({unit})", fontsize=10)
        ax.set_ylabel(f"late half ({unit})",  fontsize=10)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

    fig.text(0.025, 0.96,
             "A. Within-subject early-vs-late stability   "
             "(each point = one subject)",
             fontsize=12.5, fontweight="bold")

    # ── Row 2: quartile trajectories ──────────────────────────────────────────
    for ax, (col, kind, title, unit) in zip(axes[1], MEASURES):
        sub = quart[quart["outcome"] == col]
        x = np.arange(len(QUARTILE_ORDER))
        for grp, color, marker in [("top",  COLOR_TOP,  "o"),
                                   ("rest", COLOR_REST, "s")]:
            grp_sub = sub[sub["group"] == grp].set_index("quartile").reindex(QUARTILE_ORDER)
            n_subj = int(grp_sub["n_subjects"].dropna().iloc[0])
            ax.errorbar(x, grp_sub["mean"], yerr=grp_sub["se"],
                        color=color, marker=marker, lw=2, ms=7,
                        capsize=4, label=f"{grp} (n={n_subj})", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(QUARTILE_ORDER, fontsize=9, rotation=15)
        ax.set_xlabel("Trial-order quartile", fontsize=10)
        ax.set_ylabel(f"{title} ({unit})", fontsize=10)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="best")

    fig.text(0.025, 0.49,
             "B. Quartile trajectories   "
             "(top solvers vs rest; mean ± SE of per-subject means)",
             fontsize=12.5, fontweight="bold")

    fig.suptitle(
        "Within-experiment temporal dynamics: pacing converges with practice, "
        "chunking style does not  (n = 189 subjects with ≥40 trials, 33 top solvers)",
        fontsize=13, y=0.985,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
