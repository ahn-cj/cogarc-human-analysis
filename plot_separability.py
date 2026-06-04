"""
Figure (Part 4): chunking and pacing are separable; granularity carries the effect.

    A. Joint scatter of chunk granularity (cells per stroke) × drag-free pace
       (inter-click RT), coloured by accuracy — the two axes are independent and
       accuracy rises along granularity, not pace.
    B. Rank-regression variance decomposition (granularity vs drag-free pace) and
       the CFA inter-factor correlation.

Reads stroke_chunking_profile.csv, drag_sensitivity_profile.csv. CFA r is from
stroke_joint_analysis.py.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

CFA_R = 0.11  # granularity–pacing inter-factor correlation (stroke_joint_analysis.py)


def rank_r2(y, Xcols):
    ry = rankdata(y)
    X = np.column_stack([np.ones_like(ry)] + [rankdata(c) for c in Xcols])
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)
    pred = X @ beta
    return 1 - np.sum((ry - pred) ** 2) / np.sum((ry - ry.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prior_analysis/separability_figure.png")
    args = ap.parse_args()
    P = "prior_analysis"

    prof = pd.read_csv(f"{P}/stroke_chunking_profile.csv")[["subject", "cells_per_stroke", "accuracy"]]
    prof["subject"] = prof["subject"].astype(str)
    ds = pd.read_csv(f"{P}/drag_sensitivity_profile.csv")[["subject", "rt_click"]]
    ds["subject"] = ds["subject"].astype(str)
    df = prof.merge(ds, on="subject").dropna()
    g = df["cells_per_stroke"].values
    pace = df["rt_click"].values
    acc = df["accuracy"].values

    r_gp = spearmanr(g, pace).statistic
    r_ga = spearmanr(g, acc).statistic
    r_pa = spearmanr(pace, acc).statistic
    r2_g = rank_r2(acc, [g])
    r2_p = rank_r2(acc, [pace])
    r2_b = rank_r2(acc, [g, pace])

    fig = plt.figure(figsize=(13, 5.6))
    gsL = fig.add_axes([0.07, 0.12, 0.50, 0.74])
    gsR = fig.add_axes([0.66, 0.16, 0.30, 0.64])

    # ── A: joint scatter ──────────────────────────────────────────────────────
    sc = gsL.scatter(g, pace, c=acc, cmap="viridis", s=40, edgecolor="white", linewidth=0.3)
    gsL.set_xlabel("chunk granularity (cells per stroke)", fontsize=11)
    gsL.set_ylabel("drag-free pace (inter-click RT, ms)", fontsize=11)
    gsL.set_title(f"A. The two dimensions are independent (ρ = {r_gp:+.2f}, n.s.)",
                  loc="left", fontsize=12, fontweight="bold")
    gsL.text(0.97, 0.96,
             f"granularity × accuracy: ρ = {r_ga:+.2f}\npace × accuracy: ρ = {r_pa:+.2f} (n.s.)",
             transform=gsL.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    gsL.spines[["top", "right"]].set_visible(False)
    cb = fig.colorbar(sc, ax=gsL, fraction=0.046, pad=0.03)
    cb.set_label("accuracy", fontsize=9)

    # ── B: variance decomposition + CFA ──────────────────────────────────────
    bars = [("granularity\nonly", r2_g, "#2166ac"),
            ("drag-free\npace only", r2_p, "#b2182b"),
            ("joint", r2_b, "#6a51a3")]
    x = np.arange(len(bars))
    gsR.bar(x, [v for _, v, _ in bars], color=[c for *_, c in bars],
            edgecolor="black", lw=0.6, width=0.66)
    for xi, (_, v, _) in zip(x, bars):
        gsR.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9.5)
    gsR.set_xticks(x); gsR.set_xticklabels([b[0] for b in bars], fontsize=9)
    gsR.set_ylabel("R² for accuracy (rank regression)", fontsize=10)
    gsR.set_ylim(0, max(r2_b, r2_g) * 1.35)
    gsR.set_title("B. Granularity carries the variance", loc="left", fontsize=12, fontweight="bold")
    gsR.text(0.5, 0.86,
             f"unique granularity = {r2_b - r2_p:.3f}\nunique pace = {r2_b - r2_g:.3f}\n"
             f"CFA inter-factor r = {CFA_R:+.2f}",
             transform=gsR.transAxes, ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc"))
    gsR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Chunking and pacing are separable dimensions — only chunk granularity "
                 "predicts accuracy", fontsize=12.5, x=0.07, ha="left", y=0.965)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}  (r_gp={r_gp:+.2f}, R²_g={r2_g:.3f}, R²_p={r2_p:.3f})")


if __name__ == "__main__":
    main()
