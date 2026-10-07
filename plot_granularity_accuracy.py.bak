"""
Figure (Part 3): chunk granularity predicts accuracy, and survives controls.

    A. cells per stroke vs accuracy (scatter + LOWESS).
    B. n strokes per trajectory vs accuracy (scatter + LOWESS).
    C. raw vs controlled Spearman ρ (residualised on task difficulty + log
       trajectory length) for both measures.

Reads stroke_chunking_profile.csv, stroke_chunking_per_st.csv, trial_scores.csv,
chunking_top_solvers_merged.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

TRIAL_SCORES = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
                "Behavioral data/trial_scores.csv")
N_BOOT = 5000


def sp_ci(x, y, n_boot=N_BOOT, seed=42):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot); n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b[i], _ = spearmanr(x[idx], y[idx])
    return rho, float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)), p


def controlled_rho(per_st, feat, acc):
    ts = pd.read_csv(TRIAL_SCORES)
    ts["task_id"] = ts["trial"].str.replace(".json", "", regex=False)
    ts = ts.rename(columns={"attempt_score": "task_difficulty"})
    d = per_st.merge(ts[["task_id", "task_difficulty"]], on="task_id", how="left")
    d["log_len"] = np.log(d["n_edits"].clip(lower=1))
    d = d.dropna(subset=[feat, "task_difficulty", "log_len"])
    X = np.column_stack([np.ones(len(d)), d["task_difficulty"], d["log_len"]])
    beta, *_ = np.linalg.lstsq(X, d[feat].values, rcond=None)
    d["_r"] = d[feat].values - X @ beta
    prof = d.groupby("subject_id")["_r"].mean().reset_index().rename(columns={"subject_id": "subject"})
    prof["subject"] = prof["subject"].astype(str)
    prof = prof.merge(acc, on="subject", how="inner")
    return sp_ci(prof["_r"].values, prof["accuracy"].values)


def scatter(ax, x, y, xlabel, title, rho_ci, color="#2166ac"):
    ax.scatter(x, y, s=16, alpha=0.45, color=color, edgecolors="none")
    lo = lowess(y, x, frac=0.7, return_sorted=True)
    ax.plot(lo[:, 0], lo[:, 1], color="#b2182b", lw=2.2)
    rho, clo, chi, p = rho_ci
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    ax.text(0.04, 0.96, f"ρ = {rho:+.2f} [{clo:+.2f}, {chi:+.2f}]{star}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("accuracy", fontsize=10)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prior_analysis/granularity_accuracy_figure.png")
    args = ap.parse_args()
    P = "prior_analysis"

    prof = pd.read_csv(f"{P}/stroke_chunking_profile.csv")
    prof["subject"] = prof["subject"].astype(str)
    per_st = pd.read_csv(f"{P}/stroke_chunking_per_st.csv")
    acc = prof[["subject", "accuracy"]]

    cps = sp_ci(prof["cells_per_stroke"].values, prof["accuracy"].values)
    nst = sp_ci(prof["n_strokes"].values, prof["accuracy"].values)
    cps_c = controlled_rho(per_st, "cells_per_stroke", acc)
    nst_c = controlled_rho(per_st, "n_strokes", acc)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8),
                           gridspec_kw=dict(wspace=0.30, left=0.06, right=0.97,
                                            top=0.84, bottom=0.14))
    scatter(ax[0], prof["cells_per_stroke"], prof["accuracy"],
            "cells per stroke (coarser →)", "A. Coarser chunking → higher accuracy", cps)
    scatter(ax[1], prof["n_strokes"], prof["accuracy"],
            "n strokes per trajectory (finer →)", "B. More strokes → lower accuracy", nst,
            color="#4393c3")

    # Panel C: raw vs controlled
    labels = ["cells per\nstroke", "n strokes /\ntrajectory"]
    raw = [cps[0], nst[0]]
    ctl = [cps_c[0], nst_c[0]]
    x = np.arange(2); w = 0.36
    ax[2].bar(x - w/2, raw, w, color="#cccccc", edgecolor="black", lw=.5, label="raw")
    ax[2].bar(x + w/2, ctl, w, color="#2166ac", edgecolor="black", lw=.5,
              label="+ difficulty / length controls")
    for xi, r, c in zip(x, raw, ctl):
        ax[2].text(xi - w/2, r + (0.01 if r >= 0 else -0.03), f"{r:+.2f}", ha="center", fontsize=8.5)
        ax[2].text(xi + w/2, c + (0.01 if c >= 0 else -0.03), f"{c:+.2f}", ha="center",
                   fontsize=8.5, fontweight="bold")
    ax[2].axhline(0, color="black", lw=0.7)
    ax[2].axhspan(-0.10, 0.10, color="#f5f5f5", zorder=0)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9.5)
    ax[2].set_ylabel("Spearman ρ with accuracy", fontsize=10)
    ax[2].set_ylim(-0.45, 0.35)
    ax[2].set_title("C. The effect survives controls", loc="left", fontsize=11.5, fontweight="bold")
    ax[2].legend(fontsize=8.5, loc="lower right")
    ax[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Chunk granularity predicts accuracy (continuous, n = 195) and survives "
                 "task-difficulty and trajectory-length controls",
                 fontsize=12.5, x=0.06, ha="left", y=0.95)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}  (cps {cps[0]:+.3f}->{cps_c[0]:+.3f}; nstrokes {nst[0]:+.3f}->{nst_c[0]:+.3f})")


if __name__ == "__main__":
    main()
