"""
Figure: practice curves (power law vs exponential) for inter-edit RT.

    A. Group-mean RT by trial order with both fits overlaid.
       Shows the population learning curve and which model fits
       the aggregated data better.

    B. Per-subject model winner (exponential vs power-law) + median ΔAIC,
       plus a histogram of per-subject exponential β values (the learning
       rate; more negative = faster speedup over the experiment).

    C. Per-subject exponential β vs overall accuracy, coloured by
       top-solver status. Tests Ackerman (1988)-style claim that
       higher-ability subjects start closer to asymptote and so
       have shallower learning curves.

Reads from prior_analysis/ (outputs of practice_curves.py).
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import spearmanr


BEHAV_CSV = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
             "Behavioral data/behavioral_measures_filtered.csv")

COLOR_PL  = "#762a83"
COLOR_EXP = "#1b7837"
COLOR_TOP  = "#2166ac"
COLOR_REST = "#d6604d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--csv", default=BEHAV_CSV)
    ap.add_argument("--out",
                    default="prior_analysis/practice_curves_figure.png")
    args = ap.parse_args()

    per_subj = pd.read_csv(os.path.join(args.prior_dir,
                                         "practice_curves_per_subject.csv"))
    group    = pd.read_csv(os.path.join(args.prior_dir,
                                         "practice_curves_group.csv"))
    winners  = pd.read_csv(os.path.join(args.prior_dir,
                                         "practice_curves_winners.csv"))

    rt_sub  = per_subj[per_subj["measure"] == "rt"].copy()
    rt_grp  = group  [group  ["measure"] == "rt"].iloc[0]
    rt_win  = winners[winners["measure"] == "rt"].iloc[0]

    # Recompute group geometric-mean RT per trial order (for plotting).
    # We use geometric mean (= mean in log-space) because RT is right-skewed
    # with extreme outliers; arithmetic means are dominated by a handful of
    # very long trials per bin.
    behav = pd.read_csv(args.csv)
    behav["correct"] = behav["final_result"].astype(str).str.lower() == "success"
    n_per_subj = behav.groupby("subject")["trial"].nunique()
    keep = n_per_subj[n_per_subj >= 40].index
    sub  = behav[behav["subject"].isin(keep)].dropna(subset=["rt", "order"])
    sub  = sub[sub["rt"] > 0].copy()
    sub["log_rt"] = np.log(sub["rt"])
    g_per_order = (sub.groupby("order")["log_rt"]
                       .agg(["mean", "sem", "count"]).reset_index())
    mean_per_order = pd.DataFrame({
        "order": g_per_order["order"],
        "mean":  np.exp(g_per_order["mean"]),              # geometric mean
        # SEM in original units (delta-method approximation)
        "sem":   np.exp(g_per_order["mean"]) * g_per_order["sem"],
        "count": g_per_order["count"],
    })

    fig = plt.figure(figsize=(13, 5.4))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0],
                   wspace=0.30, left=0.07, right=0.97, top=0.86, bottom=0.14)

    # ── Panel A: group mean RT + both fits ────────────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    t   = mean_per_order["order"].values.astype(float)
    yba = mean_per_order["mean"].values
    sem = mean_per_order["sem"].values

    axA.errorbar(t, yba, yerr=sem, fmt="o", color="#444444", ms=4,
                 alpha=0.6, capsize=0, label="group mean ± SE")

    # Overlay fits — exponential
    t_smooth = np.linspace(1, 75, 200)
    y_exp_log = rt_grp["exp_b"] * t_smooth + np.log(yba[0]) - rt_grp["exp_b"] * 1
    # the above intercept anchor isn't exactly right — refit on group means here
    # for cleaner visuals
    import statsmodels.api as sm
    X = sm.add_constant(t)
    res_pl  = sm.OLS(np.log(yba), sm.add_constant(np.log(t))).fit()
    res_exp = sm.OLS(np.log(yba), X).fit()
    y_pl_smooth  = np.exp(res_pl.params[0]  + res_pl.params[1]  * np.log(t_smooth))
    y_exp_smooth = np.exp(res_exp.params[0] + res_exp.params[1] * t_smooth)

    axA.plot(t_smooth, y_pl_smooth,  color=COLOR_PL,  lw=2.0,
             label=f"power law  (b = {rt_grp['pl_b']:+.3f},  R² = {rt_grp['pl_r2_log']:.2f})")
    axA.plot(t_smooth, y_exp_smooth, color=COLOR_EXP, lw=2.0,
             label=f"exponential (b = {rt_grp['exp_b']:+.4f},  R² = {rt_grp['exp_r2_log']:.2f})")

    axA.set_xlabel("Trial order (1–75)", fontsize=10)
    axA.set_ylabel("Geometric-mean RT (s)", fontsize=10)
    axA.set_title("A. Group-aggregated practice curve",
                  loc="left", fontsize=12, fontweight="bold")
    # sensible y-range around the data
    lo, hi = np.percentile(yba, [2, 98])
    axA.set_ylim(0.7 * lo, 1.3 * hi)
    axA.legend(fontsize=9, loc="upper right")
    axA.grid(alpha=0.25)

    # ── Panel B: per-subject exp β vs continuous accuracy ────────────────────
    axC = fig.add_subplot(gs[0, 1])
    axC.scatter(rt_sub["accuracy"], rt_sub["exp_b"],
                s=42, alpha=0.7, color="#2166ac",
                edgecolors="white", linewidths=0.4)

    # LOWESS trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lo = lowess(rt_sub["exp_b"], rt_sub["accuracy"], frac=0.7, return_sorted=True)
    axC.plot(lo[:, 0], lo[:, 1], color="#b2182b", lw=2)

    axC.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    rho, p = spearmanr(rt_sub["accuracy"], rt_sub["exp_b"])
    p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
    axC.text(0.04, 0.04,
             f"Spearman ρ = {rho:+.2f}\n{p_str}",
             transform=axC.transAxes, ha="left", va="bottom",
             fontsize=10, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    axC.set_xlabel("Overall accuracy (continuous)", fontsize=10)
    axC.set_ylabel("Per-subject exponential β\n(more negative = faster speedup)", fontsize=10)
    axC.set_title("B. Flatter learning curves go with higher accuracy\n(Ackerman 1988)",
                  loc="left", fontsize=11.5, fontweight="bold")
    axC.grid(alpha=0.25)

    fig.suptitle(
        "Practice-curve fits for inter-edit RT: per-subject exponential vs power-law learning",
        fontsize=13, y=0.985,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
