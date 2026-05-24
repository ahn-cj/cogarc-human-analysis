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

    fig = plt.figure(figsize=(17, 5.4))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 1.0],
                   wspace=0.34, left=0.06, right=0.97, top=0.86, bottom=0.14)

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

    # ── Panel B: model winner stacked bar + per-subject exp β histogram ──────
    axB = fig.add_subplot(gs[0, 1])

    n_fit  = int(rt_win["n_subjects_fit"])
    n_exp  = int(rt_win["n_exp_wins"])
    n_pl   = int(rt_win["n_pl_wins"])
    pct_exp = n_exp / n_fit

    # Inset stacked-bar at top of panel
    axB.barh([0], [n_pl],            height=0.4, color=COLOR_PL,
             alpha=0.85, label=f"power law wins ({n_pl}/{n_fit}, {1-pct_exp:.0%})")
    axB.barh([0], [n_exp], left=[n_pl], height=0.4, color=COLOR_EXP,
             alpha=0.85, label=f"exponential wins ({n_exp}/{n_fit}, {pct_exp:.0%})")
    axB.text(n_pl + n_exp + 4, 0,
             f"  median ΔAIC = {rt_win['median_delta_aic_pl_minus_exp']:+.2f}\n"
             f"  (positive ⇒ exponential preferred)",
             va="center", ha="left", fontsize=9, color="#555555", style="italic")
    axB.set_yticks([])
    axB.set_xlim(0, n_fit + 80)
    axB.set_xlabel(f"subjects (n = {n_fit} with ≥ 30 valid trials)", fontsize=9.5)
    axB.set_title("B. Which model fits per-subject data better?",
                  loc="left", fontsize=12, fontweight="bold")
    axB.legend(fontsize=8.5, loc="upper right")
    for spine in ("top", "right", "left"):
        axB.spines[spine].set_visible(False)

    # ── Panel C: per-subject exp β vs accuracy, coloured by top solver ────────
    axC = fig.add_subplot(gs[0, 2])
    rest_mask = ~rt_sub["top_solver"].astype(bool)
    top_mask  =  rt_sub["top_solver"].astype(bool)

    axC.scatter(rt_sub.loc[rest_mask, "accuracy"],
                rt_sub.loc[rest_mask, "exp_b"],
                s=35, alpha=0.55, color=COLOR_REST,
                edgecolors="white", linewidths=0.4, label=f"rest (n={int(rest_mask.sum())})")
    axC.scatter(rt_sub.loc[top_mask, "accuracy"],
                rt_sub.loc[top_mask, "exp_b"],
                s=70, alpha=0.85, color=COLOR_TOP, marker="o",
                edgecolors="white", linewidths=0.6, label=f"top (n={int(top_mask.sum())})")

    axC.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    rho, p = spearmanr(rt_sub["accuracy"], rt_sub["exp_b"])
    p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
    axC.text(0.04, 0.04,
             f"Spearman ρ = {rho:+.2f}\n{p_str}",
             transform=axC.transAxes, ha="left", va="bottom",
             fontsize=10, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    axC.set_xlabel("Overall accuracy", fontsize=10)
    axC.set_ylabel("Per-subject exponential β\n(more negative = faster speedup)", fontsize=10)
    axC.set_title("C. Top solvers have flatter learning curves\n(consistent with Ackerman 1988)",
                  loc="left", fontsize=11.5, fontweight="bold")
    axC.legend(fontsize=9, loc="upper right")
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
