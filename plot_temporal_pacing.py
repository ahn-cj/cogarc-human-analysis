"""
Figure: temporal pacing (Part 2) — continuous accuracy, no top-solver split.

Two panels:
    A. Individual spread in the two timing measures (deliberation time, inter-edit
       RT), showing the range of stable between-person differences.
    B. Pacing × accuracy decomposed: execution pace (inter-edit RT during drawing)
       correlates with accuracy bivariately but is drag-confounded — measured
       drag-free it is ~0; planning pace (deliberation) is null overall but its
       post-example component is a genuine predictor.

Reads chunking_top_solvers_merged.csv, controlled_accuracy_correlations.csv,
drag_sensitivity_profile.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out", default="prior_analysis/temporal_pacing_figure.png")
    args = ap.parse_args()
    P = args.prior_dir

    df = pd.read_csv(os.path.join(P, "chunking_top_solvers_merged.csv"))
    sub = df[["deliberation_time_median", "mean_rt_median", "accuracy"]].dropna()
    dt = sub["deliberation_time_median"] / 1000.0   # s
    rt = sub["mean_rt_median"]                       # ms

    ctrl = pd.read_csv(os.path.join(P, "controlled_accuracy_correlations.csv")).set_index("feature")
    ds = pd.read_csv(os.path.join(P, "drag_sensitivity_profile.csv"))
    exec_raw = float(ctrl.loc["mean_rt_between_edits", "rho_raw"])
    exec_free = float(spearmanr(ds["rt_click"], ds["accuracy"]).statistic)
    plan_raw = float(ctrl.loc["deliberation_time", "rho_raw"])
    plan_ctrl = float(ctrl.loc["deliberation_time", "rho_ctrl"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw=dict(wspace=0.28, left=0.07,
                                                    right=0.97, top=0.84, bottom=0.13))

    # ── Panel A: individual spread ────────────────────────────────────────────
    dt_z = (dt - dt.mean()) / dt.std()
    rt_z = (rt - rt.mean()) / rt.std()
    bins = np.linspace(-3.2, 3.2, 26)
    axA.hist(dt_z, bins=bins, alpha=0.55, color="#4393c3", density=True,
             label="deliberation time")
    axA.hist(rt_z, bins=bins, alpha=0.55, color="#d6604d", density=True,
             label="inter-edit RT")
    axA.set_xlabel("standardised value (z-score)", fontsize=10)
    axA.set_ylabel("density", fontsize=10)
    axA.set_title(f"A. Individual spread in temporal pacing  (n = {len(sub)})",
                  loc="left", fontsize=12)
    axA.text(0.97, 0.96, "ICC (deliberation) = .31\nICC (inter-edit RT)  = .20",
             transform=axA.transAxes, ha="right", va="top", fontsize=8.5,
             color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    axA.legend(fontsize=9, loc="upper left")
    axA.grid(axis="y", alpha=0.25)
    axA.spines[["top", "right"]].set_visible(False)

    # ── Panel B: pacing × accuracy decomposition ──────────────────────────────
    items = [("execution pace\n(inter-edit RT)", exec_raw, "raw", "#999999"),
             ("execution pace\n(drag-free)", exec_free, "clean", "#b2182b"),
             ("planning pace\n(deliberation, raw)", plan_raw, "raw", "#999999"),
             ("planning pace\n(post-example)", plan_ctrl, "clean", "#2166ac")]
    x = np.arange(len(items))
    vals = [v for _, v, _, _ in items]
    cols = [c for _, _, _, c in items]
    axB.bar(x, vals, color=cols, edgecolor="black", lw=0.6, width=0.7)
    for xi, v in zip(x, vals):
        axB.text(xi, v + (0.012 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", fontsize=9)
    axB.axhline(0, color="black", lw=0.7)
    axB.axhspan(-0.10, 0.10, color="#f5f5f5", zorder=0)
    axB.set_xticks(x)
    axB.set_xticklabels([lab for lab, *_ in items], fontsize=8.5)
    axB.set_ylabel("Spearman ρ with accuracy", fontsize=10)
    axB.set_ylim(-0.45, 0.2)
    axB.set_title("B. Execution pace is drag-confounded; planning pace is the genuine effect\n"
                  "(grey = raw; red = collapses when cleaned; blue = survives)",
                  loc="left", fontsize=11)
    axB.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Temporal pacing: a stable trait whose accuracy-relevant component is "
                 "post-example planning speed", fontsize=12.5, x=0.07, ha="left", y=0.97)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
