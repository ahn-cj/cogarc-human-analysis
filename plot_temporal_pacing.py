"""
Figure: temporal pacing as a stable individual trait (Section 2).

Three panels:

    A. Histograms of per-subject deliberation time and inter-edit RT,
       showing the range of individual differences across participants.

    B. Top-solver vs rest violin plots for both timing measures,
       with Cliff's delta and Mann-Whitney p annotated.

    C. Scatter of deliberation time vs inter-edit RT per subject,
       coloured by accuracy, with Spearman ρ annotated.

Reads from prior_analysis/chunking_top_solvers_merged.csv.
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
from scipy.stats import spearmanr, mannwhitneyu


COLOR_TOP  = "#2166ac"
COLOR_REST = "#d6604d"


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    gt = int(np.sum(np.subtract.outer(x, y) > 0))
    lt = int(np.sum(np.subtract.outer(x, y) < 0))
    return (gt - lt) / (len(x) * len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/temporal_pacing_figure.png")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.prior_dir,
                                  "chunking_top_solvers_merged.csv"))
    sub = df[["deliberation_time_median", "mean_rt_median",
              "accuracy", "top_solver"]].dropna()

    top  = sub["top_solver"].astype(bool)
    rest = ~top

    dt  = sub["deliberation_time_median"] / 1000   # ms → s
    rt  = sub["mean_rt_median"]                    # stay in ms
    acc = sub["accuracy"]

    dt_top  = dt[top].values;   dt_rest  = dt[rest].values
    rt_top  = rt[top].values;   rt_rest  = rt[rest].values

    fig = plt.figure(figsize=(17, 6))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0],
                   wspace=0.38, left=0.06, right=0.97, top=0.87, bottom=0.13)

    # ── Panel A: overlapping histograms, one per timing measure ──────────────
    axA = fig.add_subplot(gs[0, 0])

    # Normalise both to the same x-axis (z-score across the full sample)
    dt_z = (dt - dt.mean()) / dt.std()
    rt_z = (rt - rt.mean()) / rt.std()

    bins = np.linspace(-3.2, 3.2, 26)
    axA.hist(dt_z, bins=bins, alpha=0.55, color="#4393c3",
             label="deliberation time (s)", density=True)
    axA.hist(rt_z, bins=bins, alpha=0.55, color="#d6604d",
             label="inter-edit RT (ms)", density=True)

    axA.set_xlabel("standardised value (z-score)", fontsize=10)
    axA.set_ylabel("density", fontsize=10)
    axA.set_title(
        "A. Individual spread in temporal pacing\n"
        f"(n = {len(sub)} subjects)",
        loc="left", fontsize=12,
    )
    axA.legend(fontsize=9)
    axA.grid(axis="y", alpha=0.25)

    # Annotate ICC values from document
    axA.text(0.97, 0.96,
             "ICC (deliberation) = .31\nICC (inter-edit RT)  = .20",
             transform=axA.transAxes, ha="right", va="top",
             fontsize=8.5, color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    # ── Panel B: violin plots, top vs rest — z-scored so both fit one axis ────
    axB = fig.add_subplot(gs[0, 1])

    # Z-score each measure across the full sample so they share one y-axis
    dt_z_full = (dt - dt.mean()) / dt.std()
    rt_z_full = (rt - rt.mean()) / rt.std()

    dt_z_top  = dt_z_full[top].values;  dt_z_rest  = dt_z_full[rest].values
    rt_z_top  = rt_z_full[top].values;  rt_z_rest  = rt_z_full[rest].values

    measures  = [dt_z_top, dt_z_rest, rt_z_top, rt_z_rest]
    positions = [0.75, 1.25, 2.75, 3.25]
    colors    = [COLOR_TOP, COLOR_REST, COLOR_TOP, COLOR_REST]

    for vals, pos, color in zip(measures, positions, colors):
        vp = axB.violinplot(vals, positions=[pos], widths=0.38,
                            showmedians=True, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

    # Annotate effect sizes just above the top of the data range
    y_top = axB.get_ylim()[1]
    for x_pair, a_raw, b_raw in [
        (1.0, dt_top, dt_rest),
        (3.0, rt_top, rt_rest),
    ]:
        d = _cliffs_delta(a_raw, b_raw)
        _, p = mannwhitneyu(a_raw, b_raw, alternative="two-sided")
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
        axB.text(x_pair, 3.2, f"δ={d:+.2f}{sig}\n{p_str}",
                 ha="center", va="bottom", fontsize=8.5, color="#555555")

    axB.set_xlim(0.3, 3.7)
    axB.set_ylim(-3.5, 4.8)
    axB.set_xticks([1.0, 3.0])
    axB.set_xticklabels(["deliberation\ntime", "inter-edit\nRT"], fontsize=9.5)
    axB.set_ylabel("z-score (standardised within measure)", fontsize=9)
    axB.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.4)
    axB.set_title("B. Top solvers vs rest\n(Cliff's δ, Mann–Whitney)",
                  loc="left", fontsize=12)
    axB.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=COLOR_TOP,  lw=4, alpha=0.7,
               label=f"top solvers (n={top.sum()})"),
        Line2D([0], [0], color=COLOR_REST, lw=4, alpha=0.7,
               label=f"rest (n={rest.sum()})"),
    ]
    axB.legend(handles=legend_handles, fontsize=9, loc="upper right")

    # ── Panel C: deliberation time vs inter-edit RT scatter ──────────────────
    axC = fig.add_subplot(gs[0, 2])

    sc = axC.scatter(dt, rt,
                     c=acc, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                     s=40, alpha=0.85, edgecolors="grey",
                     linewidths=0.3, zorder=3)

    # Ring top solvers
    axC.scatter(dt[top], rt[top],
                s=90, facecolors="none", edgecolors=COLOR_TOP,
                linewidths=1.6, zorder=4, label=f"top solver (n={top.sum()})")

    rho, p = spearmanr(dt, rt)
    p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
    axC.set_xlabel("median deliberation time (s)", fontsize=10)
    axC.set_ylabel("median inter-edit RT (ms)", fontsize=10)
    axC.set_title(
        f"C. Planning speed vs drawing speed\n(Spearman ρ={rho:.2f}, {p_str})",
        loc="left", fontsize=12,
    )
    axC.legend(fontsize=9)
    axC.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=axC, shrink=0.7, pad=0.04)
    cbar.ax.set_ylabel("accuracy", fontsize=9)

    fig.suptitle(
        f"Section 2: Temporal pacing as a stable individual trait  "
        f"(n={len(sub)} subjects, {top.sum()} top solvers)",
        fontsize=13, y=0.96,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
