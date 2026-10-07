"""Timing measures vs accuracy (concurrent), in the same style as the chunking scatters.

Pacing measures are aggregated per subject by MEDIAN (matches controls_regression.py AGG_RULE
and reproduces the chapter: deliberation -.01, example view +.09, planning -.19, pace +.09).
A trend line is drawn only when the rank correlation is reliable; otherwise a flat reference
line, so the picture never contradicts rho.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prospective_chunking import boot_ci, MEAS, STROKE, MIN_TRIALS
from plot_prospective_measures import rt_click_per_trial

DOT, TREND = "#4a7fb5", "#c62828"

def load():
    m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
    st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
            .rename(columns={"subject_id": "subject", "task_id": "trial"}))
    d = (m.merge(st[["subject", "trial", "cells_per_stroke"]], on=["subject", "trial"])
           .merge(rt_click_per_trial(), on=["subject", "trial"], how="left"))
    d["ok"] = (d.final_outcome.astype(str) == "success").astype(float)
    d["planning_time"] = d.deliberation_time - d.example_view_time_before_first_edit
    d["n_tr"] = d.groupby("subject")["trial"].transform("size")
    return d[d.n_tr >= MIN_TRIALS]

PANELS = [
    ("deliberation_time", "Deliberation time (s)",
     "A. Total deliberation → no relationship", 1000),
    ("example_view_time_before_first_edit", "Time studying examples (s)",
     "B. Studying the examples → no relationship", 1000),
    ("planning_time", "Planning time, edit view (s)",
     "C. Faster planning → higher accuracy", 1000),
    ("rt_click", "Drawing pace (inter-click RT, ms)",
     "D. Drawing pace → no relationship", 1),
]

def main():
    d = load(); acc = d.groupby("subject")["ok"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), facecolor="white")
    for ax, (feat, xlab, title, scale) in zip(axes.ravel(), PANELS):
        v = d.groupby("subject")[feat].median() / scale
        j = pd.concat([v.rename("x"), acc.rename("y")], axis=1).dropna()
        x, y = j.x.values, j.y.values
        r, p = spearmanr(x, y); lo, hi = boot_ci(x, y)
        ax.scatter(x, y, s=44, color=DOT, alpha=.55, edgecolor="none")
        ax.set_xscale("log")
        if p < .05:
            z = np.polyfit(np.log(x), y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, np.polyval(z, np.log(xs)), color=TREND, lw=3.2)
        else:
            ax.axhline(y.mean(), color="#999999", lw=2.4, ls="--")
        ax.set_xlabel(xlab, fontsize=14, labelpad=8)
        ax.set_ylabel("Accuracy", fontsize=14, labelpad=8)
        ax.set_title(title, fontsize=15.5, fontweight="bold", pad=11, loc="left")
        ax.tick_params(labelsize=12.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        sig = "" if p < .05 else "   n.s."
        ax.text(.97, .06, f"ρ = {r:+.2f}  [{lo:+.2f}, {hi:+.2f}]{sig}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=13, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb"))
        print(f"{feat:<36} rho={r:+.3f} p={p:.4g}")
    fig.suptitle("Deliberation splits into two parts with different relationships to accuracy  (n = 195)",
                 fontsize=17, fontweight="bold", y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("png", "svg"):
        fig.savefig(f"prior_analysis/timing_accuracy.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("saved prior_analysis/timing_accuracy.{png,svg}")

if __name__ == "__main__":
    main()
