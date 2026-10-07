"""Planning time vs accuracy, raw and after task controls (parallel to the chunking figure).

Left  : raw edit-view planning time (log axis)   rho ~ -.19
Right : planning time residualised on task_difficulty + log(trajectory_length) + n_examples,
        per-subject median of residuals (from controls_regression.py)   rho ~ -.21
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prospective_chunking import boot_ci, MEAS, STROKE, MIN_TRIALS
from plot_prospective_measures import rt_click_per_trial

DOT, TREND = "#4a7fb5", "#c62828"

m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
        .rename(columns={"subject_id": "subject", "task_id": "trial"}))
d = m.merge(st[["subject", "trial", "cells_per_stroke"]], on=["subject", "trial"])
d["ok"] = (d.final_outcome.astype(str) == "success").astype(float)
d["planning_time"] = d.deliberation_time - d.example_view_time_before_first_edit
d["n_tr"] = d.groupby("subject")["trial"].transform("size")
d = d[d.n_tr >= MIN_TRIALS]
acc = d.groupby("subject")["ok"].mean()
raw = d.groupby("subject")["planning_time"].median() / 1000.0

prof = pd.read_csv("prior_analysis/controlled_subject_profile.csv", dtype={"subject": str})
ctrl = prof.set_index("subject")["planning_time_resid"] / 1000.0

fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), facecolor="white")
for ax, (series, xlab, title, logx) in zip(axes, [
        (raw,  "Planning time, edit view (s)",
         "A. Raw planning time", True),
        (ctrl, "Planning time, residual (s)\ncontrolled for difficulty, trajectory length, n examples",
         "B. After task controls", False)]):
    j = pd.concat([series.rename("x"), acc.rename("y")], axis=1).dropna()
    x, y = j.x.values, j.y.values
    r, p = spearmanr(x, y); lo, hi = boot_ci(x, y)
    ax.scatter(x, y, s=46, color=DOT, alpha=.55, edgecolor="none")
    if logx: ax.set_xscale("log")
    xf = np.log(x) if logx else x
    z = np.polyfit(xf, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, np.polyval(z, np.log(xs) if logx else xs), color=TREND, lw=3.2)
    if not logx: ax.axvline(0, color="#bbb", lw=1.2, ls=":")
    ax.set_xlabel(xlab, fontsize=14, labelpad=9)
    ax.set_ylabel("Accuracy", fontsize=14.5, labelpad=9)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12, loc="left")
    ax.tick_params(labelsize=13)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(.97, .06, f"ρ = {r:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13.5,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb"))
    print(f"{title:26s} rho={r:+.3f} p={p:.4g} n={len(x)}")

fig.suptitle("Faster edit-view planning predicts higher accuracy, and the effect survives task controls  (n = 195)",
             fontsize=16.5, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "svg"):
    fig.savefig(f"prior_analysis/planning_controlled.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print("saved prior_analysis/planning_controlled.{png,svg}")
