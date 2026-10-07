"""Timing measures are stable individual traits (parallel to the chunk-size reliability figure).

A. Split-half scatter for planning time — same plot grammar as the chunk-size panel.
B. Split-half reliability across all measures, with the .70 convention marked.
Pacing aggregated per subject by median; 100 random task-splits.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prospective_chunking import MEAS, STROKE, MIN_TRIALS
from plot_prospective_measures import rt_click_per_trial

DOT = "#4a7fb5"

m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
        .rename(columns={"subject_id": "subject", "task_id": "trial"}))
d = (m.merge(st[["subject", "trial", "cells_per_stroke"]], on=["subject", "trial"])
       .merge(rt_click_per_trial(), on=["subject", "trial"], how="left"))
d["planning_time"] = d.deliberation_time - d.example_view_time_before_first_edit
d["n_tr"] = d.groupby("subject")["trial"].transform("size")
d = d[d.n_tr >= MIN_TRIALS]

def halves(feat, agg, seed=0):
    rng = np.random.default_rng(seed); A, B = [], []
    for _, g in d.groupby("subject"):
        v = g[feat].dropna()
        if len(v) < 8: continue
        i = rng.permutation(len(v)); h = len(i) // 2
        A.append(getattr(v.iloc[i[:h]], agg)()); B.append(getattr(v.iloc[i[h:]], agg)())
    return np.array(A), np.array(B)

def reliability(feat, agg, n=100):
    r = [spearmanr(*halves(feat, agg, s))[0] for s in range(n)]
    return float(np.mean(r)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

MEASURES = [("cells_per_stroke", "mean",   "Chunk size"),
            ("planning_time",    "median", "Planning time"),
            ("deliberation_time","median", "Deliberation"),
            ("rt_click",         "median", "Drawing pace")]

fig = plt.figure(figsize=(14.4, 6.4), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.26,
                      left=0.075, right=0.98, top=0.86, bottom=0.135)

# A — split-half scatter for planning time
A, B = halves("planning_time", "median", seed=0)
r, lo, hi = reliability("planning_time", "median")
ax = fig.add_subplot(gs[0])
ax.scatter(A / 1000, B / 1000, s=46, color=DOT, alpha=.55, edgecolor="none")
lim = [min(A.min(), B.min()) / 1000, max(A.max(), B.max()) / 1000]
ax.plot(lim, lim, ls="--", color="#888", lw=1.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Planning time — random half A  (s)", fontsize=14.5, labelpad=9)
ax.set_ylabel("Random half B  (s)", fontsize=14.5, labelpad=9)
ax.set_title("A. Planning time is a stable individual trait", fontsize=16.5,
             fontweight="bold", pad=12, loc="left")
ax.tick_params(labelsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(.04, .96, f"split-half ρ = {r:.2f}\n(n = {len(A)} participants)", transform=ax.transAxes,
        va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb"))

# B — reliability across measures
ax = fig.add_subplot(gs[1])
vals = [reliability(f, a) for f, a, _ in MEASURES]
y = np.arange(len(MEASURES))[::-1]
for yi, (mn, l, h), (_, _, lab) in zip(y, vals, MEASURES):
    ax.barh(yi, mn, height=.6, color="#2166ac" if mn >= .7 else "#b0b0b0",
            edgecolor="black", lw=.8, zorder=3)
    ax.plot([l, h], [yi, yi], color="black", lw=2, zorder=4)
    ax.text(mn + .02, yi, f"{mn:.2f}", va="center", fontsize=14.5, fontweight="bold")
ax.axvline(.70, color="#c62828", ls="--", lw=2)
ax.text(.705, len(MEASURES) - .35, "conventional\nthreshold (.70)", color="#c62828",
        fontsize=12, va="top")
ax.set_yticks(y); ax.set_yticklabels([lab for _, _, lab in MEASURES], fontsize=14.5)
ax.set_xlim(0, 1.05); ax.set_xlabel("Split-half reliability (ρ)", fontsize=14.5, labelpad=9)
ax.set_title("B. All measures are reliable individual traits", fontsize=16.5,
             fontweight="bold", pad=12, loc="left")
ax.tick_params(axis="x", labelsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.suptitle("Timing, like chunking, is a stable property of the person  (n = 195, 100 random task-splits)",
             fontsize=17.5, fontweight="bold", y=0.965)
for ext in ("png", "svg"):
    fig.savefig(f"prior_analysis/timing_reliability.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print("saved prior_analysis/timing_reliability.{png,svg}")
for (f, a, lab), (mn, l, h) in zip(MEASURES, vals):
    print(f"  {lab:16s} r={mn:.3f}  [{l:.3f}, {h:.3f}]")
