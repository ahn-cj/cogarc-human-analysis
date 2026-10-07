"""Chunk size vs planning time: the two accuracy predictors are only moderately correlated.

Establishes separability -- coarser chunkers plan slightly faster, but the relationship is far
from redundant (Spearman ~ -.37), which is what licenses treating them as two dimensions.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prospective_chunking import boot_ci, MEAS, STROKE, MIN_TRIALS

DOT, TREND = "#4a7fb5", "#c62828"

m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
        .rename(columns={"subject_id": "subject", "task_id": "trial"}))
d = m.merge(st[["subject", "trial", "cells_per_stroke"]], on=["subject", "trial"])
d["ok"] = (d.final_outcome.astype(str) == "success").astype(float)
d["planning_time"] = d.deliberation_time - d.example_view_time_before_first_edit
d["n_tr"] = d.groupby("subject")["trial"].transform("size")
d = d[d.n_tr >= MIN_TRIALS]
p = pd.DataFrame({"chunk": d.groupby("subject")["cells_per_stroke"].mean(),
                  "plan":  d.groupby("subject")["planning_time"].median() / 1000,
                  "acc":   d.groupby("subject")["ok"].mean()}).dropna()

r, pv = spearmanr(p.chunk, p.plan); lo, hi = boot_ci(p.chunk.values, p.plan.values)

fig, ax = plt.subplots(figsize=(9.0, 7.0), facecolor="white")
sc = ax.scatter(p.chunk, p.plan, s=62, c=p.acc, cmap="viridis",
                alpha=.85, edgecolor="white", lw=.5, vmin=.4, vmax=1.0)
z = np.polyfit(np.log(p.chunk), np.log(p.plan), 1)
xs = np.linspace(p.chunk.min(), p.chunk.max(), 200)
ax.plot(xs, np.exp(np.polyval(z, np.log(xs))), color=TREND, lw=3.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 2, 3, 4, 6, 8, 10]); ax.set_yticks([5, 10, 20, 40])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Chunk size  (cells per stroke, coarser →)", fontsize=15, labelpad=9)
ax.set_ylabel("Planning time  (s, slower →)", fontsize=15, labelpad=9)
ax.set_title("The two predictors are related, but far from redundant",
             fontsize=17, fontweight="bold", pad=14)
ax.tick_params(labelsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(.03, .05, f"ρ = {r:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]\n"
                  f"shared variance ≈ {100*r**2:.0f}%",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb"))
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Accuracy", fontsize=14)
cb.ax.tick_params(labelsize=12)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"prior_analysis/chunk_vs_planning.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print(f"rho={r:+.3f} p={pv:.3g} CI[{lo:+.2f},{hi:+.2f}] shared={100*r**2:.1f}%  n={len(p)}")
print("saved prior_analysis/chunk_vs_planning.{png,svg}")
