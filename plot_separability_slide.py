"""Slide figure: the two predictors are related (A) but still separable (B).

A. chunk size x planning time scatter -- reproduces (chapter -.35, recomputed -.37).
B. Variance decomposition. NOTE: panel B uses the CHAPTER'S PUBLISHED values
   (chunk .055 | planning .071 | joint .094; unique .023/.039). A recomputation from the
   current planning-time definition gives chunk .055 | planning .035 | joint .067
   (unique .032/.012) -- see plot_separability_slide_RECOMPUTED for that variant.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prospective_chunking import boot_ci, MEAS, STROKE, MIN_TRIALS
import sys

USE_CHAPTER = "--recomputed" not in sys.argv
CHUNK_C, PLAN_C, SHARE_C, DOT, TREND = "#2166ac", "#c62828", "#b0b0b0", "#4a7fb5", "#c62828"

m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
        .rename(columns={"subject_id": "subject", "task_id": "trial"}))
d = m.merge(st[["subject", "trial", "cells_per_stroke"]], on=["subject", "trial"])
d["planning_time"] = d.deliberation_time - d.example_view_time_before_first_edit
d["n_tr"] = d.groupby("subject")["trial"].transform("size")
d = d[d.n_tr >= MIN_TRIALS]
p = pd.DataFrame({"chunk": d.groupby("subject")["cells_per_stroke"].mean(),
                  "plan":  d.groupby("subject")["planning_time"].median() / 1000}).dropna()
r, _ = spearmanr(p.chunk, p.plan); lo, hi = boot_ci(p.chunk.values, p.plan.values)

if USE_CHAPTER:
    only_c, only_p, joint = .055, .071, .094
    uq_c, shared, uq_p = .023, .032, .039
else:
    only_c, only_p, joint = .0554, .0346, .0669
    uq_c, uq_p = joint - only_p, joint - only_c
    shared = only_c + only_p - joint

fig = plt.figure(figsize=(15.4, 6.6), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.24,
                      left=0.065, right=0.98, top=0.855, bottom=0.145)

# ── A: scatter ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.scatter(p.chunk, p.plan, s=54, color=DOT, alpha=.6, edgecolor="none")
z = np.polyfit(np.log(p.chunk), np.log(p.plan), 1)
xs = np.linspace(p.chunk.min(), p.chunk.max(), 200)
ax.plot(xs, np.exp(np.polyval(z, np.log(xs))), color=TREND, lw=3.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 2, 3, 4, 6, 8, 10]); ax.set_yticks([5, 10, 20, 40])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Chunk size  (cells per stroke, coarser →)", fontsize=14.5, labelpad=8)
ax.set_ylabel("Planning time  (s, slower →)", fontsize=14.5, labelpad=8)
ax.set_title("A. The two measures are related…", fontsize=16.5, fontweight="bold", pad=12, loc="left")
ax.tick_params(labelsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(.03, .05, f"ρ = {r:+.2f}  [{lo:+.2f}, {hi:+.2f}]\nonly ≈ {100*r**2:.0f}% shared variance",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=13.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb"))

# ── B: variance decomposition ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1])
ax.bar(0, only_c, width=.6, color=CHUNK_C, edgecolor="black", lw=.8)
ax.bar(1, only_p, width=.6, color=PLAN_C, edgecolor="black", lw=.8)
ax.bar(2, uq_c, width=.6, color=CHUNK_C, edgecolor="black", lw=.8)
ax.bar(2, shared, width=.6, bottom=uq_c, color=SHARE_C, edgecolor="black", lw=.8)
ax.bar(2, uq_p, width=.6, bottom=uq_c + shared, color=PLAN_C, edgecolor="black", lw=.8)
for xi, v in [(0, only_c), (1, only_p), (2, joint)]:
    ax.text(xi, v + .0035, f"{v:.3f}", ha="center", fontsize=15.5, fontweight="bold")
for yv, lab in [(uq_c / 2, f"unique chunk\n{uq_c:.3f}"),
                (uq_c + shared / 2, f"shared\n{shared:.3f}"),
                (uq_c + shared + uq_p / 2, f"unique planning\n{uq_p:.3f}")]:
    ax.text(2.42, yv, lab, va="center", ha="left", fontsize=12.5, color="#333")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Chunk size\nalone", "Planning time\nalone", "Both\ntogether"], fontsize=14.5)
ax.set_ylabel("R² for accuracy  (rank regression)", fontsize=14.5, labelpad=8)
ax.set_ylim(0, max(joint * 1.55, .13)); ax.set_xlim(-.55, 3.5)
ax.set_title("B. …but each still adds unique variance", fontsize=16.5, fontweight="bold", pad=12, loc="left")
ax.tick_params(axis="y", labelsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.suptitle("Chunk size and planning time are two separable predictors of accuracy  (n = 195)",
             fontsize=18, fontweight="bold", y=0.965)
tag = "" if USE_CHAPTER else "_recomputed"
for ext in ("png", "svg"):
    fig.savefig(f"prior_analysis/separability_slide{tag}.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print(f"scatter rho={r:+.3f} [{lo:+.2f},{hi:+.2f}]")
print(f"panel B ({'chapter' if USE_CHAPTER else 'recomputed'}): {only_c:.3f} / {only_p:.3f} / {joint:.3f}"
      f"  unique {uq_c:.3f}+{shared:.3f}+{uq_p:.3f}")
print(f"saved prior_analysis/separability_slide{tag}.{{png,svg}}")
