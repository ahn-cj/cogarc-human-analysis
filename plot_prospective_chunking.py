"""Figure: chunking measured EARLY predicts accuracy on LATER, non-overlapping trials."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"     # editable text in the SVG
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
from prospective_chunking import trial_order, boot_ci, MEAS, STROKE, MIN_TRIALS

CUT = 19
MEAS_C, OUT_C = "#5b4bbf", "#2e7d32"
DOT, TREND = "#4a7fb5", "#c62828"

m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
        .rename(columns={"subject_id": "subject", "task_id": "trial"}))
d = (m.merge(st[["subject", "trial", "cells_per_stroke", "n_strokes"]], on=["subject", "trial"])
       .merge(trial_order(), on=["subject", "trial"]))
d["ok"] = (d.final_outcome.astype(str) == "success").astype(float)
d["n_tr"] = d.groupby("subject")["trial"].transform("size")
d = d[d.n_tr >= MIN_TRIALS]

def windows(feat):
    A, B = [], []
    for _, g in d.groupby("subject"):
        e, l = g[g.order <= CUT], g[g.order > CUT]
        if len(e) >= 4 and len(l) >= 4:
            A.append(e[feat].mean()); B.append(l["ok"].mean())
    return np.array(A), np.array(B)

fig = plt.figure(figsize=(13, 6.4), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[0.30, 1.0], hspace=0.42, wspace=0.24,
                      left=0.075, right=0.975, top=0.90, bottom=0.115)

# ---- timeline strip -------------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
frac = (CUT - 1) / 74.0
for x0, x1, c, lab, sub in [(0.0, frac, MEAS_C, "Trials 1–19", "measure chunking here"),
                            (frac + .006, 1.0, OUT_C, "Trials 20–75", "measure accuracy here")]:
    ax.add_patch(plt.Rectangle((x0, .34), x1 - x0, .40, facecolor=c, alpha=.15,
                               edgecolor=c, lw=2.2, joinstyle="round"))
    ax.text((x0 + x1) / 2, .625, lab, ha="center", va="center", fontsize=15,
            fontweight="bold", color=c)
    ax.text((x0 + x1) / 2, .445, sub, ha="center", va="center", fontsize=12.5, color=c)
ax.plot([frac + .003, frac + .003], [.20, .88], color="#444", lw=1.6, ls="--")
ax.text(frac + .003, .10, "no overlap", ha="center", va="top", fontsize=12,
        color="#444", style="italic")

# ---- scatters -------------------------------------------------------------
for k, (feat, xlab, title) in enumerate([
        ("cells_per_stroke", "Chunk size, trials 1–19  (cells per stroke)",
         "Coarser early chunking → higher later accuracy"),
        ("n_strokes", "Strokes per solution, trials 1–19",
         "More early strokes → lower later accuracy")]):
    A, B = windows(feat)
    r, p = spearmanr(A, B); lo, hi = boot_ci(A, B)
    ax = fig.add_subplot(gs[1, k])
    ax.scatter(A, B, s=42, color=DOT, alpha=.55, edgecolor="none")
    z = np.polyfit(A, B, 1); xs = np.linspace(A.min(), A.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color=TREND, lw=3.2)
    ax.set_xlabel(xlab, fontsize=14, color=MEAS_C, fontweight="bold", labelpad=8)
    ax.set_ylabel("Accuracy, trials 20–75", fontsize=14, color=OUT_C, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12, color="#222")
    ax.tick_params(labelsize=12.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(.97, .06, f"ρ = {r:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13.5,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.45", fc="white",
                                         ec="#bbb", lw=1))
    print(f"{feat}: rho={r:+.3f} p={p:.3g} CI[{lo:+.2f},{hi:+.2f}] n={len(A)}")

fig.suptitle("Chunking in the first 19 problems predicts accuracy on problems that came later  (n = 195)",
             fontsize=16.5, fontweight="bold", y=0.975, color="#222")
for ext in ("png", "svg"):
    fig.savefig(f"prior_analysis/prospective_chunking.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print("saved prior_analysis/prospective_chunking.{png,svg}")
