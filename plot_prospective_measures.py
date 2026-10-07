"""Single-panel prospective figures: each measure from trials 1-19 vs accuracy on trials 20+.

Drawing pace = drag-free inter-click RT (median over click-type edits), recomputed per trial from
the typed edit sequences. Validated against the chapter: pace vs accuracy = +.085 (chapter +.09 n.s.).
NOTE: median_rt_between_edits in the measures CSV is a DIFFERENT quantity (drag-contaminated) and
correlates -.58 with chunk size, opposite in sign to the chapter's drag-free pace (+.59).
"""
import os, glob, csv
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prospective_chunking import trial_order, boot_ci, MEAS, STROKE, MIN_TRIALS

CUT = 19
MEAS_C, OUT_C, DOT, TREND = "#5b4bbf", "#2e7d32", "#4a7fb5", "#c62828"
TYPED = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/Behavioral data/"
         "Experiment 2/Edit sequences typed")
CACHE = "prior_analysis/rt_click_per_trial.csv"


def rt_click_per_trial():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, dtype={"subject": str, "trial": str})
    rows = []
    for path in glob.glob(os.path.join(TYPED, "*.json", "*.csv")):
        b = os.path.basename(path)
        subj = b.split("subj_")[1].split("_trial_")[0]
        task = b.split("_trial_")[1].replace(".json.csv", "")
        rt = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["action"] == "edit" and r.get("type") == "click":
                    try:
                        v = float(r["rt"])
                        if not np.isnan(v): rt.append(v)
                    except Exception:
                        pass
        if rt: rows.append({"subject": subj, "trial": task, "rt_click": float(np.median(rt))})
    p = pd.DataFrame(rows); p.to_csv(CACHE, index=False); return p


def load():
    m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
    st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
            .rename(columns={"subject_id": "subject", "task_id": "trial"}))
    d = (m.merge(st[["subject", "trial", "cells_per_stroke"]], on=["subject", "trial"])
           .merge(trial_order(), on=["subject", "trial"])
           .merge(rt_click_per_trial(), on=["subject", "trial"], how="left"))
    d["ok"] = (d.final_outcome.astype(str) == "success").astype(float)
    d["planning_time"] = d.deliberation_time - d.example_view_time_before_first_edit
    d["n_tr"] = d.groupby("subject")["trial"].transform("size")
    return d[d.n_tr >= MIN_TRIALS]


PACING = {"planning_time", "rt_click", "deliberation_time"}


def panel(d, feat, xlab, title, fname, logx=False):
    # pacing features aggregate by MEDIAN (matches controls_regression.py AGG_RULE and the
    # chapter's reported values); chunking features by mean.
    agg = "median" if feat in PACING else "mean"
    A, B = [], []
    for _, g in d.groupby("subject"):
        e, l = g[g.order <= CUT], g[g.order > CUT]
        if e[feat].notna().sum() >= 4 and len(l) >= 4:
            A.append(getattr(e[feat], agg)()); B.append(l["ok"].mean())
    A, B = np.array(A), np.array(B); ok = ~np.isnan(A); A, B = A[ok], B[ok]
    r, p = spearmanr(A, B); lo, hi = boot_ci(A, B)

    fig = plt.figure(figsize=(8.4, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.40,
                          left=0.135, right=0.965, top=0.885, bottom=0.125)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    frac = (CUT - 1) / 74.0
    for x0, x1, c, lab in [(0, frac, MEAS_C, "Trials 1–19\nmeasure here"),
                           (frac + .008, 1, OUT_C, "Trials 20–75\naccuracy here")]:
        ax.add_patch(plt.Rectangle((x0, .30), x1 - x0, .46, facecolor=c, alpha=.15,
                                   edgecolor=c, lw=2))
        ax.text((x0 + x1) / 2, .53, lab, ha="center", va="center", fontsize=12,
                fontweight="bold", color=c, linespacing=1.35)
    ax.plot([frac + .004] * 2, [.16, .90], color="#444", lw=1.5, ls="--")

    ax = fig.add_subplot(gs[1])
    ax.scatter(A, B, s=46, color=DOT, alpha=.55, edgecolor="none")
    # Only draw a trend line when the rank correlation is reliable; an OLS line on a
    # null relationship is dragged by outliers and visually contradicts rho.
    if p < .05:
        if logx:
            ax.set_xscale("log")
            xs = np.linspace(np.log10(A.min()), np.log10(A.max()), 100)
            z = np.polyfit(np.log10(A), B, 1); ax.plot(10**xs, np.polyval(z, xs), color=TREND, lw=3.2)
        else:
            xs = np.linspace(A.min(), A.max(), 100)
            z = np.polyfit(A, B, 1); ax.plot(xs, np.polyval(z, xs), color=TREND, lw=3.2)
    else:
        if logx: ax.set_xscale("log")
        ax.axhline(B.mean(), color="#999999", lw=2.4, ls="--")
    ax.set_xlabel(xlab, fontsize=14.5, color=MEAS_C, fontweight="bold", labelpad=9)
    ax.set_ylabel("Accuracy, trials 20–75", fontsize=14.5, color=OUT_C, fontweight="bold", labelpad=9)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=13, color="#222")
    ax.tick_params(labelsize=13)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    sig = "" if p < .05 else "   n.s."
    ax.text(.97, .06, f"ρ = {r:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]{sig}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13.5,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbb", lw=1))
    for ext in ("png", "svg"):
        fig.savefig(f"prior_analysis/{fname}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"{fname:26s} rho={r:+.3f} p={p:.4g} CI[{lo:+.2f},{hi:+.2f}] n={len(A)}")


if __name__ == "__main__":
    d = load()
    panel(d, "cells_per_stroke", "Chunk size, trials 1–19  (cells per stroke)",
          "Coarser early chunking → higher later accuracy", "prospective_chunk_size")
    panel(d, "planning_time", "Planning time, trials 1–19  (ms)",
          "Faster early planning → higher later accuracy", "prospective_planning_time", logx=True)
    panel(d, "rt_click", "Drawing pace, trials 1–19  (inter-click RT, ms)",
          "Early drawing pace does not predict later accuracy", "prospective_drawing_pace", logx=True)
