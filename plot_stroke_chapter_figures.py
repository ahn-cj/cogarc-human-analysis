"""
Regenerate the chapter figures around the motor chunking measure (cells per
stroke) and the planning/execution pacing split:

    fig3b_accuracy_correlations.png  — forest of continuous accuracy correlations
                                       for the chapter's measures + scatter panels
    fig5_joint_marginal.png          — joint scatter granularity × drag-free pace,
                                       coloured by accuracy (visualises separability)
    fig6_controls_comparison.png     — raw vs controlled/clean ρ for each measure

Reads stroke_chunking_profile.csv, stroke_chunking_per_st.csv,
drag_sensitivity_profile.csv, chunking_top_solvers_merged.csv,
controlled_accuracy_correlations.csv, trial_scores.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from statsmodels.nonparametric.smoothers_lowess import lowess

PRIOR = "prior_analysis"
TRIAL_SCORES = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
                "Behavioral data/trial_scores.csv")
N_BOOT = 5000


def sp_ci(x, y, n_boot=N_BOOT, seed=42):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b[i], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(rho=float(rho), lo=float(lo), hi=float(hi), p=float(p))


def controlled_rho(per_st, feat, acc):
    """Residualise feat on difficulty + log(n_edits) per (subj,task), aggregate, ρ."""
    ts = pd.read_csv(TRIAL_SCORES)
    ts["task_id"] = ts["trial"].str.replace(".json", "", regex=False)
    ts = ts.rename(columns={"attempt_score": "task_difficulty"})
    d = per_st.merge(ts[["task_id", "task_difficulty"]], on="task_id", how="left")
    d["log_len"] = np.log(d["n_edits"].clip(lower=1))
    d = d.dropna(subset=[feat, "task_difficulty", "log_len"])
    X = np.column_stack([np.ones(len(d)), d["task_difficulty"], d["log_len"]])
    beta, *_ = np.linalg.lstsq(X, d[feat].values, rcond=None)
    d["_r"] = d[feat].values - X @ beta
    prof = d.groupby("subject_id")["_r"].mean().reset_index().rename(columns={"subject_id": "subject"})
    prof["subject"] = prof["subject"].astype(str)
    prof = prof.merge(acc, on="subject", how="inner")
    return sp_ci(prof["_r"].values, prof["accuracy"].values)


def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def main():
    sc = pd.read_csv(f"{PRIOR}/stroke_chunking_profile.csv")[
        ["subject", "cells_per_stroke", "n_strokes"]]
    sc["subject"] = sc["subject"].astype(str)
    ds = pd.read_csv(f"{PRIOR}/drag_sensitivity_profile.csv")[["subject", "rt_click"]]
    ds["subject"] = ds["subject"].astype(str)
    mg = pd.read_csv(f"{PRIOR}/chunking_top_solvers_merged.csv")[
        ["subject", "accuracy", "deliberation_time_median", "mean_rt_median"]]
    mg["subject"] = mg["subject"].astype(str)
    per_st = pd.read_csv(f"{PRIOR}/stroke_chunking_per_st.csv")
    ctrl_csv = pd.read_csv(f"{PRIOR}/controlled_accuracy_correlations.csv").set_index("feature")

    df = sc.merge(ds, on="subject").merge(mg, on="subject")
    acc = df[["subject", "accuracy"]].copy()
    A = df["accuracy"].values

    # raw correlations
    R = {
        "cells_per_stroke": sp_ci(df["cells_per_stroke"], A),
        "n_strokes":        sp_ci(df["n_strokes"], A),
        "exec_pace":        sp_ci(df["mean_rt_median"], A),
        "dragfree_pace":    sp_ci(df["rt_click"], A),
        "deliberation":     sp_ci(df["deliberation_time_median"], A),
    }

    # ── fig3b: forest + scatter ───────────────────────────────────────────────
    order = [("cells_per_stroke", "cells per stroke (chunk granularity)", True),
             ("n_strokes", "n strokes per trajectory", True),
             ("exec_pace", "execution pace (inter-edit RT)", False),
             ("dragfree_pace", "drag-free pace (inter-click RT)", True),
             ("deliberation", "deliberation time (total)", True)]
    fig = plt.figure(figsize=(14, 5.5))
    axF = fig.add_axes([0.06, 0.15, 0.42, 0.72])
    ys = np.arange(len(order))[::-1]
    for y, (k, lab, clean) in zip(ys, order):
        r = R[k]
        col = "#2166ac" if r["p"] < .05 else "#999999"
        axF.errorbar(r["rho"], y, xerr=[[r["rho"] - r["lo"]], [r["hi"] - r["rho"]]],
                     fmt="o", color=col, ecolor=col, capsize=3, ms=7)
        axF.text(r["hi"] + .02, y, sig(r["p"]), va="center", fontsize=9, color=col)
        if not clean:
            axF.text(r["rho"], y + .28, "confounded with chunking (§3.4)", ha="center",
                     fontsize=7, color="#b2182b", style="italic")
    axF.axvline(0, color="black", lw=.7)
    axF.axvspan(-.10, .10, color="#f5f5f5", zorder=0)
    axF.set_yticks(ys)
    axF.set_yticklabels([lab for _, lab, _ in order], fontsize=9)
    axF.set_xlim(-.55, .55)
    axF.set_xlabel("Spearman ρ with accuracy (95% bootstrap CI)", fontsize=10)
    axF.set_title("A  Continuous accuracy correlations\n"
                  "(chunk granularity is the primary motor predictor)",
                  loc="left", fontsize=11)
    axF.spines[["top", "right"]].set_visible(False)

    for i, (k, lab) in enumerate([("cells_per_stroke", "cells per stroke"),
                                  ("n_strokes", "n strokes / trajectory")]):
        ax = fig.add_axes([0.57, 0.58 - i * 0.46, 0.38, 0.34])
        ax.scatter(df[k], A, s=14, alpha=.45, color="#2166ac", edgecolor="none")
        lo = lowess(A, df[k], frac=.7, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color="#b2182b", lw=2)
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel("accuracy", fontsize=9)
        ax.set_title(f"{'B' if i == 0 else 'C'}  {lab} vs accuracy "
                     f"(ρ={R[k]['rho']:+.2f})", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Continuous accuracy correlations — motor chunking measures",
                 fontsize=12, fontweight="bold", x=.5, y=.99)
    fig.savefig(f"{PRIOR}/fig3b_accuracy_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig3b")

    # ── fig5: joint granularity × drag-free pace, coloured by accuracy ────────
    fig = plt.figure(figsize=(8, 8))
    axS = fig.add_axes([0.12, 0.12, 0.62, 0.62])
    axX = fig.add_axes([0.12, 0.76, 0.62, 0.16], sharex=axS)
    axY = fig.add_axes([0.76, 0.12, 0.16, 0.62], sharey=axS)
    sca = axS.scatter(df["cells_per_stroke"], df["rt_click"], c=A, cmap="viridis",
                      s=30, edgecolor="white", linewidth=.3)
    axS.set_xlabel("chunk granularity (cells per stroke)", fontsize=11)
    axS.set_ylabel("drag-free pace (inter-click RT, ms)", fontsize=11)
    axX.hist(df["cells_per_stroke"], bins=25, color="#bdbdbd"); axX.axis("off")
    axY.hist(df["rt_click"], bins=25, orientation="horizontal", color="#bdbdbd"); axY.axis("off")
    cb = fig.colorbar(sca, ax=axY, fraction=.5, pad=.2)
    cb.set_label("accuracy", fontsize=9)
    axS.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Granularity × pace are independent (ρ={spearmanr(df['cells_per_stroke'], df['rt_click']).statistic:+.2f}); "
                 f"accuracy rises along\ngranularity (ρ=+.24), not pace (ρ=+.02)",
                 fontsize=11, x=0.07, ha="left", y=1.02)
    fig.savefig(f"{PRIOR}/fig5_joint_marginal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig5")

    # ── fig6: raw vs clean/controlled ρ ───────────────────────────────────────
    cps_c = controlled_rho(per_st, "cells_per_stroke", acc)
    nst_c = controlled_rho(per_st, "n_strokes", acc)
    delib_ctrl = float(ctrl_csv.loc["deliberation_time", "rho_ctrl"])
    rt_ctrl = float(ctrl_csv.loc["mean_rt_between_edits", "rho_ctrl"])

    items = [
        ("cells per stroke", R["cells_per_stroke"]["rho"], cps_c["rho"],
         "raw", "+ difficulty/length controls", True, True),
        ("n strokes / traj", R["n_strokes"]["rho"], nst_c["rho"],
         "raw", "+ difficulty/length controls", True, True),
        ("execution pace\n(inter-edit RT)", R["exec_pace"]["rho"], R["dragfree_pace"]["rho"],
         "raw (drag-contaminated)", "drag-free (inter-click)", True, False),
        ("planning pace\n(deliberation)", R["deliberation"]["rho"], delib_ctrl,
         "raw (total)", "post-example (controlled)", False, True),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(items)); w = 0.36
    raw = [it[1] for it in items]; cln = [it[2] for it in items]
    ax.bar(x - w/2, raw, w, color="#cccccc", edgecolor="black", lw=.5, label="raw / contaminated")
    ax.bar(x + w/2, cln, w, color=["#2166ac" if it[6] else "#b2182b" for it in items],
           edgecolor="black", lw=.5, label="cleaned (survives = blue, collapses = red)")
    for xi, it in zip(x, items):
        ax.text(xi - w/2, it[1] + (.01 if it[1] >= 0 else -.03), f"{it[1]:+.2f}", ha="center", fontsize=8)
        ax.text(xi + w/2, it[2] + (.01 if it[2] >= 0 else -.03), f"{it[2]:+.2f}", ha="center", fontsize=8,
                fontweight="bold")
    ax.axhline(0, color="black", lw=.7)
    ax.axhspan(-.10, .10, color="#f5f5f5", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([it[0] for it in items], fontsize=9)
    ax.set_ylabel("Spearman ρ with accuracy", fontsize=11)
    ax.set_title("What survives: chunk granularity (difficulty/length controls) and\n"
                 "planning pace (post-example) survive; execution pace collapses once drags are removed",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{PRIOR}/fig6_controls_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig6")
    print(f"  (cells/stroke {R['cells_per_stroke']['rho']:+.3f}->{cps_c['rho']:+.3f}; "
          f"n_strokes {R['n_strokes']['rho']:+.3f}->{nst_c['rho']:+.3f}; "
          f"exec_pace {R['exec_pace']['rho']:+.3f}->dragfree {R['dragfree_pace']['rho']:+.3f}; "
          f"delib {R['deliberation']['rho']:+.3f}->ctrl {delib_ctrl:+.3f})")


if __name__ == "__main__":
    main()
