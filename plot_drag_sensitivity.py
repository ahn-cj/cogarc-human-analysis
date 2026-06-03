"""
Figure for committee point #2: the chunk-size and pacing accuracy effects are
two faces of one drawing-fluency dimension (continuous dragged strokes vs
piecemeal clicks), recovered from the deterministic click/drag flag.

Panels:
    A  Drags are prevalent: per-subject drag fraction of edits.
    B  Sensitivity flip: ρ with accuracy collapses when size is counted in
       deliberate strokes (not cells) and pace over deliberate clicks (not drags).
    C  Coherence survives: real chunks beat the random-boundary null on
       connectedness even with drag cells removed (click-only).
    D  Unification: a single drawing-fluency factor (PC1, 70% of variance) predicts
       accuracy; partial correlations show pacing IS this factor while chunk size
       keeps a small independent residual.

Reads prior_analysis/drag_sensitivity_results.csv, drag_sensitivity_profile.csv,
drag_trajectory_stats.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    rxr = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ryr = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(rxr, ryr)[0, 1])


def main():
    out = "prior_analysis/drag_sensitivity_figure.png"
    res = pd.read_csv("prior_analysis/drag_sensitivity_results.csv").set_index(["prong", "measure"])
    prof = pd.read_csv("prior_analysis/drag_sensitivity_profile.csv")
    prof["subject"] = prof["subject"].astype(str)
    stats = pd.read_csv("prior_analysis/drag_trajectory_stats.csv")
    stats["cells_per_stroke"] = stats["n_edits"] / stats["n_strokes"].clip(lower=1)
    g = stats.groupby("subject_id").agg(
        drag_frac=("drag_frac", "mean"),
        cells_per_stroke=("cells_per_stroke", "mean"),
    ).reset_index().rename(columns={"subject_id": "subject"})
    g["subject"] = g["subject"].astype(str)
    df = prof.merge(g, on="subject", how="inner")
    acc = df["accuracy"].values

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = ax.ravel()

    # ── A: drag prevalence ────────────────────────────────────────────────────
    axA.hist(df["drag_frac"], bins=np.linspace(0, 1, 26), color="#9ecae1",
             edgecolor="white")
    med = df["drag_frac"].median()
    axA.axvline(med, color="#08519c", lw=2, label=f"median = {med:.0%}")
    axA.set_title("A  Drags are a large share of edits\n"
                  "(per-subject fraction of cells painted by dragging)",
                  loc="left", fontsize=11)
    axA.set_xlabel("drag fraction of edits")
    axA.set_ylabel("subjects")
    axA.legend(fontsize=9)
    axA.spines[["top", "right"]].set_visible(False)

    # ── B: sensitivity flip ───────────────────────────────────────────────────
    items = [
        ("chunk size\n(cells, raw)", ("size", "size in cells (raw)"), "#2166ac"),
        ("chunk size\n(strokes)", ("size", "size in strokes (drag-aware)"), "#a6cee3"),
        ("pace\n(all edits)", ("pacing", "pace, all edits (raw)"), "#b2182b"),
        ("pace\n(clicks)", ("pacing", "pace, deliberate strokes (drag-free)"), "#fcae91"),
    ]
    xs = np.arange(len(items))
    rhos = [res.loc[k, "rho"] for _, k, _ in items]
    los = [res.loc[k, "rho"] - res.loc[k, "ci_lo"] for _, k, _ in items]
    his = [res.loc[k, "ci_hi"] - res.loc[k, "rho"] for _, k, _ in items]
    cols = [c for *_, c in items]
    axB.bar(xs, rhos, yerr=[los, his], color=cols, capsize=4, edgecolor="black", lw=0.6)
    for x, (_, k, _) in zip(xs, items):
        p = res.loc[k, "p"]
        s = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        yy = res.loc[k, "ci_hi"] if res.loc[k, "rho"] >= 0 else res.loc[k, "ci_lo"]
        axB.text(x, yy + (0.02 if res.loc[k, "rho"] >= 0 else -0.05), s, ha="center", fontsize=10)
    axB.axhline(0, color="black", lw=0.7)
    axB.set_xticks(xs)
    axB.set_xticklabels([lbl for lbl, *_ in items], fontsize=9)
    axB.set_ylabel("Spearman ρ with accuracy")
    axB.set_title("B  Both effects collapse when drag inflation is removed\n"
                  "(counting deliberate strokes / clicks instead of dragged cells)",
                  loc="left", fontsize=11)
    axB.spines[["top", "right"]].set_visible(False)

    # ── C: coherence survives ─────────────────────────────────────────────────
    cvals = [res.loc[("coherence", "real_all"), "rho"],
             res.loc[("coherence", "null_all"), "rho"],
             res.loc[("coherence", "real_click_only"), "rho"]]
    clabs = ["real\n(all cells)", "null-Cut\n(all cells)", "real\n(clicks only)"]
    ccols = ["#1a9850", "#bdbdbd", "#66bd63"]
    axC.bar(range(3), cvals, color=ccols, edgecolor="black", lw=0.6)
    for i, v in enumerate(cvals):
        axC.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    axC.set_xticks(range(3))
    axC.set_xticklabels(clabs, fontsize=9)
    axC.set_ylim(0, 0.8)
    axC.set_ylabel("mean fraction of chunks 4-connected")
    axC.set_title("C  Coherence is NOT a drag artefact\n"
                  "(real beats null; click-only connectedness unchanged)",
                  loc="left", fontsize=11)
    axC.spines[["top", "right"]].set_visible(False)

    # ── D: unification (partial correlations + PC1) ───────────────────────────
    r_size = spearmanr(df["size_cells"], acc).statistic
    r_pace = spearmanr(df["rt_all"], acc).statistic
    p_size = partial_spearman(df["size_cells"].values, acc, df["drag_frac"].values)
    p_pace = partial_spearman(df["rt_all"].values, acc, df["drag_frac"].values)
    # PC1
    cols4 = [df["size_cells"].values, -df["rt_all"].values,
             df["drag_frac"].values, df["cells_per_stroke"].values]
    Z = np.column_stack([(rankdata(c) - rankdata(c).mean()) / rankdata(c).std() for c in cols4])
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var1 = (S**2 / np.sum(S**2))[0]
    pc1 = U[:, 0] * S[0]
    if spearmanr(pc1, df["drag_frac"]).statistic < 0:
        pc1 = -pc1
    r_pc1 = spearmanr(pc1, acc).statistic

    xs = np.arange(2)
    w = 0.35
    axD.bar(xs - w/2, [abs(r_size), abs(r_pace)], w, label="raw |ρ|",
            color="#6a51a3", edgecolor="black", lw=0.6)
    axD.bar(xs + w/2, [abs(p_size), abs(p_pace)], w,
            label="partial |ρ| (control drawing fluency)",
            color="#cbc9e2", edgecolor="black", lw=0.6)
    axD.set_xticks(xs)
    axD.set_xticklabels(["chunk size", "pace"], fontsize=10)
    axD.set_ylabel("|Spearman ρ| with accuracy")
    axD.legend(fontsize=8, loc="upper right")
    axD.set_title(f"D  One drawing-fluency factor (PC1 = {var1:.0%} of variance, "
                  f"ρ={r_pc1:+.2f} with accuracy)\n"
                  "controlling for it: pace collapses, chunk size keeps a small residual",
                  loc="left", fontsize=11)
    axD.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Pacing and chunking are two faces of one drawing-fluency dimension "
                 "(committee point #2)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
