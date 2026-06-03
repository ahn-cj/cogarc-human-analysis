"""
Figure for committee point #6: the chunk-size × accuracy effect is not a
baseline-RT × fixed-500 ms-threshold artefact.

Panels:
    A  Per-subject gap antimodes (within-subject bimodality). Histogram of each
       subject's fast/slow crossover gap; production 500 ms floor marked.
    B  Spearman ρ of size / cells-per-chunk / n-chunks with accuracy across a
       grid of FIXED pause thresholds, with bootstrap-CI bands; the production
       adaptive rule overlaid as points.
    C  size × accuracy ρ under the production rules vs per-subject RELATIVE
       thresholds (antimode, p80) that remove the baseline-RT × floor interaction.
    D  Between-subject rank stability: Spearman ρ of the per-subject mean-size
       profile between every pair of thresholds (a lumper stays a lumper).

Reads:
    prior_analysis/chunk_threshold_subject_antimodes.csv
    prior_analysis/chunk_threshold_sweep.csv
    prior_analysis/chunk_threshold_stability.csv
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FEAT_LABEL = {"size": "chunk size", "n_cells": "cells per chunk",
              "n_chunks_total": "chunks per trajectory"}
FEAT_COLOR = {"size": "#2166ac", "n_cells": "#4393c3",
              "n_chunks_total": "#b2182b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out", default="prior_analysis/chunk_threshold_robustness_figure.png")
    args = ap.parse_args()

    anti = pd.read_csv(os.path.join(args.prior_dir,
                                    "chunk_threshold_subject_antimodes.csv"))
    sweep = pd.read_csv(os.path.join(args.prior_dir, "chunk_threshold_sweep.csv"))
    stab = pd.read_csv(os.path.join(args.prior_dir,
                                    "chunk_threshold_stability.csv"), index_col=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.ravel()

    # ── A: per-subject antimodes ──────────────────────────────────────────────
    am = anti["antimode_ms"].dropna()
    axA.hist(am, bins=np.linspace(0, 1200, 31), color="#bdbdbd",
             edgecolor="white")
    med = am.median()
    axA.axvline(med, color="#2166ac", lw=2,
                label=f"median antimode = {med:.0f} ms")
    axA.axvline(500, color="#b2182b", lw=2, ls="--",
                label="production 500 ms floor")
    frac_bi = anti["prefers_bimodal"].mean()
    axA.set_title(f"A  Within-subject gap bimodality\n"
                  f"{frac_bi:.0%} of subjects' gap distributions are bimodal (BIC)",
                  loc="left", fontsize=11)
    axA.set_xlabel("per-subject fast/slow crossover (antimode), ms")
    axA.set_ylabel("subjects")
    axA.legend(fontsize=9, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)

    # ── B: ρ vs fixed threshold, with CI bands + adaptive overlay ─────────────
    fixed = sweep[sweep["rule"].str.startswith("fixed")].copy()
    fixed = fixed.sort_values("threshold_ms")
    for f in ["size", "n_cells", "n_chunks_total"]:
        d = fixed[fixed["feature"] == f]
        c = FEAT_COLOR[f]
        axB.plot(d["threshold_ms"], d["rho"], "-o", color=c, ms=5,
                 label=FEAT_LABEL[f])
        axB.fill_between(d["threshold_ms"], d["ci_lo"], d["ci_hi"],
                         color=c, alpha=0.15)
        adp = sweep[(sweep["rule"] == "adaptive(prod)") & (sweep["feature"] == f)]
        if len(adp):
            axB.scatter([520], adp["rho"], marker="*", s=200, color=c,
                        edgecolor="black", zorder=5)
    axB.axhline(0, color="black", lw=0.7)
    axB.axvline(500, color="#b2182b", lw=1, ls="--", alpha=0.6)
    axB.set_title("B  ρ with accuracy is stable in sign across the threshold grid\n"
                  "(★ = production adaptive rule; effect grows with threshold, "
                  "so 500 ms is conservative)", loc="left", fontsize=11)
    axB.set_xlabel("fixed pause threshold (ms)")
    axB.set_ylabel("Spearman ρ with accuracy")
    axB.legend(fontsize=9, loc="center right")
    axB.spines[["top", "right"]].set_visible(False)

    # ── C: size ρ under production vs per-subject relative thresholds ─────────
    order = [("fixed 500", "fixed 500 ms"),
             ("adaptive(prod)", "adaptive\n(production)"),
             ("per-subj antimode", "per-subj\nantimode"),
             ("per-subj p80", "per-subj\np80")]
    sz = sweep[sweep["feature"] == "size"].set_index("rule")
    xs = np.arange(len(order))
    rhos = [sz.loc[k, "rho"] for k, _ in order]
    los = [sz.loc[k, "rho"] - sz.loc[k, "ci_lo"] for k, _ in order]
    his = [sz.loc[k, "ci_hi"] - sz.loc[k, "rho"] for k, _ in order]
    cols = ["#999999", "#2166ac", "#1a9850", "#1a9850"]
    axC.bar(xs, rhos, yerr=[los, his], color=cols, capsize=4,
            edgecolor="black", linewidth=0.6)
    for x, k in zip(xs, [o[0] for o in order]):
        p = sz.loc[k, "p"]
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        axC.text(x, sz.loc[k, "ci_hi"] + 0.012, star, ha="center", fontsize=10)
    axC.axhline(0, color="black", lw=0.7)
    axC.set_xticks(xs)
    axC.set_xticklabels([lbl for _, lbl in order], fontsize=9)
    axC.set_ylabel("chunk size × accuracy  (Spearman ρ)")
    axC.set_title("C  The lumper effect survives per-subject relative thresholds\n"
                  "(green = no fixed floor; baseline-RT × floor interaction removed "
                  "by construction)", loc="left", fontsize=11)
    axC.spines[["top", "right"]].set_visible(False)

    # ── D: between-subject rank-stability heatmap ─────────────────────────────
    labels = list(stab.columns)
    short = [l.replace("fixed ", "") for l in labels]
    im = axD.imshow(stab.values, vmin=0.7, vmax=1.0, cmap="YlGnBu")
    axD.set_xticks(range(len(labels)))
    axD.set_yticks(range(len(labels)))
    axD.set_xticklabels(short, fontsize=8)
    axD.set_yticklabels(short, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axD.text(j, i, f"{stab.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if stab.values[i, j] < 0.93 else "white")
    axD.set_title("D  A lumper stays a lumper across cut heights\n"
                  "(between-subject rank correlation of per-subject mean size, "
                  "fixed thresholds in ms)", loc="left", fontsize=11)
    cbar = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ of size profile", fontsize=9)

    fig.suptitle("Chunk size is a stable trait, not a 500 ms-threshold artefact "
                 "(committee point #6)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
