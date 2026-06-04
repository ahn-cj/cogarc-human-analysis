"""
Sample descriptives: how prevalent dragging is, and the spread of the chunking
and pacing measures across participants — justifying them as individual-difference
measures.

Six panels (per-subject distributions):
    A. Drag fraction (share of a participant's edits that are dragged) — dragging
       is near-universal and substantial.
    B. Cells per stroke (chunk granularity).
    C. n strokes per trajectory.
    D. Deliberation time.
    E. Inter-edit RT.
    F. Accuracy.

Reads drag_trajectory_stats.csv, stroke_chunking_profile.csv,
chunking_top_solvers_merged.csv. Prints prevalence stats for the text.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _hist(ax, vals, color, xlabel, title, pct=False):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=24, color=color, edgecolor="white", alpha=0.9)
    med = np.median(vals)
    ax.axvline(med, color="black", lw=1.6, ls="--",
               label=f"median = {med:.0%}" if pct else f"median = {med:.2f}")
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel("participants", fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prior_analysis/sample_descriptives_figure.png")
    args = ap.parse_args()
    P = "prior_analysis"

    stats = pd.read_csv(f"{P}/drag_trajectory_stats.csv")
    g = stats.groupby("subject_id").agg(
        n_click=("n_click", "sum"), n_drag=("n_drag", "sum"),
        n_edits=("n_edits", "sum")).reset_index()
    g["drag_frac"] = g["n_drag"] / g["n_edits"]
    prof = pd.read_csv(f"{P}/stroke_chunking_profile.csv")
    mg = pd.read_csv(f"{P}/chunking_top_solvers_merged.csv")

    # ── prevalence stats for the text ─────────────────────────────────────────
    overall_drag = g["n_drag"].sum() / g["n_edits"].sum()
    print("[drag prevalence]")
    print(f"  participants: {len(g)}")
    print(f"  overall fraction of edits that are drags: {overall_drag:.1%}")
    print(f"  per-subject drag fraction: median {g['drag_frac'].median():.1%}, "
          f"IQR {g['drag_frac'].quantile(.25):.1%}–{g['drag_frac'].quantile(.75):.1%}")
    print(f"  participants who drag at all (>0 drags): "
          f"{(g['n_drag'] > 0).mean():.1%}")
    print(f"  participants who drag for >=10% of edits: {(g['drag_frac'] >= .10).mean():.1%}")
    print(f"  near-pure clickers (<5% drags): {(g['drag_frac'] < .05).sum()} "
          f"of {len(g)} ({(g['drag_frac'] < .05).mean():.1%})")

    fig, ax = plt.subplots(2, 3, figsize=(14, 8),
                           gridspec_kw=dict(hspace=0.38, wspace=0.30,
                                            left=0.06, right=0.97, top=0.90, bottom=0.08))
    _hist(ax[0, 0], g["drag_frac"], "#3182bd",
          "fraction of edits that are dragged", "A. Dragging is prevalent", pct=True)
    _hist(ax[0, 1], prof["cells_per_stroke"], "#2166ac",
          "cells per stroke", "B. Chunk granularity")
    _hist(ax[0, 2], prof["n_strokes"], "#6baed6",
          "n strokes per trajectory", "C. Strokes per trajectory")
    _hist(ax[1, 0], mg["deliberation_time_median"] / 1000, "#74c476",
          "deliberation time (s)", "D. Deliberation time")
    _hist(ax[1, 1], mg["mean_rt_median"], "#fd8d3c",
          "median inter-edit RT (ms)", "E. Inter-edit RT")
    _hist(ax[1, 2], mg["accuracy"], "#969696",
          "accuracy (proportion correct)", "F. Accuracy")

    fig.suptitle("Sample descriptives: dragging is a prevalent, near-universal behaviour, "
                 f"and the measures vary widely across participants  (n = {len(g)})",
                 fontsize=13, x=0.06, ha="left", y=0.965)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
