"""
Figure: joint prediction and mediation — Section 4.

Three panels:

    A. 2-D scatter: mean chunk size × median RT between edits.
       Points coloured by accuracy (RdYlGn), top solvers ringed in blue.
       Median-split quadrant lines with quadrant labels.
       Spearman ρ(size, RT) annotated.

    B. Quadrant accuracy profile.
       Grouped bars: mean accuracy (left axis) and % top solvers (right axis)
       per quadrant, sorted by mean accuracy.

    C. Variance decomposition (rank regression).
       Stacked horizontal bar showing unique_size, shared, unique_rt as
       fractions of total joint R², with unexplained remainder.
       Companion table with β / p for each model row.

Reads from prior_analysis/:
    chunking_top_solvers_merged.csv
    section4_regression.csv
    section4_mediation.csv
    section4_quadrant_profile.csv
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


# ── palette / style ────────────────────────────────────────────────────────────

COLOR_TOP  = "#2166ac"
COLOR_REST = "#d6604d"

QUADRANT_COLORS = {
    "global_fast": "#1a9641",
    "local_fast":  "#a6d96a",
    "local_slow":  "#fdae61",
    "global_slow": "#d7191c",
}
QUADRANT_LABELS = {
    "global_fast": "global\n+ fast",
    "local_fast":  "local\n+ fast",
    "local_slow":  "local\n+ slow",
    "global_slow": "global\n+ slow",
}

VAR_COLORS = {
    "unique_size": "#4393c3",
    "shared":      "#b2abd2",
    "unique_rt":   "#d6604d",
    "unexplained": "#e8e8e8",
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/section4_joint_figure.png")
    args = ap.parse_args()

    merged = pd.read_csv(
        os.path.join(args.prior_dir, "chunking_top_solvers_merged.csv"))
    reg    = pd.read_csv(
        os.path.join(args.prior_dir, "section4_regression.csv")).set_index("model")
    med    = pd.read_csv(
        os.path.join(args.prior_dir, "section4_mediation.csv")).iloc[0]
    quad   = pd.read_csv(
        os.path.join(args.prior_dir, "section4_quadrant_profile.csv"))
    quad   = quad.set_index("quadrant")

    sub = merged[["size", "mean_rt_median", "accuracy", "top_solver"]].dropna()
    top_mask = sub["top_solver"].astype(bool)

    fig = plt.figure(figsize=(18, 7))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 0.9],
                   wspace=0.40, left=0.06, right=0.97, top=0.87, bottom=0.14)

    # ── Panel A: 2-D scatter with quadrant lines ──────────────────────────────
    axA = fig.add_subplot(gs[0, 0])

    sz_med = sub["size"].median()
    rt_med = sub["mean_rt_median"].median()

    sc = axA.scatter(
        sub["size"], sub["mean_rt_median"],
        c=sub["accuracy"], cmap="RdYlGn", vmin=0.5, vmax=1.0,
        s=45, alpha=0.85, edgecolors="grey", linewidths=0.3, zorder=3,
    )
    # Ring top solvers
    top_sub = sub[top_mask]
    axA.scatter(
        top_sub["size"], top_sub["mean_rt_median"],
        s=100, facecolors="none", edgecolors=COLOR_TOP,
        linewidths=1.6, zorder=4, label=f"top solver (n={top_mask.sum()})",
    )

    # Quadrant dividers
    axA.axvline(sz_med, color="black", lw=0.8, ls="--", alpha=0.45, zorder=2)
    axA.axhline(rt_med, color="black", lw=0.8, ls="--", alpha=0.45, zorder=2)

    # Quadrant labels at corners
    x_lo, x_hi = axA.get_xlim()
    y_lo, y_hi = axA.get_ylim()
    x_lo, x_hi = sub["size"].min(), sub["size"].max()
    y_lo, y_hi = sub["mean_rt_median"].min(), sub["mean_rt_median"].max()
    x_pad = 0.02 * (x_hi - x_lo)
    y_pad = 0.02 * (y_hi - y_lo)

    corner_kwargs = dict(fontsize=8, alpha=0.75, style="italic")
    axA.text(sz_med + x_pad, y_lo + y_pad, "global + fast",
             va="bottom", ha="left", color=QUADRANT_COLORS["global_fast"], **corner_kwargs)
    axA.text(sz_med - x_pad, y_lo + y_pad, "local + fast",
             va="bottom", ha="right", color=QUADRANT_COLORS["local_fast"], **corner_kwargs)
    axA.text(sz_med - x_pad, y_hi - y_pad, "local + slow",
             va="top",    ha="right", color=QUADRANT_COLORS["local_slow"], **corner_kwargs)
    axA.text(sz_med + x_pad, y_hi - y_pad, "global + slow",
             va="top",    ha="left",  color=QUADRANT_COLORS["global_slow"], **corner_kwargs)

    ok  = sub[["size", "mean_rt_median"]].notna().all(axis=1)
    rho, p = spearmanr(sub.loc[ok, "size"], sub.loc[ok, "mean_rt_median"])
    p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
    axA.set_xlabel("mean chunk size (edits per chunk)", fontsize=10)
    axA.set_ylabel("median RT between edits (ms)", fontsize=10)
    axA.set_title(
        f"A. Chunk size vs edit pace\n(Spearman ρ={rho:.2f}, {p_str})",
        loc="left", fontsize=12,
    )
    axA.legend(fontsize=9, loc="upper right")
    axA.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=axA, shrink=0.7, pad=0.04)
    cbar.ax.set_ylabel("accuracy", fontsize=9)

    # ── Panel B: quadrant accuracy / top-solver profile ───────────────────────
    axB  = fig.add_subplot(gs[0, 1])
    axB2 = axB.twinx()

    # Sort by mean_accuracy descending (already sorted in CSV)
    q_order = ["global_fast", "local_fast", "local_slow", "global_slow"]
    q_labels = [QUADRANT_LABELS[q] for q in q_order]
    q_acc    = [quad.loc[q, "mean_accuracy"] for q in q_order]
    q_pct    = [quad.loc[q, "pct_top"]       for q in q_order]
    q_cols   = [QUADRANT_COLORS[q]            for q in q_order]
    x = np.arange(len(q_order))
    w = 0.38

    bars1 = axB.bar(x - w/2, q_acc, w, color=q_cols, alpha=0.75,
                    label="mean accuracy")
    bars2 = axB2.bar(x + w/2, [v * 100 for v in q_pct], w,
                     color=q_cols, alpha=0.40, hatch="//",
                     label="% top solvers")

    for bar, v in zip(bars1, q_acc):
        axB.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, q_pct):
        axB2.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.5,
                  f"{v:.0%}", ha="center", va="bottom", fontsize=8)

    axB.set_xticks(x)
    axB.set_xticklabels(q_labels, fontsize=9)
    axB.set_ylim(0.60, 1.01)
    axB.set_ylabel("mean accuracy", fontsize=10)
    axB2.set_ylim(0, 55)
    axB2.set_ylabel("% top solvers", fontsize=10)
    axB.set_title("B. Quadrant profile\n(median splits on size and RT)",
                  loc="left", fontsize=12)
    axB.grid(axis="y", alpha=0.2)

    # Combined legend
    h1 = mpatches.Patch(facecolor="grey", alpha=0.75, label="mean accuracy")
    h2 = mpatches.Patch(facecolor="grey", alpha=0.40, hatch="//",
                        label="% top solvers")
    axB.legend(handles=[h1, h2], fontsize=8.5, loc="upper right")

    # ── Panel C: variance decomposition ──────────────────────────────────────
    axC = fig.add_subplot(gs[0, 2])

    r2_joint   = float(reg.loc["joint",       "R2"])
    r2_unique_s = float(reg.loc["unique_size", "R2"])
    r2_unique_r = float(reg.loc["unique_rt",   "R2"])
    r2_shared  = float(reg.loc["shared",      "R2"])
    r2_unexpl  = 1.0 - r2_joint

    # Stacked horizontal bar (full 100%)
    bar_y = 0.62
    bar_h = 0.18
    segs = [
        ("unique\nsize",     r2_unique_s, VAR_COLORS["unique_size"]),
        ("shared",           r2_shared,   VAR_COLORS["shared"]),
        ("unique\nRT",       r2_unique_r, VAR_COLORS["unique_rt"]),
        ("unexplained",      r2_unexpl,   VAR_COLORS["unexplained"]),
    ]
    left = 0.0
    for label, width, color in segs:
        axC.barh(bar_y, width, bar_h, left=left, color=color,
                 edgecolor="white", linewidth=0.8)
        if width > 0.015:
            mid = left + width / 2
            txt_col = "black" if color == VAR_COLORS["unexplained"] else "white"
            axC.text(mid, bar_y, f"{width:.1%}", ha="center",
                     va="center", fontsize=8, color=txt_col, fontweight="bold")
        left += width

    legend_patches = [
        mpatches.Patch(facecolor=VAR_COLORS["unique_size"], label=f"unique size  R²={r2_unique_s:.3f}"),
        mpatches.Patch(facecolor=VAR_COLORS["shared"],      label=f"shared       R²={r2_shared:.3f}"),
        mpatches.Patch(facecolor=VAR_COLORS["unique_rt"],   label=f"unique RT    R²={r2_unique_r:.3f}"),
        mpatches.Patch(facecolor=VAR_COLORS["unexplained"], edgecolor="grey", label=f"unexplained  R²={r2_unexpl:.3f}"),
    ]
    axC.legend(handles=legend_patches, fontsize=8.5, loc="lower center",
               bbox_to_anchor=(0.5, 0.42), frameon=True)

    # Companion table: model rows
    r2_size = float(reg.loc["size_only", "R2"])
    r2_rt   = float(reg.loc["rt_only",   "R2"])
    beta_s_only = float(reg.loc["size_only", "beta_size"])
    p_s_only    = float(reg.loc["size_only", "p_size"])
    beta_r_only = float(reg.loc["rt_only",   "beta_rt"])
    p_r_only    = float(reg.loc["rt_only",   "p_rt"])
    beta_s_joint = float(reg.loc["joint",    "beta_size"])
    p_s_joint    = float(reg.loc["joint",    "p_size"])
    beta_r_joint = float(reg.loc["joint",    "beta_rt"])
    p_r_joint    = float(reg.loc["joint",    "p_rt"])

    def _fmt_p(p: float) -> str:
        return "<.001" if p < 0.001 else f"{p:.3f}"

    def _fmt_beta(b: float) -> str:
        return f"{b:+.3f}"

    tbl_rows = [
        ["Model",                "R²",               "β size",             "β RT"],
        ["size only",            f"{r2_size:.3f}",   f"{_fmt_beta(beta_s_only)} (p={_fmt_p(p_s_only)})",  "—"],
        ["RT only",              f"{r2_rt:.3f}",     "—",                   f"{_fmt_beta(beta_r_only)} (p={_fmt_p(p_r_only)})"],
        ["joint",                f"{r2_joint:.3f}",  f"{_fmt_beta(beta_s_joint)} (p={_fmt_p(p_s_joint)})", f"{_fmt_beta(beta_r_joint)} (p={_fmt_p(p_r_joint)})"],
    ]
    col_x = [0.01, 0.20, 0.52, 0.76]
    row_y = [0.28, 0.19, 0.10, 0.01]
    header_color = "#555555"
    for ci, (hdr, x_pos) in enumerate(zip(tbl_rows[0], col_x)):
        axC.text(x_pos, row_y[0], hdr, transform=axC.transAxes,
                 ha="left", va="bottom", fontsize=8,
                 fontweight="bold", color=header_color)
    for ri, row in enumerate(tbl_rows[1:], start=1):
        for ci, (cell, x_pos) in enumerate(zip(row, col_x)):
            axC.text(x_pos, row_y[ri], cell, transform=axC.transAxes,
                     ha="left", va="bottom", fontsize=7.5, color="#333333")

    # Mediation summary line
    ind  = float(med["indirect"])
    c_lo = float(med["ci_lo"])
    c_hi = float(med["ci_hi"])
    prop = float(med["prop_mediated"])
    sig  = bool(med["significant"])
    med_str = (
        f"Mediation RT→size→acc: indirect={ind:+.3f}  "
        f"95%BC-CI [{c_lo:+.3f}, {c_hi:+.3f}]  "
        f"prop={prop:.0%}  sig={'yes' if sig else 'no'}"
    )
    axC.text(0.01, -0.04, med_str, transform=axC.transAxes,
             ha="left", va="top", fontsize=7.5, color="#555555",
             style="italic")

    axC.set_xlim(0, 1)
    axC.set_ylim(-0.08, 0.85)
    axC.axis("off")
    axC.set_title("C. Variance decomposition (rank regression)\n"
                  "rank(accuracy) ~ rank(size) + rank(RT)",
                  loc="left", fontsize=12)

    # ── suptitle ──────────────────────────────────────────────────────────────
    n = len(sub)
    n_top = int(top_mask.sum())
    fig.suptitle(
        f"Section 4: joint prediction of accuracy by chunking style and edit pace  "
        f"(n={n} subjects, {n_top} top solvers)",
        fontsize=13, y=0.96,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
