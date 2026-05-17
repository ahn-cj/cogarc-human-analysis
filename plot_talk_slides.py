"""
Talk slides — Top-performer profile and joint distribution.

Slide A: bar chart, top solvers vs rest on three measures:
    - planning time (drawing interface, before first edit)
    - mean chunk size
    - inter-edit RT
Effect sizes annotated.

Slide B: scatter of chunk size × inter-edit RT with marginal density
plots (top vs rest) on the top and right edges. Single regression line +
Spearman ρ.

Outputs (prior_analysis/):
    talk_slideA_top_vs_rest.png
    talk_slideB_joint_marginal.png
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, mannwhitneyu, gaussian_kde


COLOR_TOP  = "#1f6fb4"   # vivid blue
COLOR_REST = "#9c9c9c"   # neutral grey

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  18,
    "axes.labelsize":  15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    gt = int(np.sum(np.subtract.outer(x, y) > 0))
    lt = int(np.sum(np.subtract.outer(x, y) < 0))
    return (gt - lt) / (len(x) * len(y))


def _tukey_keep_mask(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Return mask of values within Tukey's k×IQR fences."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x >= q1 - k * iqr) & (x <= q3 + k * iqr)


def _fmt_p(p: float) -> str:
    return "p<.001" if p < 0.001 else f"p={p:.3f}"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_per_subject() -> pd.DataFrame:
    """Merge: per-subject chunking profile (size) + per-subject timing measures
    (deliberation, RT, planning-moment time).  Restricts to subjects in the
    chunking_top_solvers_merged.csv (n=195) so top-solver labels are consistent.
    """
    merged = pd.read_csv("prior_analysis/chunking_top_solvers_merged.csv")
    behav  = pd.read_csv(_paths.BEHAVIORAL_CSV)
    behav["planning_moment_ms"] = (
        behav["deliberation_time"] - behav["example_view_time_before_first_edit"]
    )
    # only positive, finite values
    behav = behav[behav["planning_moment_ms"].between(0, 1e6)]

    pm = (behav.groupby("subject")["planning_moment_ms"]
                .median().rename("planning_moment_median").reset_index())

    out = merged.merge(pm, left_on="subject", right_on="subject", how="left")
    return out


# ── Slide A: bar chart top vs rest on 3 measures ──────────────────────────────

def slide_A(df: pd.DataFrame, out_path: str) -> None:
    top  = df["top_solver"].astype(bool)
    rest = ~top

    measures = [
        ("planning_moment_median", "planning time\n(drawing interface,\nbefore first edit)", 1000),  # ms→s
        ("size",                   "mean chunk size\n(edits per chunk)",                       1),
        ("mean_rt_median",         "inter-edit RT\n(ms)",                                       1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.16, wspace=0.45)

    for ax, (col, label, scale) in zip(axes, measures):
        vals = df[col].dropna()
        x_top_raw  = vals[top.loc[vals.index]].values / scale
        x_rest_raw = vals[rest.loc[vals.index]].values / scale

        # Drop Tukey outliers within each group for plotting + stats
        x_top  = x_top_raw[_tukey_keep_mask(x_top_raw)]
        x_rest = x_rest_raw[_tukey_keep_mask(x_rest_raw)]

        med_top  = np.median(x_top)
        med_rest = np.median(x_rest)
        iqr_top  = np.percentile(x_top,  [25, 75])
        iqr_rest = np.percentile(x_rest, [25, 75])

        positions = [0, 1]
        meds      = [med_top, med_rest]
        errs      = [
            [med_top  - iqr_top[0],  iqr_top[1]  - med_top],
            [med_rest - iqr_rest[0], iqr_rest[1] - med_rest],
        ]
        errs = np.array(errs).T  # shape (2, 2): lower, upper × 2 bars

        bars = ax.bar(
            positions, meds, width=0.6,
            color=[COLOR_TOP, COLOR_REST], alpha=0.85,
            yerr=errs, capsize=6, error_kw=dict(ecolor="#444444", lw=1.2),
            zorder=2,
        )

        # Strip jitter overlay
        rng = np.random.default_rng(7)
        for pos, vals_g, c in [(0, x_top, COLOR_TOP), (1, x_rest, COLOR_REST)]:
            jit = rng.uniform(-0.12, 0.12, len(vals_g))
            ax.scatter(pos + jit, vals_g, s=14, color=c, alpha=0.30,
                       edgecolors="none", zorder=3)

        # Median labels on bars
        for bar, m in zip(bars, meds):
            unit = ""
            if "RT" in label:        unit = " ms"
            elif "planning" in label: unit = " s"
            elif "chunk size" in label: unit = ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{m:.1f}{unit}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"top\n(n={top.sum()})", f"rest\n(n={rest.sum()})"],
            fontsize=13,
        )
        ax.set_title(label, fontsize=14, pad=14)
        ax.grid(axis="y", alpha=0.25, zorder=1)

        # Cap y-axis at IQR-based upper bound (jitter dots can go above but be clipped)
        y_max = max(meds[0] + errs[1, 0], meds[1] + errs[1, 1]) * 1.55
        ax.set_ylim(0, y_max)

        # Effect size annotation high in axes coordinates (so figure size stays sane)
        d = _cliffs_delta(x_top, x_rest)
        _, p = mannwhitneyu(x_top, x_rest, alternative="two-sided")
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(0.5, 0.95,
                f"δ = {d:+.2f}{sig}   {_fmt_p(p)}",
                ha="center", va="top", fontsize=13, color="#222222",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.95))

    fig.suptitle("Top performers plan faster and draw in larger chunks",
                 fontsize=20, fontweight="bold", y=0.95)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")
    plt.close(fig)


# ── Slide B: scatter with marginal density plots ──────────────────────────────

def slide_B(df: pd.DataFrame, out_path: str) -> None:
    sub = df[["size", "mean_rt_median", "top_solver"]].dropna()

    # Drop bivariate Tukey outliers (either axis out of fences)
    keep = _tukey_keep_mask(sub["size"].values) & _tukey_keep_mask(sub["mean_rt_median"].values)
    sub  = sub[keep].copy()

    top  = sub["top_solver"].astype(bool)
    rest = ~top

    x_all = sub["size"].values
    y_all = sub["mean_rt_median"].values

    rho, p = spearmanr(x_all, y_all)

    fig = plt.figure(figsize=(11, 9))
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[5, 1], height_ratios=[1, 5],
        wspace=0.05, hspace=0.05,
        left=0.10, right=0.95, top=0.91, bottom=0.10,
    )
    axS = fig.add_subplot(gs[1, 0])
    axT = fig.add_subplot(gs[0, 0], sharex=axS)
    axR = fig.add_subplot(gs[1, 1], sharey=axS)

    # ── Main scatter
    axS.scatter(sub.loc[rest, "size"], sub.loc[rest, "mean_rt_median"],
                s=28, color=COLOR_REST, alpha=0.55, edgecolors="none",
                label=f"rest (n={rest.sum()})", zorder=2)
    axS.scatter(sub.loc[top, "size"], sub.loc[top, "mean_rt_median"],
                s=110, color=COLOR_TOP, alpha=0.90,
                edgecolors="white", linewidths=1.2,
                marker="*", label=f"top solvers (n={top.sum()})", zorder=4)

    # Robust trend line: Theil-Sen slope through medians (won't pick up edge noise)
    from scipy.stats import theilslopes
    slope, intercept, _, _ = theilslopes(y_all, x_all)
    x_line = np.linspace(np.percentile(x_all, 2), np.percentile(x_all, 98), 50)
    y_line = slope * x_line + intercept
    axS.plot(x_line, y_line, color="#222222", lw=2.2, alpha=0.75,
             zorder=3, label="Theil–Sen trend")

    axS.set_xlabel("mean chunk size (edits per chunk)", fontsize=15)
    axS.set_ylabel("inter-edit RT (ms)", fontsize=15)
    axS.grid(alpha=0.2)
    axS.legend(fontsize=12, loc="upper right", frameon=True, framealpha=0.95)

    # Annotate ρ in corner
    p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
    axS.text(0.03, 0.05,
             f"Spearman ρ = {rho:.2f}\n{p_str}",
             transform=axS.transAxes,
             ha="left", va="bottom", fontsize=14,
             bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec="#cccccc", alpha=0.95))

    # ── Top marginal: chunk size density, top vs rest
    grid_x = np.linspace(x_all.min() - 0.5, x_all.max() + 0.5, 400)
    for mask, color, lw in [(rest, COLOR_REST, 1.8), (top, COLOR_TOP, 2.4)]:
        vals = sub.loc[mask, "size"].values
        if len(vals) >= 5:
            kde = gaussian_kde(vals, bw_method=0.35)
            d = kde(grid_x)
            axT.fill_between(grid_x, d, alpha=0.30, color=color)
            axT.plot(grid_x, d, color=color, lw=lw)
    axT.set_yticks([])
    axT.tick_params(axis="x", labelbottom=False)
    axT.spines["left"].set_visible(False)
    axT.set_xlim(axS.get_xlim())

    # ── Right marginal: inter-edit RT density, top vs rest
    grid_y = np.linspace(y_all.min() * 0.9, y_all.max() * 1.05, 400)
    for mask, color, lw in [(rest, COLOR_REST, 1.8), (top, COLOR_TOP, 2.4)]:
        vals = sub.loc[mask, "mean_rt_median"].values
        if len(vals) >= 5:
            kde = gaussian_kde(vals, bw_method=0.35)
            d = kde(grid_y)
            axR.fill_betweenx(grid_y, d, alpha=0.30, color=color)
            axR.plot(d, grid_y, color=color, lw=lw)
    axR.set_xticks([])
    axR.tick_params(axis="y", labelleft=False)
    axR.spines["bottom"].set_visible(False)
    axR.set_ylim(axS.get_ylim())

    fig.suptitle("Faster planning and larger chunks go together",
                 fontsize=20, fontweight="bold", y=0.97)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="prior_analysis")
    args = ap.parse_args()

    df = _load_per_subject()
    print(f"[load] {len(df)} subjects; "
          f"{df['planning_moment_median'].notna().sum()} with planning-moment data")

    slide_A(df, os.path.join(args.out_dir, "talk_slideA_top_vs_rest.png"))
    slide_B(df, os.path.join(args.out_dir, "talk_slideB_joint_marginal.png"))


if __name__ == "__main__":
    main()
