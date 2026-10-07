"""Talk figure: timing measures vs accuracy.

A. Decomposition — deliberation splits into example-viewing and edit-view planning;
   only planning time relates to accuracy (drawing pace does not).
B. Robustness — the planning effect holds after task controls
   (task_difficulty + log trajectory length + n_examples).
All values from controls_regression.py; pacing aggregated per subject by median.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

NULL_C, EFF_C, CTRL_C = "#b0b0b0", "#c62828", "#2166ac"
ctrl = pd.read_csv("prior_analysis/controlled_accuracy_correlations.csv", index_col=0)

def val(feat, kind="raw"):
    return float(ctrl.loc[feat, f"rho_{kind}"]), float(ctrl.loc[feat, f"p_{kind}"])

def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

def draw(ax, items, title, ymin=-0.30, ymax=0.22, width=0.62):
    x = np.arange(len(items))
    ax.axhspan(-0.10, 0.10, color="#f2f2f2", zorder=0)
    for xi, (lab, r, p, c) in zip(x, items):
        ax.bar(xi, r, width=width, color=c, edgecolor="black", lw=0.8, zorder=3)
        off = 0.018 if r >= 0 else -0.018
        ax.text(xi, r + off, f"{r:+.2f}", ha="center", va="bottom" if r >= 0 else "top",
                fontsize=15.5, fontweight="bold" if p < .05 else "normal")
        ax.text(xi, r + (0.062 if r >= 0 else -0.066), star(p), ha="center",
                va="bottom" if r >= 0 else "top", fontsize=12.5, color="#555")
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], fontsize=14.5)
    ax.set_ylim(ymin, ymax); ax.tick_params(axis="y", labelsize=13)
    ax.set_title(title, fontsize=16.5, fontweight="bold", pad=13, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig = plt.figure(figsize=(15.2, 6.8), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.22,
                      left=0.065, right=0.985, top=0.85, bottom=0.13)

axA = fig.add_subplot(gs[0])
draw(axA, [
    ("Total\ndeliberation",      *val("deliberation_time"),                    NULL_C),
    ("Studying\nthe examples",   *val("example_view_time_before_first_edit"),  NULL_C),
    ("Planning time\n(edit view)", *val("planning_time"),                      EFF_C),
    ("Drawing pace\n(drag-free)", *val("rt_click"),                            NULL_C),
], "A. Deliberation splits into two parts — only planning predicts accuracy")
axA.set_ylabel("Spearman ρ with accuracy", fontsize=15.5, labelpad=10)
axA.text(0.99, 0.03, "grey band = |ρ| < .10", transform=axA.transAxes,
         ha="right", fontsize=11.5, color="#777", style="italic")

axB = fig.add_subplot(gs[1])
draw(axB, [("Raw",              *val("planning_time", "raw"),  EFF_C),
           ("+ task\ncontrols", *val("planning_time", "ctrl"), CTRL_C)],
     "B. Planning time survives controls", width=0.5)
axB.set_ylabel("")

fig.suptitle("Timing and accuracy: the effect is specific to edit-view planning  (n = 195)",
             fontsize=18, fontweight="bold", y=0.965)
for ext in ("png", "svg"):
    fig.savefig(f"prior_analysis/timing_bars.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print("saved prior_analysis/timing_bars.{png,svg}")
for f in ["deliberation_time", "example_view_time_before_first_edit", "planning_time", "rt_click"]:
    r, p = val(f); print(f"  {f:36s} raw {r:+.3f} (p={p:.3g})")
print(f"  planning_time controlled: {val('planning_time','ctrl')[0]:+.3f} (p={val('planning_time','ctrl')[1]:.3g})")
