"""
Export every chapter figure as an editable SVG (vector) for Adobe Illustrator.

Monkeypatches matplotlib's Figure.savefig so that whenever a plotting script
writes a .png it ALSO writes the same figure as .svg, then runs each chapter
figure script unchanged (no edits to the scripts), and copies the SVGs into
~/Documents/thesis/ch3/figures with their figN names.

Run from the worktree:  python export_figures_svg.py
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import sys
import runpy
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mfig

# ── patch savefig: emit .svg next to every .png ───────────────────────────────
_orig_savefig = mfig.Figure.savefig


def _savefig_with_svg(self, fname, *args, **kwargs):
    out = _orig_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png"):
        svg = str(fname)[:-4] + ".svg"
        kw = {k: v for k, v in kwargs.items() if k != "dpi"}  # dpi irrelevant for svg
        try:
            _orig_savefig(self, svg, *args, **kw)
            print(f"    + {svg}")
        except Exception as e:  # noqa: BLE001
            print(f"    ! svg failed for {fname}: {e}")
    return out


mfig.Figure.savefig = _savefig_with_svg

# ── run each chapter-figure script unchanged ──────────────────────────────────
SCRIPTS = [
    "plot_stroke_chapter_figures.py",   # fig3b, fig5, fig6
    "plot_drag_sensitivity.py",         # fig10
    "plot_temporal_pacing.py",          # fig2
    "plot_temporal_dynamics.py",        # fig7
    "plot_practice_curves.py",          # fig8
    "plot_chunk_threshold_robustness.py",  # fig9 (bonus; retained-for-reference)
]
for script in SCRIPTS:
    print(f"[run] {script}")
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {script}: {e}")

# ── copy SVGs into the chapter figures folder with figN names ─────────────────
FIG_DIR = "/Users/carolineahn/Documents/thesis/ch3/figures"
MAPPING = {
    "prior_analysis/fig3b_accuracy_correlations.svg": "fig3b_accuracy_correlations.svg",
    "prior_analysis/fig5_joint_marginal.svg":         "fig5_joint_marginal.svg",
    "prior_analysis/fig6_controls_comparison.svg":    "fig6_controls_comparison.svg",
    "prior_analysis/drag_sensitivity_figure.svg":     "fig10_drag_fluency.svg",
    "prior_analysis/temporal_pacing_figure.svg":      "fig2_temporal_pacing.svg",
    "prior_analysis/temporal_dynamics_figure.svg":    "fig7_temporal_dynamics.svg",
    "prior_analysis/practice_curves_figure.svg":      "fig8_practice_curves.svg",
    "prior_analysis/chunk_threshold_robustness_figure.svg": "fig9_threshold_robustness.svg",
}
print("\n[copy] -> ch3/figures/")
for src, dst in MAPPING.items():
    if os.path.exists(src):
        shutil.copy(src, os.path.join(FIG_DIR, dst))
        print(f"    {dst}")
    else:
        print(f"    MISSING {src}")
print("[done]")
