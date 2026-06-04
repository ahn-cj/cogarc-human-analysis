"""Gallery of splitter candidates on task d9f24cd1 (with target + lumper for
reference) to choose the example for the concept figure (Figure 1)."""

from __future__ import annotations

import _paths  # noqa: F401
import matplotlib.pyplot as plt
from plot_concept_strokes import solution_grid, load_strokes, draw_strokes, draw_grid

TASK = "d9f24cd1"
LUMPER = "j2dvablb"
# (subject, label) spanning few -> many strokes, all IoU >= .86
CANDS = ["aqmjput5", "a6jpzpek", "t2eyvuxq", "c2lwacqc",
         "rhbgnqhz", "74l1qvc1", "q04yitea", "0yfthc4y"]

sol = solution_grid(TASK)
shape = sol.shape

fig, axes = plt.subplots(2, 5, figsize=(18, 8),
                         gridspec_kw=dict(hspace=0.32, wspace=0.12,
                                          left=0.03, right=0.98, top=0.9, bottom=0.03))
ax = axes.ravel()

draw_grid(ax[0], sol, "TARGET solution")
le, ls = load_strokes(TASK, LUMPER)
draw_strokes(ax[1], shape, le, ls, f"LUMPER {LUMPER}\n{max(ls)+1} strokes")

for a, subj in zip(ax[2:], CANDS):
    e, s = load_strokes(TASK, subj)
    cps = len(e) / (max(s) + 1)
    draw_strokes(a, shape, e, s, f"{subj}\n{max(s)+1} strokes ({cps:.1f}/stroke)")

fig.suptitle("Splitter candidates (task d9f24cd1) — each colour = one continuous stroke; "
             "pick one for the concept figure", fontsize=14, y=0.97)
fig.savefig("prior_analysis/splitter_candidates.png", dpi=130, bbox_inches="tight")
print("saved prior_analysis/splitter_candidates.png")
