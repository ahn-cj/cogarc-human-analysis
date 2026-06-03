"""
Exploratory (committee point #2): drags painted across multiple cells are logged
as separate `edit` events. There is NO explicit drag flag in the raw logs
(columns: action,x,y,color,time,outcome,rt), so a drag must be *inferred*.

A drag-continuation event should look like: very short inter-edit RT (events fire
at mouse-move rate, not click rate), spatial adjacency to the previous cell (the
cursor enters a neighbouring cell), and the same color (you can't recolor mid-drag).

This script characterises the signature so we can choose a principled rule:
    (A) overall edit-to-edit RT distribution
    (B) RT distribution conditional on the spatial move (same cell / 4-adj /
        8-adj diagonal / jump) and on same-vs-different color
    (C) candidate-drag fraction across a grid of RT thresholds (adjacent+fast)
    (D) preview impact of collapsing maximal drag runs: edit count, chunk size,
        median inter-edit RT before vs after

Diagnostics only — does not modify the pipeline.
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import re
import numpy as np
import pandas as pd

from human_targets import available_task_ids
from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)

_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")

ADJ_THRESHOLDS = [40, 60, 80, 100, 120, 150, 200]


def load_edit_pairs():
    """Return a DataFrame of consecutive edit transitions across all Exp-2
    trajectories: rt, chebyshev distance, same_color, plus trajectory id."""
    rows = []
    traj_meta = []  # (traj_id, n_edits, edits list) for the collapse preview
    ids = available_task_ids()
    for n, tid in enumerate(ids, 1):
        d = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{tid}.json")
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = _FNAME_RE.match(fname)
            if not m:
                continue
            tr = _parse_trajectory(os.path.join(d, fname))
            edits = tr["edits"]
            if len(edits) < 2:
                if edits:
                    traj_meta.append((tid + "/" + m.group(1), edits))
                continue
            traj_meta.append((tid + "/" + m.group(1), edits))
            for i in range(1, len(edits)):
                a, b = edits[i - 1], edits[i]
                rt = b.get("rt", np.nan)
                if np.isnan(rt):
                    continue
                cheb = max(abs(b["x"] - a["x"]), abs(b["y"] - a["y"]))
                rows.append((rt, int(cheb), int(b["color"] == a["color"])))
        if n % 15 == 0:
            print(f"  loaded {n}/{len(ids)} tasks")
    df = pd.DataFrame(rows, columns=["rt", "cheb", "same_color"])
    return df, traj_meta


def move_label(cheb):
    if cheb == 0:
        return "same cell"
    if cheb == 1:
        return "adjacent (king-move)"
    return "jump (>1 cell)"


def collapse_drags(edits, rt_thresh, require_same_color=True):
    """Collapse maximal runs of drag-continuation edits into one event each.
    A drag-continuation = adjacent (chebyshev==1) to previous + rt<=thresh
    (+ same color if required). Returns (n_events_after, sizes_of_runs)."""
    if not edits:
        return 0, []
    n_events = 1
    for i in range(1, len(edits)):
        a, b = edits[i - 1], edits[i]
        rt = b.get("rt", np.nan)
        cheb = max(abs(b["x"] - a["x"]), abs(b["y"] - a["y"]))
        is_drag = (not np.isnan(rt) and rt <= rt_thresh and cheb == 1
                   and (not require_same_color or b["color"] == a["color"]))
        if not is_drag:
            n_events += 1
    return n_events, None


def main():
    print("[load] reading Experiment-2 edit transitions…")
    df, traj_meta = load_edit_pairs()
    print(f"[load] {len(df)} edit-to-edit transitions across "
          f"{len(traj_meta)} trajectories\n")

    # ── A: overall RT distribution ────────────────────────────────────────────
    print("[A] overall edit-to-edit RT (ms) percentiles")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(df["rt"], pcts)
    print("    " + "  ".join(f"p{p}={v:.0f}" for p, v in zip(pcts, vals)))
    print(f"    fraction rt < 100ms: {(df['rt'] < 100).mean():.1%}; "
          f"< 60ms: {(df['rt'] < 60).mean():.1%}\n")

    # ── B: RT conditional on spatial move + color ─────────────────────────────
    print("[B] median RT and share by spatial move (and same-color share)")
    df["move"] = df["cheb"].map(move_label)
    g = df.groupby("move").agg(
        n=("rt", "size"),
        share=("rt", lambda s: len(s) / len(df)),
        median_rt=("rt", "median"),
        p25_rt=("rt", lambda s: np.percentile(s, 25)),
        frac_lt100=("rt", lambda s: (s < 100).mean()),
        same_color=("same_color", "mean"),
    )
    for mv, r in g.iterrows():
        print(f"    {mv:22s} n={int(r['n']):>6d} ({r['share']:.1%})  "
              f"median={r['median_rt']:.0f}ms  p25={r['p25_rt']:.0f}ms  "
              f"frac<100ms={r['frac_lt100']:.1%}  same_color={r['same_color']:.1%}")

    # adjacent moves split by same/diff color
    adj = df[df["cheb"] == 1]
    for sc, lab in [(1, "adjacent + same-color"), (0, "adjacent + diff-color")]:
        s = adj[adj["same_color"] == sc]["rt"]
        if len(s):
            print(f"    {lab:22s} n={len(s):>6d}  median={s.median():.0f}ms  "
                  f"frac<100ms={(s<100).mean():.1%}")
    print()

    # ── C: candidate-drag fraction across thresholds ──────────────────────────
    print("[C] candidate drag-continuation share (adjacent + same-color + rt<=T)")
    for T in ADJ_THRESHOLDS:
        mask = (df["cheb"] == 1) & (df["same_color"] == 1) & (df["rt"] <= T)
        print(f"    T={T:>4d}ms : {mask.mean():.1%} of all transitions "
              f"({int(mask.sum())} events)")
    print()

    # ── D: collapse impact preview (per trajectory) ───────────────────────────
    print("[D] collapse-drags impact on per-trajectory edit count "
          "(median over trajectories)")
    for T in [60, 100, 150]:
        ratios = []
        for tid, edits in traj_meta:
            if len(edits) < 2:
                continue
            after, _ = collapse_drags(edits, T)
            ratios.append(after / len(edits))
        ratios = np.array(ratios)
        print(f"    T={T:>4d}ms : median events-retained = {np.median(ratios):.1%} "
              f"(i.e. ~{1-np.median(ratios):.0%} of logged edits are drag-continuations); "
              f"mean over traj = {ratios.mean():.1%}")
    print("\n[done] use these to pick the drag rule for the preprocessing redo")


if __name__ == "__main__":
    main()
