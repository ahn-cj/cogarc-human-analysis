"""
Committee point #2 (upstream fix): recover the click/drag flag the sequence-file
preprocessing dropped.

In the RAW mturk logs each `edit` action carries an explicit `type` field that is
either 'click' (a deliberate mouse-down that begins a stroke) or 'drag' (a cell
painted while the button is held and the cursor moves). While the button is held
the same (x, y, color) fires many times; the published sequence files collapsed
consecutive duplicates to the first instance (and dropped `type`). That collapse
is kept — we only RE-ATTACH the type of the surviving first instance:

    a stroke = one 'click' followed by zero or more 'drag' cells.

Validated recipe (matches the existing sequence files for 98.9% of trajectories):
    concat a problem's attempts in order  →  drop consecutive duplicate
    (x, y, color), keeping the first row and its type.

This script mirrors every existing Edit-sequences CSV into a sibling
"Edit sequences typed/" tree with one added column `type` (click/drag for edit
rows; empty otherwise), and writes a per-trajectory drag-stats summary.

Outputs:
    data/.../Experiment 2/Edit sequences typed/<task>/subj_<s>_trial_<task>.csv
    prior_analysis/drag_trajectory_stats.csv
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import re
import csv
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd

from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR

RAW_DIR = "/Users/carolineahn/Documents/GitHub/ARC-data/MTurk/mturk subject data"
SEQ_DIR = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR)
TYPED_DIR = os.path.join(DEFAULT_DATA_ROOT, os.path.dirname(_EXP2_EDIT_DIR),
                         "Edit sequences typed")
_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_(.+)\.csv$")


def raw_problem_edits_by_task(raw_path: str) -> dict:
    """Return {task_filename: [(x,y,color,type), ...] deduped} for one subject."""
    try:
        d = json.load(open(raw_path))
    except Exception:
        return {}
    out = {}
    for pr in d.get("session", []):
        task = None
        for at in pr.get("attempts", []):
            for a in at:
                if isinstance(a, dict) and a.get("desc") == "new task":
                    task = a.get("problem")
            if task:
                break
        if not task:
            continue
        raw = []
        for at in pr.get("attempts", []):
            for a in at:
                if isinstance(a, dict) and a.get("desc") == "edit":
                    try:
                        raw.append((int(float(a["x"])), int(float(a["y"])),
                                    int(float(a["color"])), a.get("type")))
                    except (KeyError, ValueError, TypeError):
                        pass
        ded = []
        for e in raw:
            if not ded or e[:3] != ded[-1][:3]:
                ded.append(e)
        out[task] = ded
    return out


def assign_types(seq_rows: list, ded: list) -> tuple:
    """Assign a type to each edit row of seq_rows from the deduped raw list `ded`.
    Returns (types_per_row, n_aligned, n_unknown). Robust to small mismatches:
    walks both in order, matching (x,y,color); on divergence advances the raw
    pointer up to a small window to re-sync, else marks 'unknown'."""
    types = [""] * len(seq_rows)
    edit_idx = [i for i, r in enumerate(seq_rows) if r["action"] == "edit"]
    j = 0
    n_aligned = n_unknown = 0
    for i in edit_idx:
        r = seq_rows[i]
        key = (int(float(r["x"])), int(float(r["y"])), int(float(r["color"])))
        matched = False
        # try current and a small look-ahead window in ded
        for k in range(j, min(j + 4, len(ded))):
            if ded[k][:3] == key:
                types[i] = ded[k][3] or "unknown"
                j = k + 1
                matched = True
                break
        if not matched:
            types[i] = "unknown"
            n_unknown += 1
        else:
            n_aligned += 1
    return types, n_aligned, n_unknown


def main():
    os.makedirs(TYPED_DIR, exist_ok=True)
    seq_files = glob.glob(os.path.join(SEQ_DIR, "*.json", "subj_*_trial_*.csv"))
    print(f"[scan] {len(seq_files)} existing sequence files")

    # cache raw per subject
    raw_cache = {}
    stats_rows = []
    n_files = n_written = n_no_raw = n_no_task = 0
    total_aligned = total_unknown = 0
    exact = 0

    for sf in seq_files:
        n_files += 1
        m = _FNAME_RE.match(os.path.basename(sf))
        if not m:
            continue
        subj = m.group(1)
        task_fn = m.group(2)  # e.g. 00d62c1b.json
        if subj not in raw_cache:
            rp = os.path.join(RAW_DIR, subj + ".json")
            raw_cache[subj] = raw_problem_edits_by_task(rp) if os.path.exists(rp) else None
        by_task = raw_cache[subj]
        if by_task is None:
            n_no_raw += 1
            continue
        ded = by_task.get(task_fn)
        if ded is None:
            n_no_task += 1
            continue

        # read existing sequence file preserving all columns/rows
        with open(sf, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        types, n_al, n_unk = assign_types(rows, ded)
        total_aligned += n_al
        total_unknown += n_unk
        n_edit = sum(1 for r in rows if r["action"] == "edit")
        if n_unk == 0 and n_edit == len(ded):
            exact += 1

        # write augmented copy
        out_task_dir = os.path.join(TYPED_DIR, task_fn + ".json"
                                    if not task_fn.endswith(".json") else task_fn)
        os.makedirs(out_task_dir, exist_ok=True)
        out_path = os.path.join(out_task_dir, os.path.basename(sf))
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames) + ["type"])
            w.writeheader()
            for r, t in zip(rows, types):
                r = dict(r)
                r["type"] = t
                w.writerow(r)
        n_written += 1

        et = Counter(t for i, t in enumerate(types) if rows[i]["action"] == "edit")
        n_click = et.get("click", 0)
        n_drag = et.get("drag", 0)
        stats_rows.append(dict(
            subject_id=subj, task_id=task_fn.replace(".json", ""),
            n_edits=n_edit, n_click=n_click, n_drag=n_drag,
            n_unknown=et.get("unknown", 0),
            n_strokes=n_click,  # one stroke begins per click
            drag_frac=(n_drag / n_edit) if n_edit else np.nan,
        ))

    stats = pd.DataFrame(stats_rows)
    os.makedirs("prior_analysis", exist_ok=True)
    stats.to_csv("prior_analysis/drag_trajectory_stats.csv", index=False)

    print(f"[write] typed files: {n_written}  (no raw subj: {n_no_raw}, "
          f"no task in raw: {n_no_task})")
    print(f"[align] exact-match trajectories: {exact}/{n_written} "
          f"({exact/max(n_written,1):.1%})")
    print(f"[align] edit rows aligned: {total_aligned}; unknown: {total_unknown} "
          f"({total_unknown/max(total_aligned+total_unknown,1):.2%})")
    print(f"[stats] drag fraction of edits: median per-traj "
          f"{stats['drag_frac'].median():.1%}, overall "
          f"{stats['n_drag'].sum()/stats['n_edits'].sum():.1%}")
    print(f"        strokes vs edits: median strokes/edits = "
          f"{(stats['n_strokes']/stats['n_edits']).median():.2f}")
    print("wrote prior_analysis/drag_trajectory_stats.csv")


if __name__ == "__main__":
    main()
