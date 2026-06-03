"""
Motor-defined chunking: cells-per-stroke as the primary chunking measure.

A chunk is redefined as a STROKE — one continuous drag (a 'click' mouse-down plus
the cells dragged into while the button is held), recovered from the click/drag
flag (build_typed_sequences.py). This replaces the RT-pause segmentation: chunk
identity now comes from the motor decision to drag an area, NOT from inter-edit
timing, so it is independent of (and separable from) pacing.

Primary measure: cells_per_stroke = how large an area a participant commits to in
one continuous motor act (lumper = long drags; splitter = many small clicks).

Validates it as a chapter measure, matching the rigour applied to the old RT-chunk
size:
    (A) Reliability — split-half Spearman across random task splits.
    (B) Trial-level ICC (one-way ANOVA).
    (C) Continuous accuracy correlation (Spearman ρ, bootstrap 95% CI), alongside
        n_strokes/trajectory; compared with the old RT-chunk size.
    (D) Regression controls — residualise on task_difficulty + log(trajectory
        length); does the effect survive?
    (E) Degeneracy check — stroke connectedness / color homogeneity (trivially high
        by motor construction, so they are NOT informative validity signals; the
        meaningful trait is the granularity itself + its reliability).

Reads typed sequences, accuracy from chunking_top_solvers_merged.csv, difficulty
from trial_scores.csv. Outputs prior_analysis/stroke_chunking_*.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import csv
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR

TYPED_DIR = os.path.join(DEFAULT_DATA_ROOT, os.path.dirname(_EXP2_EDIT_DIR),
                         "Edit sequences typed")
_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_(.+)\.csv$")
TRIAL_SCORES = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
                "Behavioral data/trial_scores.csv")
N_BOOT = 5000


def stroke_features(path):
    """Segment one typed trajectory into strokes; return per-trajectory measures."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["action"] != "edit":
                continue
            try:
                x, y, c = int(float(r["x"])), int(float(r["y"])), int(float(r["color"]))
            except (ValueError, KeyError):
                continue
            rows.append((x, y, c, r.get("type", "")))
    if not rows:
        return None
    # strokes: new stroke at each 'click' (first edit always starts one)
    strokes, cur = [], [rows[0]]
    for i in range(1, len(rows)):
        if rows[i][3] == "click":
            strokes.append(cur)
            cur = [rows[i]]
        else:
            cur.append(rows[i])
    strokes.append(cur)

    sizes = [len({(e[1], e[0]) for e in s}) for s in strokes]   # distinct cells/stroke
    conn = [_connected([(e[1], e[0]) for e in s]) for s in strokes]
    homog = [max(np.bincount([e[2] for e in s]).max() / len(s), 0) for s in strokes]
    return dict(
        n_strokes=len(strokes),
        n_edits=len(rows),
        cells_per_stroke=float(np.mean(sizes)),
        stroke_connected=float(np.mean(conn)),
        stroke_homog=float(np.mean(homog)),
    )


def _connected(cells):
    s = set(cells)
    if len(s) <= 1:
        return True
    seen, stack = set(), [next(iter(s))]
    while stack:
        cy, cx = stack.pop()
        if (cy, cx) in seen:
            continue
        seen.add((cy, cx))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (cy + dy, cx + dx) in s and (cy + dy, cx + dx) not in seen:
                stack.append((cy + dy, cx + dx))
    return len(seen) == len(s)


def spearman_ci(x, y, n_boot=N_BOOT, seed=42):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b[i], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return rho, lo, hi, p, n


def anova_icc(df, gcol, vcol):
    d = df[[gcol, vcol]].dropna()
    groups = d.groupby(gcol)[vcol]
    gm = d[vcol].mean()
    nt, ns = len(d), d[gcol].nunique()
    SSb = float(sum(len(g) * (g.mean() - gm) ** 2 for _, g in groups))
    SSw = float(sum(((g - g.mean()) ** 2).sum() for _, g in groups))
    dfb, dfw = ns - 1, nt - ns
    MSb, MSw = SSb / dfb, SSw / dfw
    k = ns / sum(1.0 / groups.count())
    return (MSb - MSw) / (MSb + (k - 1) * MSw)


def split_half(per_st, feat, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    rhos = []
    by_subj = {s: g for s, g in per_st.groupby("subject_id")}
    for _ in range(n_iter):
        a, b = [], []
        for s, g in by_subj.items():
            if len(g) < 6:
                continue
            idx = rng.permutation(len(g))
            h = len(g) // 2
            va = g.iloc[idx[:h]][feat].mean()
            vb = g.iloc[idx[h:]][feat].mean()
            if np.isfinite(va) and np.isfinite(vb):
                a.append(va); b.append(vb)
        if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
            rhos.append(spearmanr(a, b).statistic)
    return float(np.mean(rhos)), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    files = glob.glob(os.path.join(TYPED_DIR, "*.json", "subj_*_trial_*.csv"))
    print(f"[load] {len(files)} typed trajectories")
    rows = []
    for sf in files:
        m = _FNAME_RE.match(os.path.basename(sf))
        if not m:
            continue
        feat = stroke_features(sf)
        if feat is None:
            continue
        rows.append(dict(subject_id=m.group(1),
                         task_id=m.group(2).replace(".json", ""), **feat))
    per_st = pd.DataFrame(rows)
    per_st.to_csv("prior_analysis/stroke_chunking_per_st.csv", index=False)
    print(f"[per-st] {len(per_st)} (subject,task) rows, "
          f"{per_st.subject_id.nunique()} subjects\n")

    # ── E: degeneracy of stroke connectedness/homogeneity ─────────────────────
    print("[E] Stroke connectedness/homogeneity are trivially high (motor "
          "construction) — NOT validity signals:")
    print(f"    mean stroke 4-connected = {per_st['stroke_connected'].mean():.3f}; "
          f"mean stroke color-homogeneity = {per_st['stroke_homog'].mean():.3f}")
    print("    → the meaningful trait is granularity (cells/stroke) + its "
          "reliability, not boundary coherence.\n")

    # ── A: reliability ────────────────────────────────────────────────────────
    mr, lo, hi = split_half(per_st, "cells_per_stroke")
    mrn, lon, hin = split_half(per_st, "n_strokes")
    print("[A] Split-half reliability (100 random task-splits, Spearman ρ)")
    print(f"    cells_per_stroke : ρ = {mr:.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"    n_strokes        : ρ = {mrn:.2f}  [{lon:.2f}, {hin:.2f}]")
    print("    (compare old RT-chunk size split-half ρ ≈ .51)\n")

    # ── B: ICC ────────────────────────────────────────────────────────────────
    icc = anova_icc(per_st, "subject_id", "cells_per_stroke")
    print(f"[B] Trial-level one-way ICC (cells_per_stroke) = {icc:.3f} "
          f"(low like other chunk geometry; trait recovered by per-subject averaging)\n")

    # ── per-subject profile + accuracy ────────────────────────────────────────
    prof = per_st.groupby("subject_id").agg(
        cells_per_stroke=("cells_per_stroke", "mean"),
        n_strokes=("n_strokes", "mean"),
    ).reset_index().rename(columns={"subject_id": "subject"})
    acc = pd.read_csv("prior_analysis/chunking_top_solvers_merged.csv")[["subject", "accuracy", "size"]]
    acc["subject"] = acc["subject"].astype(str)
    prof["subject"] = prof["subject"].astype(str)
    prof = prof.merge(acc, on="subject", how="inner")
    prof.to_csv("prior_analysis/stroke_chunking_profile.csv", index=False)

    # ── C: continuous accuracy correlations ───────────────────────────────────
    print(f"[C] Continuous accuracy correlations (n={len(prof)})")
    for lab, col in [("cells_per_stroke (PRIMARY granularity)", "cells_per_stroke"),
                     ("n_strokes per trajectory", "n_strokes"),
                     ("old RT-chunk size (for comparison)", "size")]:
        rho, lo, hi, p, n = spearman_ci(prof[col].values, prof["accuracy"].values)
        print(f"    {lab:38s} ρ={rho:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  p={p:.4f}")
    print()

    # ── D: regression controls ────────────────────────────────────────────────
    print("[D] Controls: residualise cells_per_stroke on task_difficulty + "
          "log(trajectory length)")
    ts = pd.read_csv(TRIAL_SCORES)
    ts["task_id"] = ts["trial"].str.replace(".json", "", regex=False)
    ts = ts.rename(columns={"attempt_score": "task_difficulty"})
    d = per_st.merge(ts[["task_id", "task_difficulty"]], on="task_id", how="left")
    d["log_len"] = np.log(d["n_edits"].clip(lower=1))
    d = d.dropna(subset=["cells_per_stroke", "task_difficulty", "log_len"])
    X = np.column_stack([np.ones(len(d)), d["task_difficulty"], d["log_len"]])
    beta, *_ = np.linalg.lstsq(X, d["cells_per_stroke"].values, rcond=None)
    d["resid"] = d["cells_per_stroke"].values - X @ beta
    ctrl = d.groupby("subject_id")["resid"].mean().reset_index().rename(columns={"subject_id": "subject"})
    ctrl["subject"] = ctrl["subject"].astype(str)
    ctrl = ctrl.merge(prof[["subject", "accuracy"]], on="subject", how="inner")
    rho, lo, hi, p, n = spearman_ci(ctrl["resid"].values, ctrl["accuracy"].values)
    rho0, *_ = spearman_ci(prof["cells_per_stroke"].values, prof["accuracy"].values)
    print(f"    cells_per_stroke × accuracy : raw ρ={rho0:+.3f} → controlled ρ={rho:+.3f} "
          f"CI[{lo:+.3f},{hi:+.3f}] p={p:.4f}")
    print("\n[done] wrote stroke_chunking_per_st.csv, stroke_chunking_profile.csv")


if __name__ == "__main__":
    main()
