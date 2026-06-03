"""
Committee point #2 follow-up: can chunks be defined by the MOTOR decision to drag
an area (the click/drag flag) instead of by RT pauses — making chunking
RT-independent and therefore separable from pacing?

Three questions:
  (1) OVERLAP. How similar is the drag-stroke partition (new unit at each click)
      to the RT-pause partition (current chunks)? Boundary containment, units per
      trajectory, and Adjusted Rand Index between the two segmentations.
  (2) PROPORTIONS. Per-participant click vs drag share. Did everyone drag, or are
      there pure-clickers for whom a drag-defined chunk is undefined?
  (3) SEPARABILITY. If granularity is the motor measure cells-per-stroke (how large
      an area you drag in one go — no timing) and pace is an RT measure, are they
      separable? Correlate cells-per-stroke against a drag-contaminated pace
      (median inter-edit RT) and against drag-FREE pace measures (deliberation time
      before drawing; inter-click RT), and test independent prediction of accuracy.

Reads typed sequences, drag_trajectory_stats.csv, drag_sensitivity_profile.csv,
chunking_top_solvers_merged.csv. Diagnostics only.
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import csv
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR

TYPED_DIR = os.path.join(DEFAULT_DATA_ROOT, os.path.dirname(_EXP2_EDIT_DIR),
                         "Edit sequences typed")
_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_(.+)\.csv$")
PROD_FACTOR, PROD_FLOOR = 2.0, 500.0


def load_typed(path):
    edits = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["action"] != "edit":
                continue
            try:
                rt = float(r["rt"]) if r["rt"] not in ("", None) else np.nan
            except ValueError:
                rt = np.nan
            edits.append((r.get("type", ""), rt))
    return edits


def partition_labels(edits):
    """Return (chunk_ids, stroke_ids) aligned to edits."""
    rts = np.array([e[1] for e in edits], dtype=float)
    valid = rts[~np.isnan(rts)]
    thresh = max(PROD_FACTOR * float(np.median(valid)) if valid.size else 0.0, PROD_FLOOR)
    chunk_ids, stroke_ids = [], []
    c = s = 0
    for i, (typ, rt) in enumerate(edits):
        if i > 0 and not np.isnan(rt) and rt > thresh:
            c += 1
        if i > 0 and typ == "click":
            s += 1
        chunk_ids.append(c)
        stroke_ids.append(s)
    return np.array(chunk_ids), np.array(stroke_ids)


def adjusted_rand(a, b):
    """ARI between two integer labelings."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return np.nan
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    cont = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for x, y in zip(a, b):
        cont[ua[x], ub[y]] += 1
    from math import comb
    sum_c = sum(comb(v, 2) for v in cont.flatten() if v >= 2)
    sa = sum(comb(v, 2) for v in cont.sum(1) if v >= 2)
    sb = sum(comb(v, 2) for v in cont.sum(0) if v >= 2)
    tot = comb(n, 2)
    exp = sa * sb / tot if tot else 0
    maxi = 0.5 * (sa + sb)
    return (sum_c - exp) / (maxi - exp) if (maxi - exp) else np.nan


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    rxr = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ryr = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(rxr, ryr)[0, 1])


def main():
    files = glob.glob(os.path.join(TYPED_DIR, "*.json", "subj_*_trial_*.csv"))
    print(f"[load] {len(files)} typed trajectories\n")

    # ── (1) overlap ───────────────────────────────────────────────────────────
    n_chunks, n_strokes, aris = [], [], []
    chunk_starts_are_clicks = []  # of chunk boundaries, fraction at a click
    strokes_per_chunk = []
    for sf in files:
        edits = load_typed(sf)
        if len(edits) < 3:
            continue
        cids, sids = partition_labels(edits)
        nc, ns = cids.max() + 1, sids.max() + 1
        n_chunks.append(nc)
        n_strokes.append(ns)
        aris.append(adjusted_rand(cids, sids))
        strokes_per_chunk.append(ns / nc)
        # chunk boundaries = indices where chunk id increments; are they clicks?
        bnds = [i for i in range(1, len(edits)) if cids[i] != cids[i - 1]]
        if bnds:
            chunk_starts_are_clicks.append(
                np.mean([edits[i][0] == "click" for i in bnds]))
    print("[1] OVERLAP between drag-stroke and RT-pause partitions")
    print(f"    units/trajectory: RT-chunks median={np.median(n_chunks):.0f}, "
          f"strokes median={np.median(n_strokes):.0f}")
    print(f"    strokes per RT-chunk: mean={np.mean(strokes_per_chunk):.2f}, "
          f"median={np.median(strokes_per_chunk):.2f}  "
          f"(>1 ⇒ a pause-chunk usually bundles several drag-strokes)")
    print(f"    fraction of RT-chunk boundaries that fall at a click (stroke onset): "
          f"{np.mean(chunk_starts_are_clicks):.1%}  "
          f"(high ⇒ chunk boundaries are a subset of stroke boundaries; strokes nest in chunks)")
    print(f"    Adjusted Rand Index (stroke partition vs chunk partition): "
          f"mean={np.nanmean(aris):.3f}, median={np.nanmedian(aris):.3f}  "
          f"(1=identical, 0=chance) — measures how much the two definitions agree\n")

    # ── (2) per-participant proportions ────────────────────────────────────────
    stats = pd.read_csv("prior_analysis/drag_trajectory_stats.csv")
    g = stats.groupby("subject_id").agg(
        n_click=("n_click", "sum"), n_drag=("n_drag", "sum"),
        n_edits=("n_edits", "sum")).reset_index()
    g["drag_frac"] = g["n_drag"] / g["n_edits"]
    print("[2] PER-PARTICIPANT click vs drag proportions (n=%d subjects)" % len(g))
    qs = g["drag_frac"].quantile([0, .1, .25, .5, .75, .9, 1.0])
    print("    drag-fraction percentiles: " +
          "  ".join(f"{int(p*100)}%={v:.2f}" for p, v in qs.items()))
    print(f"    subjects with drag_frac < 5%  (essentially pure clickers): "
          f"{(g['drag_frac'] < 0.05).sum()}")
    print(f"    subjects with drag_frac < 10%: {(g['drag_frac'] < 0.10).sum()};  "
          f"> 50%: {(g['drag_frac'] > 0.50).sum()};  > 80%: {(g['drag_frac'] > 0.80).sum()}")
    print(f"    every subject drew with at least one drag? "
          f"{'YES' if (g['n_drag'] > 0).all() else 'NO'}  "
          f"(min drags by a subject = {int(g['n_drag'].min())})\n")

    # ── (3) separability: motor granularity vs RT-free pace ────────────────────
    g["cells_per_stroke"] = g["n_edits"] / g["n_click"].clip(lower=1)
    g = g.rename(columns={"subject_id": "subject"})
    g["subject"] = g["subject"].astype(str)
    prof = pd.read_csv("prior_analysis/drag_sensitivity_profile.csv")
    prof["subject"] = prof["subject"].astype(str)
    merged = pd.read_csv("prior_analysis/chunking_top_solvers_merged.csv")[
        ["subject", "accuracy", "deliberation_time_median", "mean_rt_median"]]
    merged["subject"] = merged["subject"].astype(str)
    df = g.merge(prof[["subject", "rt_all", "rt_click", "size_cells"]], on="subject") \
          .merge(merged, on="subject")
    print("[3] SEPARABILITY: motor granularity (cells/stroke) vs pace measures")
    print("    correlation of cells-per-stroke (motor, RT-free) with each pace measure:")
    for lab, col, free in [
        ("median inter-edit RT (drag-contaminated)", "mean_rt_median", "no"),
        ("inter-edit RT incl drags (recomputed)", "rt_all", "no"),
        ("inter-CLICK RT (drag-free pace)", "rt_click", "yes"),
        ("deliberation time before drawing (drag-free)", "deliberation_time_median", "yes"),
    ]:
        r = spearmanr(df["cells_per_stroke"], df[col], nan_policy="omit")
        print(f"      vs {lab:42s}: ρ={r.statistic:+.3f} (p={r.pvalue:.3f})  [drag-free: {free}]")
    print("\n    accuracy prediction (Spearman ρ):")
    for lab, col in [("cells/stroke (motor granularity)", "cells_per_stroke"),
                     ("inter-click RT (drag-free pace)", "rt_click"),
                     ("deliberation time (drag-free pace)", "deliberation_time_median")]:
        r = spearmanr(df[col], df["accuracy"], nan_policy="omit")
        print(f"      {lab:38s} ρ={r.statistic:+.3f} (p={r.pvalue:.4f})")
    # does motor granularity survive controlling for a drag-free pace?
    d2 = df.dropna(subset=["cells_per_stroke", "accuracy", "deliberation_time_median", "rt_click"])
    print("\n    partial correlations (do they each survive controlling for the other?):")
    print(f"      cells/stroke × accuracy | deliberation_time : "
          f"{partial_spearman(d2['cells_per_stroke'].values, d2['accuracy'].values, d2['deliberation_time_median'].values):+.3f}")
    print(f"      cells/stroke × accuracy | inter-click RT     : "
          f"{partial_spearman(d2['cells_per_stroke'].values, d2['accuracy'].values, d2['rt_click'].values):+.3f}")
    print(f"      deliberation × accuracy | cells/stroke       : "
          f"{partial_spearman(d2['deliberation_time_median'].values, d2['accuracy'].values, d2['cells_per_stroke'].values):+.3f}")
    print("\n[done]")


if __name__ == "__main__":
    main()
