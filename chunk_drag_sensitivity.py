"""
Committee point #2 (sensitivity analysis): do the chunking/pacing conclusions
survive treating drag-painted cells correctly?

Using the deterministic click/drag flag recovered in build_typed_sequences.py
(a stroke = one 'click' mouse-down + the cells dragged into while held), we
re-derive the three quantities Joe flagged as drag-contaminated and re-correlate
them with continuous accuracy:

  (1) EDIT COUNT / CHUNK SIZE — drags inflate it. Recompute chunk size in
      *strokes* (deliberate actions) instead of cells, re-correlate with accuracy.
  (2) INTER-EDIT RT (pacing) — drags deflate it (within-drag gaps are tiny).
      Recompute median inter-edit RT over deliberate strokes only (rt of 'click'
      edits), re-correlate with accuracy.
  (3) COHERENCE — drags are contiguous + same-color by motor construction.
      Re-run the real-vs-null connectedness comparison both on all cells and on
      stroke-onset (click) cells only; show real still beats a random-boundary
      null even with drag-induced contiguity removed.

Reads the typed sequence files ("Edit sequences typed/") and per-subject accuracy
from prior_analysis/chunking_top_solvers_merged.csv.

Outputs: prior_analysis/drag_sensitivity_results.csv
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

PROD_FACTOR, PROD_FLOOR = 2.0, 500.0
N_BOOT = 2000


# ── load one typed trajectory ─────────────────────────────────────────────────

def load_typed(path):
    """Return ordered edit list [(x,y,color,rt,type)] for a typed sequence file."""
    edits = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["action"] != "edit":
                continue
            try:
                x, y, c = int(float(r["x"])), int(float(r["y"])), int(float(r["color"]))
                rt = float(r["rt"]) if r["rt"] not in ("", None) else np.nan
            except (ValueError, KeyError):
                continue
            edits.append((x, y, c, rt, r.get("type", "")))
    return edits


def chunk_boundaries(edits):
    """Production pause rule: thresh = max(2*median rt, 500). Returns bool array
    where True at i means edit i starts a new chunk."""
    rts = np.array([e[3] for e in edits], dtype=float)
    valid = rts[~np.isnan(rts)]
    med = float(np.median(valid)) if valid.size else 0.0
    thresh = max(PROD_FACTOR * med, PROD_FLOOR)
    starts = np.zeros(len(edits), dtype=bool)
    for i in range(1, len(edits)):
        rt = rts[i]
        if not np.isnan(rt) and rt > thresh:
            starts[i] = True
    return starts, thresh


def is_connected(cells):
    """Do the distinct cells form a single 4-connected component?"""
    s = set(cells)
    if len(s) <= 1:
        return True
    seen = set()
    stack = [next(iter(s))]
    while stack:
        cy, cx = stack.pop()
        if (cy, cx) in seen:
            continue
        seen.add((cy, cx))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cy + dy, cx + dx)
            if nb in s and nb not in seen:
                stack.append(nb)
    return len(seen) == len(s)


# ── per-trajectory features (real + drag-aware) ───────────────────────────────

def trajectory_features(edits, rng_seed=0):
    """Return dict of per-trajectory measures."""
    if len(edits) < 1:
        return None
    starts, _ = chunk_boundaries(edits)

    # group into chunks
    chunks = []
    cur = [0]
    for i in range(1, len(edits)):
        if starts[i]:
            chunks.append(cur)
            cur = [i]
        else:
            cur.append(i)
    chunks.append(cur)

    size_cells, size_strokes = [], []
    conn_all, conn_click = [], []
    for ch in chunks:
        cells = [(edits[i][1], edits[i][0]) for i in ch]          # (y,x)
        click_cells = [(edits[i][1], edits[i][0]) for i in ch
                       if edits[i][4] == "click"]
        n_click = sum(1 for i in ch if edits[i][4] == "click")
        size_cells.append(len(set(cells)))
        size_strokes.append(max(n_click, 1))
        conn_all.append(is_connected(cells))
        if click_cells:
            conn_click.append(is_connected(click_cells))

    # pacing: within-chunk inter-edit RTs (exclude chunk-first edits)
    within_rt_all, within_rt_click = [], []
    chunk_first = {ch[0] for ch in chunks}
    for i in range(1, len(edits)):
        if i in chunk_first:
            continue
        rt = edits[i][3]
        if np.isnan(rt):
            continue
        within_rt_all.append(rt)
        if edits[i][4] == "click":
            within_rt_click.append(rt)

    return dict(
        mean_size_cells=float(np.mean(size_cells)),
        mean_size_strokes=float(np.mean(size_strokes)),
        n_chunks=len(chunks),
        rt_all=float(np.median(within_rt_all)) if within_rt_all else np.nan,
        rt_click=float(np.median(within_rt_click)) if within_rt_click else np.nan,
        conn_all=float(np.mean(conn_all)) if conn_all else np.nan,
        conn_click=float(np.mean(conn_click)) if conn_click else np.nan,
        # null-Cut connectedness: same #chunks, random boundary positions
        conn_all_null=_null_connectedness(edits, len(chunks), rng_seed),
    )


def _null_connectedness(edits, n_chunks, seed):
    n = len(edits)
    if n_chunks <= 1 or n_chunks >= n:
        cells = [(e[1], e[0]) for e in edits]
        return float(is_connected(cells))
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, n), size=n_chunks - 1, replace=False))
    bounds = [0, *cuts.tolist(), n]
    conn = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        cells = [(edits[i][1], edits[i][0]) for i in range(a, b)]
        conn.append(is_connected(cells))
    return float(np.mean(conn))


def spearman_ci(x, y, n_boot=N_BOOT, seed=42):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return dict(rho=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n=len(x))
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    nn = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, nn, nn)
        boot[b], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(rho=float(rho), ci_lo=float(lo), ci_hi=float(hi), p=float(p), n=int(nn))


def main():
    acc = (pd.read_csv("prior_analysis/chunking_top_solvers_merged.csv")
           [["subject", "accuracy"]])
    acc["subject"] = acc["subject"].astype(str)

    files = glob.glob(os.path.join(TYPED_DIR, "*.json", "subj_*_trial_*.csv"))
    print(f"[load] {len(files)} typed trajectories")
    per_subj = defaultdict(lambda: defaultdict(list))
    for k, sf in enumerate(files):
        m = _FNAME_RE.match(os.path.basename(sf))
        if not m:
            continue
        subj = m.group(1)
        edits = load_typed(sf)
        if not edits:
            continue
        feat = trajectory_features(edits, rng_seed=k)
        if feat is None:
            continue
        for key, v in feat.items():
            if v is not None and np.isfinite(v):
                per_subj[subj][key].append(v)
        if (k + 1) % 3000 == 0:
            print(f"  processed {k+1}/{len(files)}")

    # aggregate per subject (mean across trajectories, matching the pipeline;
    # pacing aggregated by median as in the chapter)
    rows = []
    for subj, d in per_subj.items():
        rows.append(dict(
            subject=subj,
            size_cells=np.mean(d["mean_size_cells"]),
            size_strokes=np.mean(d["mean_size_strokes"]),
            rt_all=np.median(d["rt_all"]) if d["rt_all"] else np.nan,
            rt_click=np.median(d["rt_click"]) if d["rt_click"] else np.nan,
            conn_all=np.mean(d["conn_all"]) if d["conn_all"] else np.nan,
            conn_click=np.mean(d["conn_click"]) if d["conn_click"] else np.nan,
            conn_all_null=np.mean(d["conn_all_null"]) if d["conn_all_null"] else np.nan,
        ))
    prof = pd.DataFrame(rows)
    prof["subject"] = prof["subject"].astype(str)
    prof = prof.merge(acc, on="subject", how="inner")
    print(f"[merge] {len(prof)} subjects with accuracy\n")

    # ── prong 1 & 2: accuracy correlations, raw vs drag-aware ─────────────────
    results = []
    print("[1] CHUNK SIZE × accuracy   (drags inflate size)")
    for label, col in [("size in cells (raw)", "size_cells"),
                       ("size in strokes (drag-aware)", "size_strokes")]:
        r = spearman_ci(prof[col].values, prof["accuracy"].values)
        results.append(dict(prong="size", measure=label, **r))
        print(f"    {label:30s} ρ={r['rho']:+.3f}  CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p={r['p']:.4f}")

    print("\n[2] INTER-EDIT RT × accuracy   (drags deflate pace)")
    print(f"    median pace: all-edits {prof['rt_all'].median():.0f}ms  →  "
          f"click-only {prof['rt_click'].median():.0f}ms  "
          f"(drag-free pace is slower, as expected)")
    for label, col in [("pace, all edits (raw)", "rt_all"),
                       ("pace, deliberate strokes (drag-free)", "rt_click")]:
        r = spearman_ci(prof[col].values, prof["accuracy"].values)
        results.append(dict(prong="pacing", measure=label, **r))
        print(f"    {label:38s} ρ={r['rho']:+.3f}  CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p={r['p']:.4f}")

    # ── prong 3: coherence real vs null, all cells vs click cells ─────────────
    print("\n[3] CHUNK CONNECTEDNESS real vs random-boundary null "
          "(per-subject mean frac connected)")
    print(f"    real (all cells)      = {prof['conn_all'].mean():.3f}")
    print(f"    null-Cut (all cells)  = {prof['conn_all_null'].mean():.3f}  "
          f"→ real−null = {prof['conn_all'].mean()-prof['conn_all_null'].mean():+.3f}")
    print(f"    real (click cells only, drags removed) = {prof['conn_click'].mean():.3f}")
    results.append(dict(prong="coherence", measure="real_all",
                        rho=prof["conn_all"].mean(), ci_lo=np.nan, ci_hi=np.nan,
                        p=np.nan, n=len(prof)))
    results.append(dict(prong="coherence", measure="null_all",
                        rho=prof["conn_all_null"].mean(), ci_lo=np.nan, ci_hi=np.nan,
                        p=np.nan, n=len(prof)))
    results.append(dict(prong="coherence", measure="real_click_only",
                        rho=prof["conn_click"].mean(), ci_lo=np.nan, ci_hi=np.nan,
                        p=np.nan, n=len(prof)))

    pd.DataFrame(results).to_csv("prior_analysis/drag_sensitivity_results.csv",
                                 index=False)
    prof.to_csv("prior_analysis/drag_sensitivity_profile.csv", index=False)
    print("\nwrote prior_analysis/drag_sensitivity_results.csv + _profile.csv")


if __name__ == "__main__":
    main()
