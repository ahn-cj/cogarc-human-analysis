"""
Committee point #6: is the chunk-size × accuracy relationship an artefact of the
baseline-RT × fixed-500 ms-threshold interaction?

The production segmenter cuts a trajectory at every inter-edit gap exceeding
    thresh = max(pause_factor * median_rt, min_pause_ms)   (factor 2.0, floor 500 ms)
The 2×median term is per-trajectory adaptive, but the 500 ms floor binds for fast
drawers (median gap < 250 ms). Joe's worry: fast subjects (who are also more
accurate) keep more edits under a fixed 500 ms cut, mechanically inflating their
chunk size — so "lumpers are more accurate" could be a thresholding artefact rather
than a genuine chunking-style difference.

Four analyses, all from a single disk pass over the Experiment-2 edit sequences:

  (A) Within-subject gap bimodality. Pool each subject's inter-edit gaps; on
      log10(gap) compare a 1- vs 2-component Gaussian mixture (BIC) and record the
      per-subject antimode (posterior crossover, in ms). If gaps are bimodal within
      person, a pause threshold is meaningful; if the antimode does not track
      accuracy, a single fixed cut is not differentially mis-placed for top solvers.

  (B) Multi-threshold sweep. Re-segment at a grid of FIXED thresholds and at the
      production ADAPTIVE rule; recompute per-subject mean chunk size, mean cells
      per chunk, and chunks per trajectory; Spearman ρ (bootstrap CI) vs accuracy
      at each threshold. Tests whether sign/significance is threshold-invariant.

  (C) Per-subject relative threshold. Threshold = each subject's own antimode (A),
      and separately a per-subject gap percentile — NO fixed floor. This removes the
      baseline-RT × fixed-threshold interaction by construction. If ρ survives, the
      lumper/splitter effect is not a thresholding artefact.

  (D) Between-subject rank stability ("dendrogram" view). Cutting the 1-D edit
      sequence at a gap height is single-linkage agglomeration; sweeping the height
      traverses the dendrogram. Spearman ρ between per-subject mean-size profiles at
      each pair of thresholds shows whether a lumper stays a lumper across cut
      heights even as absolute sizes change.

Reads the Experiment-2 edit sequences (via human_chunking loaders) and per-subject
accuracy from prior_analysis/chunking_top_solvers_merged.csv.

Outputs (prior_analysis/):
    chunk_threshold_sweep.csv          (rule, threshold_ms, feature, rho, ci, p, n)
    chunk_threshold_subject_antimodes.csv
    chunk_threshold_stability.csv      (per-subject size profile rank-corr matrix)
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

from human_targets import available_task_ids
from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)
import re

_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")

# Production segmentation constants (human_chunking.identify_chunks)
PROD_FACTOR = 2.0
PROD_FLOOR = 500.0

# Fixed-threshold grid (ms) spanning well below and above the 500 ms floor.
FIXED_GRID = [250.0, 350.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]

# Features whose lumper/splitter interpretation point #6 targets.
FEATURES = ["size", "n_cells", "n_chunks_total"]

N_BOOT = 2000
MIN_GAPS_FOR_GMM = 40


# ── data loading: one disk pass ───────────────────────────────────────────────

def load_trajectories() -> List[Dict]:
    """Return one record per Exp-2 trajectory:
        {subject, task, gaps: np.ndarray, cells: List[(y,x)]}
    `gaps[i]` is the inter-edit RT preceding edit i (gaps[0] precedes edit 0 and is
    NOT a candidate boundary, matching identify_chunks which starts the scan at i=1).
    """
    out: List[Dict] = []
    ids = available_task_ids()
    for n, tid in enumerate(ids, 1):
        traj_dir = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{tid}.json")
        if not os.path.isdir(traj_dir):
            continue
        for fname in os.listdir(traj_dir):
            m = _FNAME_RE.match(fname)
            if not m:
                continue
            subj = m.group(1)
            tr = _parse_trajectory(os.path.join(traj_dir, fname))
            edits = tr["edits"]
            if not edits:
                continue
            gaps = np.array([e.get("rt", np.nan) for e in edits], dtype=float)
            cells = [(e["y"], e["x"]) for e in edits]
            out.append({"subject": str(subj), "task": tid,
                        "gaps": gaps, "cells": cells})
        if n % 15 == 0:
            print(f"  loaded {n}/{len(ids)} tasks")
    return out


# ── segmentation at an arbitrary threshold ────────────────────────────────────

def _boundaries(gaps: np.ndarray, thresh: float) -> np.ndarray:
    """Indices i (>=1) where edit i starts a new chunk (gap exceeds thresh)."""
    b = np.zeros(len(gaps), dtype=bool)
    if len(gaps) > 1:
        g = gaps[1:]
        b[1:] = np.where(np.isnan(g), False, g > thresh)
    return b


def segment_features(rec: Dict, thresh: float) -> Tuple[float, float, int]:
    """Return (mean_size, mean_n_cells, n_chunks) for one trajectory at `thresh`."""
    gaps, cells = rec["gaps"], rec["cells"]
    starts = _boundaries(gaps, thresh)
    sizes, ncells, cur, cur_cells = [], [], 0, set()
    for i in range(len(cells)):
        if i > 0 and starts[i]:
            sizes.append(cur)
            ncells.append(len(cur_cells))
            cur, cur_cells = 0, set()
        cur += 1
        cur_cells.add(cells[i])
    sizes.append(cur)
    ncells.append(len(cur_cells))
    return float(np.mean(sizes)), float(np.mean(ncells)), len(sizes)


def adaptive_thresh(gaps: np.ndarray,
                    factor: float = PROD_FACTOR,
                    floor: float = PROD_FLOOR) -> float:
    valid = gaps[~np.isnan(gaps)]
    med = float(np.median(valid)) if valid.size else 0.0
    return max(factor * med, floor)


# ── per-subject aggregation matching individual_differences_chunking ──────────
# per-(subject, task) mean of chunk features, then per-subject mean across tasks.

def subject_profile(records: List[Dict], thresh_fn) -> pd.DataFrame:
    """`thresh_fn(rec, subj_ctx) -> float` returns the threshold for a trajectory.
    subj_ctx allows per-subject relative thresholds (carries the subject's pooled
    gap stats)."""
    # First pass: pooled gaps per subject (for relative-threshold rules).
    subj_gaps: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in records:
        g = r["gaps"][1:]
        subj_gaps[r["subject"]].append(g[~np.isnan(g)])
    subj_ctx = {s: np.concatenate(v) if v else np.array([])
                for s, v in subj_gaps.items()}

    rows = []
    for r in records:
        th = thresh_fn(r, subj_ctx)
        if not np.isfinite(th):
            continue
        sz, nc, nch = segment_features(r, th)
        rows.append((r["subject"], r["task"], sz, nc, nch))
    st = pd.DataFrame(rows, columns=["subject", "task",
                                     "size", "n_cells", "n_chunks_total"])
    # mean across tasks per subject
    return st.groupby("subject")[FEATURES].mean().reset_index()


# ── stats helpers ─────────────────────────────────────────────────────────────

def spearman_ci(x: np.ndarray, y: np.ndarray,
                n_boot: int = N_BOOT, seed: int = 42) -> Dict:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return dict(rho=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n=len(x))
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(rho=float(rho), ci_lo=float(lo), ci_hi=float(hi),
                p=float(p), n=int(n))


def gmm_antimode(log_gaps: np.ndarray) -> Tuple[float, bool]:
    """Fit 1- and 2-component GMMs on log10(gap). Return (antimode_ms, prefers_2).
    Antimode = posterior crossover between the two components (ms)."""
    X = log_gaps.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=0).fit(X)
    g2 = GaussianMixture(2, n_init=3, random_state=0).fit(X)
    prefers_2 = g2.bic(X) < g1.bic(X)
    xs = np.linspace(log_gaps.min(), log_gaps.max(), 2001)
    hi = int(np.argmax(g2.means_.ravel()))
    post = g2.predict_proba(xs.reshape(-1, 1))[:, hi]
    cross = [xs[i] for i in range(1, len(xs))
             if (post[i - 1] - 0.5) * (post[i] - 0.5) < 0]
    antimode = 10 ** cross[0] if cross else np.nan
    return float(antimode), bool(prefers_2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    args = ap.parse_args()

    acc = (pd.read_csv(os.path.join(args.prior_dir,
                                    "chunking_top_solvers_merged.csv"))
           [["subject", "accuracy"]])
    acc["subject"] = acc["subject"].astype(str)

    print("[load] reading Experiment-2 edit sequences (one pass)…")
    records = load_trajectories()
    n_subj = len({r["subject"] for r in records})
    print(f"[load] {len(records)} trajectories, {n_subj} subjects\n")

    def merge_acc(prof: pd.DataFrame) -> pd.DataFrame:
        return prof.merge(acc, on="subject", how="inner")

    # ── (A) within-subject gap bimodality + antimodes ─────────────────────────
    print("[A] within-subject gap bimodality (pooled log10 gaps per subject)")
    subj_gaps: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in records:
        g = r["gaps"][1:]
        subj_gaps[r["subject"]].append(g[(~np.isnan(g)) & (g > 0)])
    anti_rows = []
    for s, parts in subj_gaps.items():
        allg = np.concatenate(parts) if parts else np.array([])
        if allg.size < MIN_GAPS_FOR_GMM:
            continue
        antimode, pref2 = gmm_antimode(np.log10(allg))
        anti_rows.append((s, int(allg.size), float(np.median(allg)),
                          antimode, pref2))
    anti = pd.DataFrame(anti_rows, columns=["subject", "n_gaps", "median_gap_ms",
                                            "antimode_ms", "prefers_bimodal"])
    anti["subject"] = anti["subject"].astype(str)
    anti = anti.merge(acc, on="subject", how="left")
    anti.to_csv(os.path.join(args.prior_dir,
                             "chunk_threshold_subject_antimodes.csv"), index=False)
    frac_bi = anti["prefers_bimodal"].mean()
    med_anti = anti["antimode_ms"].median()
    print(f"    subjects analysed: {len(anti)}")
    print(f"    prefer 2-component (bimodal) by BIC: {frac_bi:.1%}")
    print(f"    median per-subject antimode: {med_anti:.0f} ms "
          f"(IQR {anti['antimode_ms'].quantile(.25):.0f}–"
          f"{anti['antimode_ms'].quantile(.75):.0f})")
    a = anti.dropna(subset=["antimode_ms", "accuracy"])
    r_anti = spearmanr(a["antimode_ms"], a["accuracy"])
    print(f"    ρ(antimode, accuracy) = {r_anti.statistic:+.3f} "
          f"(p={r_anti.pvalue:.3f}) — near zero ⇒ fixed cut not differentially "
          f"mis-placed for top solvers")
    r_med = spearmanr(a["median_gap_ms"], a["accuracy"])
    print(f"    ρ(median gap, accuracy)= {r_med.statistic:+.3f} "
          f"(p={r_med.pvalue:.3f})  [the pacing/baseline-RT channel]\n")

    # ── (B) fixed-threshold sweep + (adaptive) production rule ─────────────────
    print("[B] multi-threshold sweep — Spearman ρ of each feature with accuracy")
    sweep_rows = []
    profiles_by_thresh: Dict[str, pd.DataFrame] = {}

    def run_rule(label, thresh_ms, thresh_fn):
        prof = merge_acc(subject_profile(records, thresh_fn))
        profiles_by_thresh[label] = prof
        line = [f"    {label:>14s}"]
        for f in FEATURES:
            res = spearman_ci(prof[f].values, prof["accuracy"].values)
            sweep_rows.append(dict(rule=label, threshold_ms=thresh_ms, feature=f,
                                   **res))
            star = "*" if res["p"] < 0.05 else " "
            line.append(f"{f}={res['rho']:+.3f}{star}")
        print("  ".join(line))

    for t in FIXED_GRID:
        run_rule(f"fixed {int(t)}", t, lambda r, ctx, _t=t: _t)
    run_rule("adaptive(prod)", np.nan,
             lambda r, ctx: adaptive_thresh(r["gaps"]))

    # ── (C) per-subject relative thresholds (no fixed floor) ──────────────────
    print("\n[C] per-subject RELATIVE thresholds (removes baseline-RT × floor interaction)")
    anti_map = dict(zip(anti["subject"], anti["antimode_ms"]))

    def thresh_antimode(r, ctx):
        return anti_map.get(r["subject"], np.nan)

    def thresh_pct(pct):
        def fn(r, ctx):
            g = ctx.get(r["subject"], np.array([]))
            return float(np.percentile(g, pct)) if g.size else np.nan
        return fn

    run_rule("per-subj antimode", np.nan, thresh_antimode)
    run_rule("per-subj p80", np.nan, thresh_pct(80))
    run_rule("per-subj p90", np.nan, thresh_pct(90))

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(args.prior_dir, "chunk_threshold_sweep.csv"),
                 index=False)
    print(f"\n  wrote chunk_threshold_sweep.csv ({len(sweep)} rows)")

    # ── (D) between-subject rank stability across cut heights ─────────────────
    print("\n[D] between-subject rank stability of per-subject mean SIZE")
    labels = [f"fixed {int(t)}" for t in FIXED_GRID]
    base = profiles_by_thresh[labels[0]][["subject", "size"]].rename(
        columns={"size": labels[0]})
    for lb in labels[1:]:
        base = base.merge(profiles_by_thresh[lb][["subject", "size"]].rename(
            columns={"size": lb}), on="subject")
    mat = base[labels].corr(method="spearman")
    mat.to_csv(os.path.join(args.prior_dir, "chunk_threshold_stability.csv"))
    print("    Spearman ρ of per-subject size profile between adjacent thresholds:")
    for i in range(len(labels) - 1):
        print(f"      {labels[i]:>10s} ↔ {labels[i+1]:>10s}: "
              f"ρ = {mat.iloc[i, i+1]:.3f}")
    print(f"      {labels[0]:>10s} ↔ {labels[-1]:>10s} (extremes): "
          f"ρ = {mat.loc[labels[0], labels[-1]]:.3f}")
    print("\n[done]")


if __name__ == "__main__":
    main()
