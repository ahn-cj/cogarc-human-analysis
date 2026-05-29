"""
Cross-analysis: chunking style × top-solver performance and timing.

Bridges the chunking individual-differences analysis (cogarc-human-analysis)
with the behavioral timing data (CogARC-dataRepository/Behavioral data).

Three analyses:

    (A) Chunk features × timing correlations
        Spearman ρ between each chunk style dimension and per-subject median
        deliberation time / mean RT between edits.

    (B) Top-solver vs rest on chunk features
        Mann-Whitney U + Cliff's delta for each chunk feature comparing the
        35 top solvers (≥ 95 % accuracy, ≥ 40 tasks) against the remaining
        160 participants.

    (C) ICC of chunk features across tasks
        One-way ANOVA ICC quantifying how much of the variance in each chunk
        feature is attributable to stable between-subject differences vs
        within-subject trial-to-trial noise.  Requires
        prior_analysis/individual_differences_per_subj_task.csv, which is
        produced by individual_differences_chunking.py.

Outputs (all in prior_analysis/):
    chunking_timing_correlations.csv
    chunking_top_solvers_stats.csv
    chunking_top_solvers_merged.csv
    chunking_icc.csv                     (only if per_subj_task CSV exists)
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, f as f_dist


# ── constants ─────────────────────────────────────────────────────────────────

CHUNK_FEATURES = [
    "size",
    "n_chunks_total",
    "n_cells",
    "is_connected",
    "fill_ratio",
    "color_homogeneity",
    "bbox_area",
    "nn_chain_rate",
    "success_iou_best",
]

TIMING_FEATURES = [
    "deliberation_time_median",
    "mean_rt_median",
]

TOP_SOLVER_ACC = 0.95
MIN_TASKS = 40


# ── helpers ───────────────────────────────────────────────────────────────────

def _spearman_with_ci(x: np.ndarray, y: np.ndarray,
                      n_boot: int = 5000, seed: int = 42) -> dict:
    """Spearman rho with a percentile bootstrap 95% CI on rho."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        return dict(n=int(n), rho=float("nan"), p_value=float("nan"),
                    ci_lo=float("nan"), ci_hi=float("nan"))
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i], _ = spearmanr(x[idx], y[idx])
    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(n=int(n), rho=float(rho), p_value=float(p),
                ci_lo=float(ci_lo), ci_hi=float(ci_hi))


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = int(np.sum(np.subtract.outer(x, y) > 0))
    lt = int(np.sum(np.subtract.outer(x, y) < 0))
    return (gt - lt) / (len(x) * len(y))


def _anova_icc(data: pd.DataFrame, subject_col: str, rating_col: str) -> dict:
    """One-way ANOVA ICC with harmonic-mean correction for unbalanced data."""
    data = data[[subject_col, rating_col]].dropna()
    groups = data.groupby(subject_col)[rating_col]
    grand_mean = data[rating_col].mean()
    n_total = len(data)
    n_subjects = data[subject_col].nunique()

    SSb = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups))
    SSw = float(sum(((g - g.mean()) ** 2).sum() for _, g in groups))
    dfb = n_subjects - 1
    dfw = n_total - n_subjects
    if dfb <= 0 or dfw <= 0:
        return dict(ICC=float("nan"), CI_lo=float("nan"), CI_hi=float("nan"),
                    n_subjects=n_subjects, n_obs=n_total, p=float("nan"))

    MSb = SSb / dfb
    MSw = SSw / dfw

    k = n_subjects / groups.count().apply(lambda n: 1.0 / n).sum()
    icc = (MSb - MSw) / (MSb + (k - 1) * MSw)

    F = MSb / MSw
    F_lo = F / f_dist.ppf(0.975, dfb, dfw)
    F_hi = F * f_dist.ppf(0.975, dfw, dfb)
    ci_lo = (F_lo - 1) / (F_lo + k - 1)
    ci_hi = (F_hi - 1) / (F_hi + k - 1)
    p = float(1 - f_dist.cdf(F, dfb, dfw))

    return dict(ICC=float(icc), CI_lo=float(ci_lo), CI_hi=float(ci_hi),
                n_subjects=int(n_subjects), n_obs=int(n_total), p=p)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_chunk_profile(prior_dir: str) -> pd.DataFrame:
    path = os.path.join(prior_dir, "individual_differences_merged.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"subject_id": "subject"})
    return df


def _load_behavioral(csv_path: str) -> pd.DataFrame:
    """
    Aggregate the per-trial behavioral CSV to one row per subject.
    Returns: subject, n_tasks, accuracy, top_solver, deliberation_time_median,
             mean_rt_median.
    """
    dt = pd.read_csv(csv_path)
    dt["correct"] = dt["final_outcome"].str.lower() == "success"

    subj = (
        dt.groupby("subject")
        .agg(
            n_tasks=("trial", "nunique"),
            accuracy=("correct", "mean"),
            deliberation_time_median=("deliberation_time", "median"),
            mean_rt_median=("mean_rt_between_edits", "median"),
        )
        .reset_index()
    )
    subj = subj[subj["n_tasks"] >= MIN_TASKS].copy()
    subj["top_solver"] = subj["accuracy"] >= TOP_SOLVER_ACC
    subj["group"] = np.where(subj["top_solver"], "top", "rest")
    return subj


# ── analysis A: chunk × timing correlations ───────────────────────────────────

def _chunk_timing_correlations(merged: pd.DataFrame,
                               chunk_feats: list[str],
                               timing_feats: list[str]) -> pd.DataFrame:
    rows = []
    for cf in chunk_feats:
        for tf in timing_feats:
            x = merged[cf].values
            y = merged[tf].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                rows.append(dict(chunk_feature=cf, timing_feature=tf,
                                 n=int(ok.sum()), rho=float("nan"),
                                 p_value=float("nan")))
                continue
            rho, p = spearmanr(x[ok], y[ok])
            rows.append(dict(chunk_feature=cf, timing_feature=tf,
                             n=int(ok.sum()), rho=float(rho),
                             p_value=float(p)))
    return pd.DataFrame(rows)


# ── analysis A2: feature × accuracy (continuous) ──────────────────────────────

def _accuracy_correlations(merged: pd.DataFrame,
                           feats: list[str],
                           outcome: str = "accuracy") -> pd.DataFrame:
    """Spearman ρ (with bootstrap CI) between each feature and continuous accuracy.

    This is the continuous counterpart to the top-solver dichotomy: it treats
    performance as the graded variable it is, rather than splitting subjects at
    an arbitrary accuracy threshold.
    """
    rows = []
    y = merged[outcome].values
    for f in feats:
        res = _spearman_with_ci(merged[f].values, y)
        res["feature"] = f
        rows.append(res)
    df = pd.DataFrame(rows)[["feature", "n", "rho", "ci_lo", "ci_hi", "p_value"]]
    return df.sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True)


# ── analysis B: top-solver vs rest ────────────────────────────────────────────

def _top_solver_comparisons(merged: pd.DataFrame,
                             chunk_feats: list[str]) -> pd.DataFrame:
    rows = []
    for cf in chunk_feats:
        top = merged.loc[merged["top_solver"], cf].dropna().values
        rest = merged.loc[~merged["top_solver"], cf].dropna().values
        if len(top) < 5 or len(rest) < 5:
            rows.append(dict(feature=cf, n_top=len(top), n_rest=len(rest),
                             top_median=float("nan"), rest_median=float("nan"),
                             median_diff=float("nan"),
                             cliffs_delta=float("nan"),
                             p_mannwhitney=float("nan")))
            continue
        U, p = mannwhitneyu(top, rest, alternative="two-sided")
        rows.append(dict(
            feature=cf,
            n_top=len(top),
            n_rest=len(rest),
            top_median=float(np.median(top)),
            rest_median=float(np.median(rest)),
            median_diff=float(np.median(top) - np.median(rest)),
            cliffs_delta=float(_cliffs_delta(top, rest)),
            p_mannwhitney=float(p),
        ))
    df = pd.DataFrame(rows)
    df = df.sort_values("cliffs_delta", key=np.abs, ascending=False)
    return df


# ── analysis C: ICC of chunk features ────────────────────────────────────────

def _chunk_icc(per_subj_task: pd.DataFrame,
               chunk_feats: list[str]) -> pd.DataFrame:
    rows = []
    for cf in chunk_feats:
        if cf not in per_subj_task.columns:
            continue
        res = _anova_icc(per_subj_task, "subject_id", cf)
        res["feature"] = cf
        rows.append(res)
    df = pd.DataFrame(rows)[["feature", "ICC", "CI_lo", "CI_hi",
                               "n_subjects", "n_obs", "p"]]
    return df.sort_values("ICC", ascending=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--behavioral_csv",
                    default=str(_paths.BEHAVIORAL_CSV))
    args = ap.parse_args()

    print(f"[load] chunk profile from {args.prior_dir}/individual_differences_merged.csv")
    chunks = _load_chunk_profile(args.prior_dir)
    print(f"       {len(chunks)} subjects")

    print(f"[load] behavioral data from {args.behavioral_csv}")
    behav = _load_behavioral(args.behavioral_csv)
    print(f"       {len(behav)} subjects after ≥{MIN_TASKS}-task filter "
          f"({behav['top_solver'].sum()} top, {(~behav['top_solver']).sum()} rest)")

    merged = chunks.merge(behav, on="subject", how="inner")
    print(f"[merge] {len(merged)} subjects in common")
    merged.to_csv(os.path.join(args.prior_dir,
                               "chunking_top_solvers_merged.csv"),
                  index=False)

    # ── A: timing correlations ────────────────────────────────────────────────
    avail_chunk = [f for f in CHUNK_FEATURES if f in merged.columns]
    avail_timing = [f for f in TIMING_FEATURES if f in merged.columns]

    print("\n[A] chunk-feature × timing Spearman correlations")
    timing_corr = _chunk_timing_correlations(merged, avail_chunk, avail_timing)
    pivot = timing_corr.pivot(index="chunk_feature",
                               columns="timing_feature",
                               values="rho")
    print(pivot.round(3).to_string())
    timing_corr.to_csv(os.path.join(args.prior_dir,
                                    "chunking_timing_correlations.csv"),
                       index=False)

    # ── A2: feature × accuracy (continuous) ───────────────────────────────────
    print("\n[A2] feature × accuracy Spearman correlations (continuous outcome)")
    acc_feats = avail_chunk + avail_timing
    acc_corr = _accuracy_correlations(merged, acc_feats)
    print(acc_corr.round(3).to_string(index=False))
    acc_corr.to_csv(os.path.join(args.prior_dir,
                                 "chunking_accuracy_correlations.csv"),
                    index=False)

    # ── B: top-solver comparisons ─────────────────────────────────────────────
    print("\n[B] chunk features: top-solver vs rest (Mann-Whitney U + Cliff's δ)")
    top_stats = _top_solver_comparisons(merged, avail_chunk)
    print(top_stats.to_string(index=False))
    top_stats.to_csv(os.path.join(args.prior_dir,
                                  "chunking_top_solvers_stats.csv"),
                     index=False)

    # ── C: ICC ────────────────────────────────────────────────────────────────
    per_st_path = os.path.join(args.prior_dir,
                               "individual_differences_per_subj_task.csv")
    if os.path.exists(per_st_path):
        print(f"\n[C] ICC of chunk features per subject × task ({per_st_path})")
        per_st = pd.read_csv(per_st_path)
        icc_df = _chunk_icc(per_st, avail_chunk)
        print(icc_df.round(3).to_string(index=False))
        icc_df.to_csv(os.path.join(args.prior_dir, "chunking_icc.csv"),
                      index=False)
    else:
        print(f"\n[C] ICC skipped — {per_st_path} not found.")
        print("    Re-run individual_differences_chunking.py to generate it.")

    print("\n[done]")


if __name__ == "__main__":
    main()
