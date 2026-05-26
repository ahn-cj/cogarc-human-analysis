"""
Regression controls for task difficulty, trajectory length, and number of
examples — applied to BOTH chunking and pacing measures.

For each per-(subject, task) measure, we fit a per-trial OLS regression with
covariates tailored to the measure's confounds, take residuals, and average
or median them per subject to obtain a *controlled* per-subject profile.
All §3 (top-solver vs rest) and §4 (joint rank regression) analyses are
then re-run on the controlled profile.

Covariate sets
--------------
Chunking measures (size, n_cells, n_chunks_total, is_connected, fill_ratio,
                   color_homogeneity, bbox_area, nn_chain_rate, success_iou_best):
    feature  ~  task_difficulty  +  log(trajectory_length)

Inter-edit RT (mean_rt_between_edits):
    feature  ~  task_difficulty  +  log(trajectory_length)  +  n_examples

Deliberation time (deliberation_time):
    feature  ~  task_difficulty  +  log(trajectory_length)  +  n_examples
                                 +  example_view_time_before_first_edit

The example-view-time covariate for deliberation time isolates the
"planning-after-examples" component of deliberation_time — i.e. it asks
whether top solvers plan faster *once we control for how long they look at
the examples* (cf. §3.2.3).

Per-subject aggregation matches the chapter's existing measures:
    Chunking residuals → mean per subject  (matches per-subject chunk profile)
    Pacing residuals   → median per subject (matches deliberation_time_median,
                                              mean_rt_median in the merged CSV)

Outputs (prior_analysis/):
    controlled_per_subj_task.csv          per-trial raw + residuals (all measures)
    controlled_subject_profile.csv        per-subject residual aggregates
    controlled_top_solver_stats.csv       §3 top-vs-rest on controlled measures
    controlled_section4_regression.csv    §4 joint rank regression
                                          (controlled size + controlled RT)
    controls_regression_diagnostics.csv   per-feature R² of each control model
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata, mannwhitneyu


# ── feature lists and covariate sets ─────────────────────────────────────────

CHUNK_LEVEL_FEATURES = [
    "size", "n_cells", "is_connected", "fill_ratio",
    "color_homogeneity", "bbox_area", "nn_chain_rate", "success_iou_best",
]
TRAJ_LEVEL_FEATURES  = ["n_chunks_total"]
PACING_FEATURES      = ["mean_rt_between_edits", "deliberation_time"]

CHUNKING_FEATURES = CHUNK_LEVEL_FEATURES + TRAJ_LEVEL_FEATURES
ALL_FEATURES      = CHUNKING_FEATURES + PACING_FEATURES

# Per-measure covariate sets
COMMON_COVARIATES = ["task_difficulty", "log_traj_len"]
COVARIATE_SETS: Dict[str, List[str]] = {
    f: COMMON_COVARIATES.copy() for f in CHUNKING_FEATURES
}
COVARIATE_SETS["mean_rt_between_edits"] = COMMON_COVARIATES + ["n_examples"]
COVARIATE_SETS["deliberation_time"]     = (
    COMMON_COVARIATES + ["n_examples", "example_view_time_before_first_edit"]
)

# Aggregation rule for residuals (mean for chunking, median for pacing)
AGG_RULE = {f: "mean" for f in CHUNKING_FEATURES}
AGG_RULE.update({f: "median" for f in PACING_FEATURES})


# ── helpers ───────────────────────────────────────────────────────────────────

def _rank(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average")
    return (r - r.mean()) / r.std(ddof=1)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = int(np.sum(np.subtract.outer(x, y) > 0))
    lt = int(np.sum(np.subtract.outer(x, y) < 0))
    return (gt - lt) / (len(x) * len(y))


def _build_per_st_table(prior_dir: str, behavioral_csv: str) -> pd.DataFrame:
    """One row per (subject, task) cell with chunking aggregates,
    per-trial pacing measures, trajectory length, and task covariates."""
    chunks = pd.read_csv(os.path.join(prior_dir, "chunks_per_trajectory.csv"))
    agg = {f: "mean" for f in CHUNK_LEVEL_FEATURES}
    agg["n_chunks_total"] = "first"
    agg["size"] = ["mean", "sum"]
    per_st = chunks.groupby(["subject_id", "task_id"]).agg(agg)
    per_st.columns = [
        "size" if c == ("size", "mean")
        else "trajectory_length" if c == ("size", "sum")
        else c[0] if isinstance(c, tuple) else c
        for c in per_st.columns
    ]
    per_st = per_st.reset_index().rename(columns={"subject_id": "subject"})

    # Merge per-trial pacing + example_view_time
    behav = pd.read_csv(behavioral_csv)
    behav = behav.rename(columns={"trial": "task_id"})
    keep_cols = ["subject", "task_id",
                 "mean_rt_between_edits", "deliberation_time",
                 "example_view_time_before_first_edit"]
    per_st = per_st.merge(behav[keep_cols], on=["subject", "task_id"], how="left")

    # Attach task-level covariates
    ts = pd.read_csv(os.path.join("/Users/carolineahn/Documents/GitHub/"
                                   "CogARC-dataRepository/Behavioral data/",
                                   "trial_scores.csv"))
    ts["task_id"] = ts["trial"].str.replace(".json", "", regex=False)
    ts = ts.rename(columns={"attempt_score": "task_difficulty"})
    per_st = per_st.merge(ts[["task_id", "task_difficulty"]],
                           on="task_id", how="left")

    ne = pd.read_csv(os.path.join(prior_dir, "task_n_examples.csv"))
    per_st = per_st.merge(ne[["task_id", "n_examples"]],
                           on="task_id", how="left")

    # Derived covariate
    per_st["log_traj_len"] = np.log(per_st["trajectory_length"].clip(lower=1))
    return per_st


def _residualise(per_st: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = per_st.copy()
    rows = []
    for feat, covars in COVARIATE_SETS.items():
        cols_needed = [feat] + covars
        df = per_st[cols_needed].dropna()
        if len(df) < 30:
            print(f"  [skip] {feat}: only {len(df)} valid rows")
            continue
        X = sm.add_constant(df[covars])
        res = sm.OLS(df[feat], X).fit()

        # Compute residuals for the full table (NaN-safe)
        full_X = sm.add_constant(per_st[covars], has_constant="add")
        pred   = res.predict(full_X)
        out[f"{feat}_resid"] = per_st[feat] - pred

        row = dict(feature=feat, n=int(res.nobs), R2=float(res.rsquared))
        # Capture each coefficient + p
        for c in covars:
            row[f"beta_{c}"] = float(res.params[c])
            row[f"p_{c}"]    = float(res.pvalues[c])
        rows.append(row)
    return out, pd.DataFrame(rows)


def _per_subject_profile(per_st_resid: pd.DataFrame) -> pd.DataFrame:
    """Aggregate residuals per subject — mean for chunking, median for pacing."""
    cols = {f"{f}_resid": AGG_RULE[f]
            for f in ALL_FEATURES if f"{f}_resid" in per_st_resid.columns}
    profile = (per_st_resid.groupby("subject")
                            .agg(cols)
                            .reset_index())
    return profile


def _top_solver_comparison(profile: pd.DataFrame,
                            behavioral_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    behav = pd.read_csv(behavioral_csv)
    behav["correct"] = behav["final_outcome"].astype(str).str.lower() == "success"
    subj = (behav.groupby("subject")
                  .agg(n_tasks=("trial","nunique"), accuracy=("correct","mean"))
                  .reset_index())
    subj = subj[subj["n_tasks"] >= 40]
    subj["top_solver"] = subj["accuracy"] >= 0.95

    merged = profile.merge(subj[["subject", "accuracy", "top_solver", "n_tasks"]],
                            on="subject", how="inner")

    rows = []
    top_mask  = merged["top_solver"].astype(bool)
    rest_mask = ~top_mask
    for f in ALL_FEATURES:
        col = f"{f}_resid"
        if col not in merged.columns:
            continue
        top  = merged.loc[top_mask, col].dropna().values
        rest = merged.loc[rest_mask, col].dropna().values
        if len(top) < 5 or len(rest) < 5:
            continue
        _, p = mannwhitneyu(top, rest, alternative="two-sided")
        rows.append({
            "feature": f, "n_top": len(top), "n_rest": len(rest),
            "top_median":  float(np.median(top)),
            "rest_median": float(np.median(rest)),
            "median_diff": float(np.median(top) - np.median(rest)),
            "cliffs_delta": float(_cliffs_delta(top, rest)),
            "p_mannwhitney": float(p),
        })
    out = pd.DataFrame(rows).sort_values("cliffs_delta", key=np.abs, ascending=False)
    return out, merged


def _joint_rank_regression(merged: pd.DataFrame) -> pd.DataFrame:
    """rank(accuracy) ~ rank(controlled size) + rank(controlled RT)."""
    cols = ["size_resid", "mean_rt_between_edits_resid", "accuracy"]
    ok = merged[cols].notna().all(axis=1)
    sub = merged[ok].copy()
    n = len(sub)

    sz = _rank(sub["size_resid"].values)
    rt = _rank(sub["mean_rt_between_edits_resid"].values)
    y  = _rank(sub["accuracy"].values)

    def _fit(cols_X):
        X = sm.add_constant(np.column_stack(cols_X) if len(cols_X) > 1 else cols_X[0])
        return sm.OLS(y, X).fit()

    m1 = _fit([sz])
    m2 = _fit([rt])
    m3 = _fit([sz, rt])

    rows = [
        {"model": "size_ctrl_only",   "R2": m1.rsquared,
         "beta_size": m1.params[1], "p_size": m1.pvalues[1],
         "beta_rt": np.nan, "p_rt": np.nan, "n": n},
        {"model": "rt_ctrl_only",     "R2": m2.rsquared,
         "beta_size": np.nan, "p_size": np.nan,
         "beta_rt": m2.params[1], "p_rt": m2.pvalues[1], "n": n},
        {"model": "joint_both_ctrl", "R2": m3.rsquared,
         "beta_size": m3.params[1], "p_size": m3.pvalues[1],
         "beta_rt": m3.params[2], "p_rt": m3.pvalues[2], "n": n},
        {"model": "unique_size_ctrl", "R2": m3.rsquared - m2.rsquared,
         "beta_size": np.nan, "p_size": np.nan, "beta_rt": np.nan, "p_rt": np.nan, "n": n},
        {"model": "unique_rt_ctrl",   "R2": m3.rsquared - m1.rsquared,
         "beta_size": np.nan, "p_size": np.nan, "beta_rt": np.nan, "p_rt": np.nan, "n": n},
        {"model": "shared",           "R2": m1.rsquared + m2.rsquared - m3.rsquared,
         "beta_size": np.nan, "p_size": np.nan, "beta_rt": np.nan, "p_rt": np.nan, "n": n},
    ]
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--behavioral_csv", default=str(_paths.BEHAVIORAL_CSV))
    args = ap.parse_args()

    print("[load + merge] building per-(subject, task) table")
    per_st = _build_per_st_table(args.prior_dir, args.behavioral_csv)
    print(f"  {len(per_st):,} (subject, task) cells, "
          f"{per_st['subject'].nunique()} subjects")

    print("\n[fit] residualising each feature on its covariate set")
    per_st_resid, diag = _residualise(per_st)
    print(diag.round(4).to_string(index=False))
    diag.to_csv(os.path.join(args.prior_dir,
                              "controls_regression_diagnostics.csv"), index=False)
    per_st_resid.to_csv(os.path.join(args.prior_dir,
                                      "controlled_per_subj_task.csv"), index=False)

    profile = _per_subject_profile(per_st_resid)
    profile.to_csv(os.path.join(args.prior_dir,
                                 "controlled_subject_profile.csv"), index=False)
    print(f"\n[prof] {len(profile)} subjects in controlled profile")

    print("\n[§3] top-solver comparison on CONTROLLED features")
    ts_stats, merged = _top_solver_comparison(profile, args.behavioral_csv)
    print(ts_stats.to_string(index=False))
    ts_stats.to_csv(os.path.join(args.prior_dir,
                                  "controlled_top_solver_stats.csv"), index=False)

    print("\n[§4] joint rank regression: BOTH measures controlled")
    joint = _joint_rank_regression(merged)
    print(joint.round(4).to_string(index=False))
    joint.to_csv(os.path.join(args.prior_dir,
                               "controlled_section4_regression.csv"), index=False)

    print("\n[done]")


if __name__ == "__main__":
    main()
