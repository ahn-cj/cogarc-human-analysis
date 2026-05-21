"""
Regression controls for task difficulty and trajectory length.

For each chunk feature, we fit a per-trial regression

    feature_value  ~  task_difficulty  +  log(trajectory_length)

at the (subject, task) level, take the residuals, and average them per
subject to obtain a *controlled* per-subject chunking profile.  All
downstream §3 and §4 analyses are then re-run on this controlled profile
so we can see whether the chunking-style effects survive after accounting
for the difficulty of the tasks each person attempted and the length of
their solution.

Outputs (prior_analysis/):
    controlled_per_subj_task.csv      one row per (subject, task) with raw
                                       and residualised feature values
    controlled_subject_profile.csv    one row per subject, residual means
    controlled_top_solver_stats.csv   §3 top-vs-rest on controlled profile
    controlled_section4_regression.csv §4 joint rank regression on controlled
                                       profile
    controls_regression_diagnostics.csv  per-feature R² of the control model

Justification of choices:
    * Per-trial regression (not per-task) so we use the full 14k cells of
      information rather than 75 task-mean rows.
    * log(trajectory_length) because trajectory length is heavily
      right-skewed (median 27 edits, max ~2500); the log gives a roughly
      symmetric covariate without throwing away the tail.
    * OLS on raw feature values (not ranks) for the residualisation step —
      we want the residual to live in the same units as the original
      feature so the per-subject mean is interpretable.  Downstream
      inferential tests on the residuals remain rank-based (Spearman ρ,
      Mann–Whitney, Cliff's δ, rank regression).
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata, mannwhitneyu


# ── feature lists ─────────────────────────────────────────────────────────────

# Per-chunk features — averaged across chunks within each (subject, task) cell
CHUNK_LEVEL_FEATURES = [
    "size", "n_cells", "is_connected", "fill_ratio",
    "color_homogeneity", "bbox_area", "nn_chain_rate", "success_iou_best",
]
# Per-trajectory features — one value per (subject, task) cell already
TRAJ_LEVEL_FEATURES = ["n_chunks_total"]

ALL_FEATURES = CHUNK_LEVEL_FEATURES + TRAJ_LEVEL_FEATURES


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


def _aggregate_per_subject_task(chunks: pd.DataFrame) -> pd.DataFrame:
    """Per-(subject, task) means of chunk features + trajectory length."""
    agg = {f: "mean" for f in CHUNK_LEVEL_FEATURES}
    agg["n_chunks_total"] = "first"        # already per-trajectory
    agg["size"]           = ["mean", "sum"]  # capture both mean size and total edits
    per_st = chunks.groupby(["subject_id", "task_id"]).agg(agg)
    # Flatten MultiIndex from the size dual-agg
    per_st.columns = [
        "size" if c == ("size", "mean")
        else "trajectory_length" if c == ("size", "sum")
        else c[0] if isinstance(c, tuple) else c
        for c in per_st.columns
    ]
    return per_st.reset_index()


def _attach_difficulty(per_st: pd.DataFrame, trial_scores_path: str) -> pd.DataFrame:
    ts = pd.read_csv(trial_scores_path)
    ts["task_id"] = ts["trial"].str.replace(".json", "", regex=False)
    ts = ts.rename(columns={"attempt_score": "task_difficulty"})[["task_id", "task_difficulty"]]
    merged = per_st.merge(ts, on="task_id", how="left")
    n_missing = merged["task_difficulty"].isna().sum()
    if n_missing:
        print(f"[warn] {n_missing}/{len(merged)} rows missing difficulty score")
    return merged


def _residualise(per_st: pd.DataFrame, features: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress each feature on difficulty + log(trajectory_length); store residuals.

    Returns
    -------
    per_st_resid : DataFrame
        Adds `{feature}_resid` columns for every feature.
    diagnostics : DataFrame
        Per-feature R², regression coefficients, n.
    """
    per_st = per_st.copy()
    per_st["log_traj_len"] = np.log(per_st["trajectory_length"].clip(lower=1))

    diag_rows = []
    for feat in features:
        df = per_st[[feat, "task_difficulty", "log_traj_len"]].dropna()
        X = sm.add_constant(df[["task_difficulty", "log_traj_len"]])
        res = sm.OLS(df[feat], X).fit()

        # residuals (NaN for rows with missing covariates)
        full_X = sm.add_constant(per_st[["task_difficulty", "log_traj_len"]],
                                  has_constant="add")
        # Use predict to handle the full set, including NaN propagation
        pred = res.predict(full_X)
        per_st[f"{feat}_resid"] = per_st[feat] - pred

        diag_rows.append({
            "feature": feat,
            "n": int(res.nobs),
            "R2": float(res.rsquared),
            "beta_difficulty": float(res.params["task_difficulty"]),
            "p_difficulty": float(res.pvalues["task_difficulty"]),
            "beta_log_traj_len": float(res.params["log_traj_len"]),
            "p_log_traj_len": float(res.pvalues["log_traj_len"]),
        })

    return per_st, pd.DataFrame(diag_rows)


def _per_subject_residual_profile(per_st_resid: pd.DataFrame,
                                  features: List[str]) -> pd.DataFrame:
    """Mean of residuals per subject — the controlled per-subject profile."""
    resid_cols = [f"{f}_resid" for f in features]
    profile = (per_st_resid.groupby("subject_id")[resid_cols]
                          .mean().reset_index())
    profile = profile.rename(columns={"subject_id": "subject"})
    return profile


# ── §3: top-solver comparison on controlled features ─────────────────────────

def _top_solver_comparison_controlled(profile: pd.DataFrame,
                                      behavioral_csv: str,
                                      features: List[str]) -> pd.DataFrame:
    """Mann-Whitney + Cliff's δ for top vs rest on each controlled feature."""
    behav = pd.read_csv(behavioral_csv)
    behav["correct"] = behav["final_outcome"].str.lower() == "success"
    subj = (behav.groupby("subject")
                  .agg(n_tasks=("trial", "nunique"),
                       accuracy=("correct", "mean"))
                  .reset_index())
    subj = subj[subj["n_tasks"] >= 40]
    subj["top_solver"] = subj["accuracy"] >= 0.95

    merged = profile.merge(subj[["subject", "accuracy", "top_solver", "n_tasks"]],
                            on="subject", how="inner")

    rows = []
    top_mask  = merged["top_solver"].astype(bool)
    rest_mask = ~top_mask
    for f in features:
        col = f"{f}_resid"
        top  = merged.loc[top_mask, col].dropna().values
        rest = merged.loc[rest_mask, col].dropna().values
        if len(top) < 5 or len(rest) < 5:
            continue
        _, p = mannwhitneyu(top, rest, alternative="two-sided")
        rows.append({
            "feature": f,
            "n_top": len(top), "n_rest": len(rest),
            "top_median":  float(np.median(top)),
            "rest_median": float(np.median(rest)),
            "median_diff": float(np.median(top) - np.median(rest)),
            "cliffs_delta": float(_cliffs_delta(top, rest)),
            "p_mannwhitney": float(p),
        })
    out = pd.DataFrame(rows).sort_values("cliffs_delta", key=np.abs, ascending=False)
    return out, merged


# ── §4: joint rank regression on controlled chunk size ───────────────────────

def _joint_rank_regression_controlled(merged: pd.DataFrame,
                                       size_col: str = "size_resid") -> pd.DataFrame:
    """Replicate the Section 4 rank regression using the controlled size."""
    behav_csv = str(_paths.BEHAVIORAL_CSV)
    behav = pd.read_csv(behav_csv)
    behav["correct"] = behav["final_outcome"].str.lower() == "success"
    subj = (behav.groupby("subject")
                  .agg(mean_rt_median=("mean_rt_between_edits", "median"))
                  .reset_index())

    df = merged.merge(subj, on="subject", how="inner")
    ok = df[[size_col, "mean_rt_median", "accuracy"]].notna().all(axis=1)
    sub = df[ok].copy()
    n = len(sub)

    sz = _rank(sub[size_col].values)
    rt = _rank(sub["mean_rt_median"].values)
    y  = _rank(sub["accuracy"].values)

    def _fit(X_cols):
        X = sm.add_constant(np.column_stack(X_cols) if len(X_cols) > 1 else X_cols[0])
        return sm.OLS(y, X).fit()

    m1 = _fit([sz])
    m2 = _fit([rt])
    m3 = _fit([sz, rt])

    rows = [
        {"model": "size_only_ctrl", "R2": m1.rsquared,
         "beta_size": m1.params[1], "p_size": m1.pvalues[1],
         "beta_rt": np.nan, "p_rt": np.nan, "n": n},
        {"model": "rt_only", "R2": m2.rsquared,
         "beta_size": np.nan, "p_size": np.nan,
         "beta_rt": m2.params[1], "p_rt": m2.pvalues[1], "n": n},
        {"model": "joint_ctrl", "R2": m3.rsquared,
         "beta_size": m3.params[1], "p_size": m3.pvalues[1],
         "beta_rt": m3.params[2], "p_rt": m3.pvalues[2], "n": n},
        {"model": "unique_size_ctrl", "R2": m3.rsquared - m2.rsquared,
         "beta_size": np.nan, "p_size": np.nan,
         "beta_rt": np.nan, "p_rt": np.nan, "n": n},
        {"model": "unique_rt", "R2": m3.rsquared - m1.rsquared,
         "beta_size": np.nan, "p_size": np.nan,
         "beta_rt": np.nan, "p_rt": np.nan, "n": n},
        {"model": "shared", "R2": m1.rsquared + m2.rsquared - m3.rsquared,
         "beta_size": np.nan, "p_size": np.nan,
         "beta_rt": np.nan, "p_rt": np.nan, "n": n},
    ]
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--trial_scores",
                    default=str(_paths.BEHAVIORAL_DATA_DIR / "trial_scores.csv"))
    ap.add_argument("--behavioral_csv",
                    default=str(_paths.BEHAVIORAL_CSV))
    args = ap.parse_args()

    # ── Load + aggregate per (subject, task) ──────────────────────────────────
    print("[load] chunks_per_trajectory.csv")
    chunks = pd.read_csv(os.path.join(args.prior_dir, "chunks_per_trajectory.csv"))
    print(f"       {len(chunks):,} chunks")

    per_st = _aggregate_per_subject_task(chunks)
    print(f"[agg ] {len(per_st):,} (subject, task) cells")

    # ── Attach difficulty + residualise ───────────────────────────────────────
    per_st = _attach_difficulty(per_st, args.trial_scores)

    print("\n[fit ] regressing each feature on difficulty + log(trajectory_length)")
    per_st_resid, diag = _residualise(per_st, ALL_FEATURES)
    print(diag.round(4).to_string(index=False))
    diag.to_csv(os.path.join(args.prior_dir,
                              "controls_regression_diagnostics.csv"), index=False)
    per_st_resid.to_csv(os.path.join(args.prior_dir,
                                      "controlled_per_subj_task.csv"), index=False)

    # ── Build controlled per-subject profile ──────────────────────────────────
    profile = _per_subject_residual_profile(per_st_resid, ALL_FEATURES)
    profile.to_csv(os.path.join(args.prior_dir,
                                 "controlled_subject_profile.csv"), index=False)
    print(f"\n[prof] {len(profile)} subjects in controlled profile")

    # ── §3 top-solver comparison on controlled features ───────────────────────
    print("\n[B] top-solver comparison on CONTROLLED features (Mann-Whitney, Cliff's δ)")
    ts_stats, merged = _top_solver_comparison_controlled(
        profile, args.behavioral_csv, ALL_FEATURES)
    print(ts_stats.to_string(index=False))
    ts_stats.to_csv(os.path.join(args.prior_dir,
                                  "controlled_top_solver_stats.csv"), index=False)

    # ── §4 joint rank regression on controlled chunk size ─────────────────────
    print("\n[C] joint rank regression on CONTROLLED chunk size + RT")
    joint = _joint_rank_regression_controlled(merged, size_col="size_resid")
    print(joint.round(4).to_string(index=False))
    joint.to_csv(os.path.join(args.prior_dir,
                               "controlled_section4_regression.csv"), index=False)

    print("\n[done]")


if __name__ == "__main__":
    main()
