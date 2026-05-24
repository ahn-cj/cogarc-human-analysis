"""
Within-experiment temporal dynamics of chunking and pacing style.

Question: do participants' chunking / pacing patterns change over the course
of the 75-problem session, or are they established early?

Three complementary analyses on the Experiment-2-filtered dataset
(behavioral_measures_filtered.csv, 218 subjects × ≤75 trials each, with
trial `order` ranging 1–75):

    (A) Group-level practice effects with individual variation
        Linear mixed-effects model per measure:
            measure  ~  order  +  (1 + order | subject)
        Fixed effect of order = population-level practice trend.
        Random slope of order per subject = how much individuals vary
        in their trajectories.  Same model fit with order × top_solver
        interaction to test whether top solvers learn at a different rate.

    (B) Early-vs-late within-subject stability
        For each subject, split trials in half by `order`.  Correlate
        first-half per-subject mean with second-half per-subject mean.
        High correlation = style established early.

    (C) Quartile trajectories: top solvers vs rest
        Bin trials by `order` quartile (Q1: 1-19, Q2: 20-37, Q3: 38-56,
        Q4: 57-75).  Per-quartile means for top vs rest.

Measures analysed:
    rt                  inter-edit RT (s)
    deliberation_time   time to first edit (s)
    num_chunks          n chunks per trajectory

Outputs (prior_analysis/):
    temporal_dynamics_lmm.csv         group-level + interaction results
    temporal_dynamics_early_late.csv  per-subject early vs late means
    temporal_dynamics_stability.csv   early-vs-late correlations per measure
    temporal_dynamics_quartiles.csv   group × quartile means
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=UserWarning)


# ── config ────────────────────────────────────────────────────────────────────

BEHAV_CSV = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
             "Behavioral data/behavioral_measures_filtered.csv")

# Measures to analyse and how to transform them
MEASURES = [
    ("rt",                "log",   "log(inter-edit RT, s)"),
    ("deliberation_time", "log",   "log(deliberation time, s)"),
    ("num_chunks",        "linear","n chunks per trajectory"),
]

TOP_SOLVER_ACC = 0.95
MIN_TASKS      = 40


# ── helpers ───────────────────────────────────────────────────────────────────

def _transform(s: pd.Series, kind: str) -> pd.Series:
    if kind == "log":
        # add a tiny offset for any zero / near-zero values
        return np.log(s.clip(lower=0.05))
    return s.astype(float)


def _attach_subject_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-subject accuracy, n_tasks, top_solver, group."""
    df = df.copy()
    df["correct"] = df["final_result"].astype(str).str.lower() == "success"
    subj = (df.groupby("subject")
              .agg(n_tasks=("trial", "nunique"),
                   accuracy=("correct", "mean"))
              .reset_index())
    subj["top_solver"] = (subj["accuracy"] >= TOP_SOLVER_ACC) & (subj["n_tasks"] >= MIN_TASKS)
    subj["group"]      = np.where(subj["top_solver"], "top", "rest")
    return df.merge(subj, on="subject", how="left")


# ── A: group-level mixed-effects + interaction with top_solver ───────────────

def _fit_lmm(df: pd.DataFrame, outcome: str) -> dict:
    """Two LMMs per outcome:
       main:  y ~ order_c        +  (1 + order_c | subject)
       inter: y ~ order_c * top  +  (1 + order_c | subject)
    Returns a dict of fixed-effect estimates + random-effect variances.
    order_c = (order - 38) / 10  so a 1-unit coef = effect per 10 trials.
    """
    data = df.dropna(subset=[outcome, "order", "subject"]).copy()
    data = data[data["n_tasks"] >= MIN_TASKS]
    data["order_c"] = (data["order"] - 38) / 10.0
    data["y"]       = _transform(data[outcome], MEASURES_BY_NAME[outcome])
    data["top"]     = data["top_solver"].astype(int)

    out = {"outcome": outcome, "n_obs": len(data),
           "n_subjects": data["subject"].nunique()}

    # — main effect of order —
    try:
        md_main = smf.mixedlm("y ~ order_c", data, groups=data["subject"],
                              re_formula="~ order_c")
        m_main = md_main.fit(method="lbfgs")
        out["main_intercept"]       = float(m_main.fe_params["Intercept"])
        out["main_b_order_per10"]   = float(m_main.fe_params["order_c"])
        out["main_p_order"]         = float(m_main.pvalues["order_c"])
        # Random-effect variances
        re_cov = m_main.cov_re
        out["main_var_intercept"]   = float(re_cov.iloc[0, 0])
        out["main_var_slope"]       = float(re_cov.iloc[1, 1]) if re_cov.shape[0] > 1 else float("nan")
    except Exception as e:
        out["main_error"] = str(e)

    # — interaction with top_solver —
    try:
        md_int = smf.mixedlm("y ~ order_c * top", data, groups=data["subject"],
                             re_formula="~ order_c")
        m_int = md_int.fit(method="lbfgs")
        out["int_b_order_per10"]  = float(m_int.fe_params["order_c"])
        out["int_b_top"]          = float(m_int.fe_params["top"])
        out["int_b_order_x_top"]  = float(m_int.fe_params["order_c:top"])
        out["int_p_order"]        = float(m_int.pvalues["order_c"])
        out["int_p_top"]          = float(m_int.pvalues["top"])
        out["int_p_order_x_top"]  = float(m_int.pvalues["order_c:top"])
    except Exception as e:
        out["int_error"] = str(e)

    return out


# ── B: early-vs-late within-subject stability ─────────────────────────────────

def _early_late_stability(df: pd.DataFrame, outcome: str) -> dict:
    data = df.dropna(subset=[outcome, "order", "subject"]).copy()
    data = data[data["n_tasks"] >= MIN_TASKS]
    data["y"] = _transform(data[outcome], MEASURES_BY_NAME[outcome])

    rows = []
    for sid, g in data.groupby("subject"):
        if len(g) < 10:
            continue
        median_ord = g["order"].median()
        early = g.loc[g["order"] <= median_ord, "y"].mean()
        late  = g.loc[g["order"] >  median_ord, "y"].mean()
        rows.append({"subject": sid, "early": early, "late": late,
                     "n_early": int((g["order"] <= median_ord).sum()),
                     "n_late":  int((g["order"] >  median_ord).sum())})
    el = pd.DataFrame(rows).dropna()

    rho, p_rho = spearmanr(el["early"], el["late"])
    r,   p_r   = pearsonr(el["early"], el["late"])
    delta = (el["late"] - el["early"]).values
    return {
        "outcome": outcome, "n_subjects": len(el),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "pearson_r":    float(r),   "pearson_p":  float(p_r),
        "mean_change":  float(delta.mean()),
        "median_change":float(np.median(delta)),
        "ci_lo": float(np.percentile(delta, 2.5)),
        "ci_hi": float(np.percentile(delta, 97.5)),
        # Per-subject table also returned separately
        "_early_late_df": el,
    }


# ── C: quartile trajectories ─────────────────────────────────────────────────

def _quartile_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(quartile × group) mean for each outcome (and SE)."""
    d = df[df["n_tasks"] >= MIN_TASKS].copy()
    # Use fixed bins (1-19, 20-37, 38-56, 57-75)
    bins   = [0, 19, 37, 56, 75]
    labels = ["Q1 (1-19)", "Q2 (20-37)", "Q3 (38-56)", "Q4 (57-75)"]
    d["quartile"] = pd.cut(d["order"], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for outcome, kind, _ in MEASURES:
        sub = d.dropna(subset=[outcome])
        sub = sub.assign(y=_transform(sub[outcome], kind))
        for grp_name, gd in sub.groupby("group", observed=True):
            for q, qd in gd.groupby("quartile", observed=True):
                if len(qd) < 5:
                    continue
                # First aggregate per subject, then take mean of subject means
                # (so participants who attempted more trials don't dominate).
                per_subj = qd.groupby("subject")["y"].mean()
                rows.append({
                    "outcome": outcome, "group": grp_name, "quartile": str(q),
                    "n_subjects": len(per_subj),
                    "mean": float(per_subj.mean()),
                    "se":   float(per_subj.std(ddof=1) / np.sqrt(len(per_subj))),
                })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

MEASURES_BY_NAME = {m[0]: m[1] for m in MEASURES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--csv", default=BEHAV_CSV)
    args = ap.parse_args()

    print(f"[load] {args.csv}")
    df = pd.read_csv(args.csv)
    df = _attach_subject_attrs(df)
    print(f"       {len(df):,} trials  •  {df['subject'].nunique()} subjects  "
          f"•  {(df['top_solver'] & (df['n_tasks']>=MIN_TASKS)).groupby(df['subject']).first().sum()} top solvers")

    # ── A: LMMs ───────────────────────────────────────────────────────────────
    print("\n[A] linear mixed-effects models  (measure ~ order + (1+order|subject))")
    lmm_rows = []
    for outcome, _, label in MEASURES:
        print(f"    fitting {outcome} …")
        res = _fit_lmm(df, outcome)
        lmm_rows.append(res)
    lmm_df = pd.DataFrame(lmm_rows)
    print(lmm_df.round(4).to_string(index=False))
    lmm_df.to_csv(os.path.join(args.prior_dir, "temporal_dynamics_lmm.csv"), index=False)

    # ── B: early-vs-late stability ────────────────────────────────────────────
    print("\n[B] early-vs-late within-subject stability  (median-split by order)")
    stab_rows = []
    el_dfs    = []
    for outcome, _, _ in MEASURES:
        res = _early_late_stability(df, outcome)
        el_dfs.append(res.pop("_early_late_df").assign(outcome=outcome))
        stab_rows.append(res)
    stab_df = pd.DataFrame(stab_rows)
    print(stab_df.round(4).to_string(index=False))
    stab_df.to_csv(os.path.join(args.prior_dir,
                                 "temporal_dynamics_stability.csv"), index=False)
    pd.concat(el_dfs, ignore_index=True).to_csv(
        os.path.join(args.prior_dir, "temporal_dynamics_early_late.csv"), index=False)

    # ── C: quartile trajectories ──────────────────────────────────────────────
    print("\n[C] quartile trajectories: top vs rest")
    quart_df = _quartile_trajectories(df)
    print(quart_df.round(3).to_string(index=False))
    quart_df.to_csv(os.path.join(args.prior_dir,
                                  "temporal_dynamics_quartiles.csv"), index=False)

    print("\n[done]")


if __name__ == "__main__":
    main()
