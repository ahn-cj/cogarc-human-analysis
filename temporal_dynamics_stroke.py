"""
Within-experiment temporal dynamics, on the motor stroke measures.

Question: do participants' chunking / pacing patterns change over the 75-problem
session, or are they established early? Same three analyses as before, but with
chunk granularity measured by cells-per-stroke (and n-strokes) rather than
pause-segmented chunk counts:

    (A) Group-level practice effects: LMM  measure ~ order + (1 + order | subject)
        and the order × top_solver interaction.
    (B) Early-vs-late within-subject stability (median split by order).
    (C) Quartile trajectories: top solvers vs rest (descriptive grouping).

Measures:
    cells_per_stroke   chunk granularity (primary)
    n_strokes          n strokes per trajectory
    rt                 inter-edit RT (execution pace)
    deliberation_time  time to first edit (planning pace)

Joins the Experiment-2 behavioral CSV (for trial `order`, rt, deliberation,
final_result) with stroke_chunking_per_st.csv (cells_per_stroke, n_strokes).

Outputs prior_analysis/temporal_dynamics_stroke_{lmm,stability,quartiles,early_late}.csv
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=UserWarning)

BEHAV_CSV = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
             "Behavioral data/behavioral_measures_filtered.csv")

MEASURES = [
    ("cells_per_stroke",  "linear", "cells per stroke (chunk granularity)"),
    ("n_strokes",         "linear", "n strokes per trajectory"),
    ("rt",                "log",    "log(inter-edit RT, s)  [execution pace]"),
    ("deliberation_time", "log",    "log(deliberation time, s)  [planning pace]"),
]
MEASURES_BY_NAME = {m[0]: m[1] for m in MEASURES}
TOP_SOLVER_ACC, MIN_TASKS = 0.95, 40


def _transform(s, kind):
    return np.log(s.clip(lower=0.05)) if kind == "log" else s.astype(float)


def _load():
    b = pd.read_csv(BEHAV_CSV)
    b["subject"] = b["subject"].astype(str)
    b["task_id"] = b["trial"].astype(str).str.replace(".json", "", regex=False)
    s = pd.read_csv("prior_analysis/stroke_chunking_per_st.csv")
    s["subject"] = s["subject_id"].astype(str)
    s["task_id"] = s["task_id"].astype(str)
    df = b.merge(s[["subject", "task_id", "cells_per_stroke", "n_strokes"]],
                 on=["subject", "task_id"], how="inner")
    df["correct"] = df["final_result"].astype(str).str.lower() == "success"
    subj = (df.groupby("subject")
              .agg(n_tasks=("trial", "nunique"), accuracy=("correct", "mean"))
              .reset_index())
    subj["top_solver"] = (subj["accuracy"] >= TOP_SOLVER_ACC) & (subj["n_tasks"] >= MIN_TASKS)
    subj["group"] = np.where(subj["top_solver"], "top", "rest")
    return df.merge(subj, on="subject", how="left")


def _fit_lmm(df, outcome):
    data = df.dropna(subset=[outcome, "order", "subject"]).copy()
    data = data[data["n_tasks"] >= MIN_TASKS]
    data["order_c"] = (data["order"] - 38) / 10.0
    data["y"] = _transform(data[outcome], MEASURES_BY_NAME[outcome])
    data["top"] = data["top_solver"].astype(int)
    out = {"outcome": outcome, "n_obs": len(data), "n_subjects": data["subject"].nunique()}
    try:
        m = smf.mixedlm("y ~ order_c", data, groups=data["subject"],
                        re_formula="~ order_c").fit(method="lbfgs")
        out["main_b_order_per10"] = float(m.fe_params["order_c"])
        out["main_p_order"] = float(m.pvalues["order_c"])
        out["main_var_slope"] = float(m.cov_re.iloc[1, 1]) if m.cov_re.shape[0] > 1 else np.nan
    except Exception as e:
        out["main_error"] = str(e)
    try:
        m = smf.mixedlm("y ~ order_c * top", data, groups=data["subject"],
                        re_formula="~ order_c").fit(method="lbfgs")
        out["int_b_order_x_top"] = float(m.fe_params["order_c:top"])
        out["int_p_order_x_top"] = float(m.pvalues["order_c:top"])
        out["int_b_top"] = float(m.fe_params["top"])
        out["int_p_top"] = float(m.pvalues["top"])
    except Exception as e:
        out["int_error"] = str(e)
    return out


def _early_late(df, outcome):
    data = df.dropna(subset=[outcome, "order", "subject"]).copy()
    data = data[data["n_tasks"] >= MIN_TASKS]
    data["y"] = _transform(data[outcome], MEASURES_BY_NAME[outcome])
    rows = []
    for sid, g in data.groupby("subject"):
        if len(g) < 10:
            continue
        mo = g["order"].median()
        rows.append({"subject": sid, "outcome": outcome,
                     "early": g.loc[g["order"] <= mo, "y"].mean(),
                     "late": g.loc[g["order"] > mo, "y"].mean()})
    el = pd.DataFrame(rows).dropna()
    rho, p = spearmanr(el["early"], el["late"])
    return {"outcome": outcome, "n_subjects": len(el),
            "spearman_rho": float(rho), "spearman_p": float(p),
            "mean_change": float((el["late"] - el["early"]).mean())}, el


def _quartiles(df):
    d = df[df["n_tasks"] >= MIN_TASKS].copy()
    d["quartile"] = pd.cut(d["order"], bins=[0, 19, 37, 56, 75],
                           labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    rows = []
    for outcome, kind, _ in MEASURES:
        sub = d.dropna(subset=[outcome]).assign(y=lambda x: _transform(x[outcome], kind))
        for grp, gd in sub.groupby("group", observed=True):
            for q, qd in gd.groupby("quartile", observed=True):
                if len(qd) < 5:
                    continue
                ps = qd.groupby("subject")["y"].mean()
                rows.append({"outcome": outcome, "group": grp, "quartile": str(q),
                             "n_subjects": len(ps), "mean": float(ps.mean()),
                             "se": float(ps.std(ddof=1) / np.sqrt(len(ps)))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    args = ap.parse_args()
    df = _load()
    n_top = df.groupby("subject")["top_solver"].first().sum()
    print(f"[load] {len(df):,} trials • {df['subject'].nunique()} subjects • {int(n_top)} top solvers\n")

    print("[A] LMM: measure ~ order + (1+order|subject), per 10 trials")
    lmm = pd.DataFrame([_fit_lmm(df, m) for m, _, _ in MEASURES])
    print(lmm[["outcome", "main_b_order_per10", "main_p_order",
               "int_b_order_x_top", "int_p_order_x_top"]].round(4).to_string(index=False))
    lmm.to_csv(f"{args.prior_dir}/temporal_dynamics_stroke_lmm.csv", index=False)

    print("\n[B] early-vs-late within-subject stability (median split by order)")
    stab, els = [], []
    for m, _, _ in MEASURES:
        r, el = _early_late(df, m)
        stab.append(r); els.append(el)
    stab = pd.DataFrame(stab)
    print(stab.round(4).to_string(index=False))
    stab.to_csv(f"{args.prior_dir}/temporal_dynamics_stroke_stability.csv", index=False)
    pd.concat(els, ignore_index=True).to_csv(
        f"{args.prior_dir}/temporal_dynamics_stroke_early_late.csv", index=False)

    print("\n[C] quartile trajectories: top vs rest")
    q = _quartiles(df)
    print(q.round(3).to_string(index=False))
    q.to_csv(f"{args.prior_dir}/temporal_dynamics_stroke_quartiles.csv", index=False)
    print("\n[done]")


if __name__ == "__main__":
    main()
