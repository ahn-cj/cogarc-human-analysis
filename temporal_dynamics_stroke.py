"""
Within-experiment temporal dynamics, on the motor stroke measures.

Question: do participants' chunking / pacing patterns change over the 75-problem
session, or are they established early? Same three analyses as before, but with
chunk granularity measured by cells-per-stroke (and n-strokes) rather than
pause-segmented chunk counts:

    (A) Group-level practice effects: LMM  measure ~ order + (1 + order | subject),
        plus a continuous moderation  measure ~ order * accuracy_z  (does the
        practice slope depend on accuracy?).
    (B) Early-vs-late within-subject stability (median split by order).
    (C) Per-quartile correlation with continuous accuracy: in each trial-order
        quartile, correlate each subject's mean measure with their overall
        accuracy. A stable correlation from Q1 onward = the predictive
        relationship is present from the start (no top-solver split).

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
MIN_TASKS = 40


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
    return df.merge(subj, on="subject", how="left")


def _spearman_ci(x, y, n_boot=2000, seed=1):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot); n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b[i], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(rho), float(lo), float(hi), float(p), int(n)


def _fit_lmm(df, outcome):
    data = df.dropna(subset=[outcome, "order", "subject", "accuracy"]).copy()
    data = data[data["n_tasks"] >= MIN_TASKS]
    data["order_c"] = (data["order"] - 38) / 10.0
    data["y"] = _transform(data[outcome], MEASURES_BY_NAME[outcome])
    acc = data.groupby("subject")["accuracy"].first()
    data["acc_z"] = (data["accuracy"] - acc.mean()) / acc.std()
    out = {"outcome": outcome, "n_obs": len(data), "n_subjects": data["subject"].nunique()}
    try:
        m = smf.mixedlm("y ~ order_c", data, groups=data["subject"],
                        re_formula="~ order_c").fit(method="lbfgs")
        out["main_b_order_per10"] = float(m.fe_params["order_c"])
        out["main_p_order"] = float(m.pvalues["order_c"])
    except Exception as e:
        out["main_error"] = str(e)
    try:  # continuous moderation by accuracy (replaces order x top_solver)
        m = smf.mixedlm("y ~ order_c * acc_z", data, groups=data["subject"],
                        re_formula="~ order_c").fit(method="lbfgs")
        out["b_order_x_acc"] = float(m.fe_params["order_c:acc_z"])
        out["p_order_x_acc"] = float(m.pvalues["order_c:acc_z"])
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


def _quartile_corr(df):
    """In each trial-order quartile, correlate each subject's mean measure with
    their overall (continuous) accuracy — and report the quartile mean for context."""
    d = df[df["n_tasks"] >= MIN_TASKS].copy()
    d["quartile"] = pd.cut(d["order"], bins=[0, 19, 37, 56, 75],
                           labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    acc = d.groupby("subject")["accuracy"].first()
    rows = []
    for outcome, kind, _ in MEASURES:
        sub = d.dropna(subset=[outcome]).assign(y=lambda x: _transform(x[outcome], kind))
        for q, qd in sub.groupby("quartile", observed=True):
            ps = qd.groupby("subject")["y"].mean()
            merged = pd.DataFrame({"y": ps, "accuracy": acc.reindex(ps.index)}).dropna()
            if len(merged) < 10:
                continue
            rho, lo, hi, p, n = _spearman_ci(merged["y"].values, merged["accuracy"].values)
            rows.append({"outcome": outcome, "quartile": str(q), "n_subjects": n,
                         "rho_with_accuracy": rho, "ci_lo": lo, "ci_hi": hi, "p": p,
                         "quartile_mean": float(ps.mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    args = ap.parse_args()
    df = _load()
    n_subj = df[df["n_tasks"] >= MIN_TASKS]["subject"].nunique()
    print(f"[load] {len(df):,} trials • {df['subject'].nunique()} subjects "
          f"({n_subj} with ≥{MIN_TASKS} tasks)\n")

    print("[A] LMM: measure ~ order + (1+order|subject), per 10 trials; "
          "+ order × accuracy moderation")
    lmm = pd.DataFrame([_fit_lmm(df, m) for m, _, _ in MEASURES])
    print(lmm[["outcome", "main_b_order_per10", "main_p_order",
               "b_order_x_acc", "p_order_x_acc"]].round(4).to_string(index=False))
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

    print("\n[C] per-quartile correlation with continuous accuracy "
          "(is the predictive relationship present from Q1?)")
    q = _quartile_corr(df)
    print(q[["outcome", "quartile", "n_subjects", "rho_with_accuracy",
             "ci_lo", "ci_hi", "p"]].round(3).to_string(index=False))
    q.to_csv(f"{args.prior_dir}/temporal_dynamics_stroke_quartile_corr.csv", index=False)
    print("\n[done]")


if __name__ == "__main__":
    main()
