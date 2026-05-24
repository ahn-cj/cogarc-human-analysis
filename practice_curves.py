"""
Power-law vs exponential practice curves.

Fits Newell & Rosenbloom-style power-law (log(RT) = a + b·log(t)) and
the Heathcote, Brown & Mewhort (2000) exponential alternative
(log(RT) = a + b·t) to each subject's trial-by-trial timing data, then
compares which functional form fits better at the individual level.
Also fits both models to the trial-by-trial group means to test
Heathcote's claim that group aggregation creates apparent power-law
behaviour even when individuals are exponential.

Three analyses
--------------
    (A)  Per-subject fits of both models for RT and deliberation_time.
         Report per-subject β estimates and RMSE on the raw scale.
    (B)  Per-subject "model winner" tallies (which model has lower RMSE
         on raw RT?) and median ΔAIC.
    (C)  Group-aggregated fit: compute mean RT per trial-order across
         subjects, fit both models, compare to per-subject conclusion.
    (D)  Individual learning rate × accuracy / top-solver status.

Outputs (prior_analysis/):
    practice_curves_per_subject.csv   per-subject coefficients + RMSEs
    practice_curves_group.csv         group-aggregated fits
    practice_curves_winners.csv       per-measure model-winner tallies
    practice_curves_summary.txt       narrative summary
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import statsmodels.api as sm


BEHAV_CSV = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/"
             "Behavioral data/behavioral_measures_filtered.csv")

MEASURES = [
    ("rt",                "inter-edit RT (s)"),
    ("deliberation_time", "deliberation time (s)"),
]

TOP_SOLVER_ACC = 0.95
MIN_TASKS_TS   = 40
MIN_TRIALS_FIT = 30   # need at least 30 valid trials to fit a per-subject curve


# ── helpers ───────────────────────────────────────────────────────────────────

def _aic(rss: float, n: int, k: int) -> float:
    """AIC for a Gaussian model: n*log(rss/n) + 2*k."""
    if rss <= 0 or n <= 0:
        return float("nan")
    return n * np.log(rss / n) + 2 * k


def _fit_power_law(t: np.ndarray, y: np.ndarray) -> dict:
    """log(y) = a + b*log(t).   y is the raw RT, t is trial order.
    Returns coefficients + RMSE on the raw-y scale."""
    ok = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0)
    t, y = t[ok], y[ok]
    if len(t) < 5:
        return None
    X = sm.add_constant(np.log(t))
    res = sm.OLS(np.log(y), X).fit()
    a_log = float(res.params[0])
    b     = float(res.params[1])   # exponent in y = exp(a)*t^b; expect b < 0
    y_pred = np.exp(a_log + b * np.log(t))
    rss = float(np.sum((y - y_pred) ** 2))
    return dict(model="power_law", a_log=a_log, b=b,
                rmse=float(np.sqrt(rss / len(y))),
                rss=rss, n=len(y), aic=_aic(rss, len(y), 2),
                r2_logspace=float(res.rsquared))


def _fit_exponential(t: np.ndarray, y: np.ndarray) -> dict:
    """log(y) = a + b*t.    Heathcote-style exponential learning."""
    ok = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    if len(t) < 5:
        return None
    X = sm.add_constant(t)
    res = sm.OLS(np.log(y), X).fit()
    a_log = float(res.params[0])
    b     = float(res.params[1])   # expect b < 0 (RT decreases with practice)
    y_pred = np.exp(a_log + b * t)
    rss = float(np.sum((y - y_pred) ** 2))
    return dict(model="exponential", a_log=a_log, b=b,
                rmse=float(np.sqrt(rss / len(y))),
                rss=rss, n=len(y), aic=_aic(rss, len(y), 2),
                r2_logspace=float(res.rsquared))


def _attach_subject_attrs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["correct"] = df["final_result"].astype(str).str.lower() == "success"
    subj = (df.groupby("subject")
              .agg(n_tasks=("trial","nunique"), accuracy=("correct","mean"))
              .reset_index())
    subj["top_solver"] = (subj["accuracy"] >= TOP_SOLVER_ACC) & (subj["n_tasks"] >= MIN_TASKS_TS)
    return df.merge(subj, on="subject", how="left")


# ── main per-subject loop ─────────────────────────────────────────────────────

def per_subject_fits(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("subject"):
        sub = g.dropna(subset=[measure, "order"])
        if len(sub) < MIN_TRIALS_FIT:
            continue
        t = sub["order"].values.astype(float)
        y = sub[measure].values.astype(float)

        pl  = _fit_power_law(t, y)
        exp = _fit_exponential(t, y)
        if pl is None or exp is None:
            continue

        winner = "exponential" if exp["rmse"] < pl["rmse"] else "power_law"
        rows.append({
            "subject": sid, "measure": measure, "n_trials": int(len(sub)),
            "top_solver": bool(sub["top_solver"].iloc[0]),
            "accuracy":   float(sub["accuracy"].iloc[0]),
            "pl_b":   pl["b"],   "pl_rmse":   pl["rmse"],
            "pl_aic": pl["aic"], "pl_r2_log": pl["r2_logspace"],
            "exp_b":   exp["b"],  "exp_rmse":  exp["rmse"],
            "exp_aic": exp["aic"],"exp_r2_log":exp["r2_logspace"],
            "delta_aic_pl_minus_exp": pl["aic"] - exp["aic"],
            "winner": winner,
        })
    return pd.DataFrame(rows)


def group_aggregated_fits(df: pd.DataFrame, measure: str) -> dict:
    sub = df.dropna(subset=[measure, "order"]).copy()
    sub = sub[sub["n_tasks"] >= MIN_TASKS_TS]
    mean_per_order = (sub.groupby("order")[measure].mean().reset_index()
                          .rename(columns={measure: "mean"}))
    t = mean_per_order["order"].values.astype(float)
    y = mean_per_order["mean"].values.astype(float)
    pl  = _fit_power_law(t, y)
    exp = _fit_exponential(t, y)
    return dict(measure=measure, t=t, mean=y, pl=pl, exp=exp)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--csv", default=BEHAV_CSV)
    args = ap.parse_args()

    print(f"[load] {args.csv}")
    df = _attach_subject_attrs(pd.read_csv(args.csv))
    n_subj = df["subject"].nunique()
    print(f"       {len(df):,} trials, {n_subj} subjects")

    # ── per-subject fits ──────────────────────────────────────────────────────
    print("\n[A] per-subject fits (≥30 trials per subject)")
    all_per_subj = []
    winner_rows  = []
    for measure, label in MEASURES:
        per_subj = per_subject_fits(df, measure)
        all_per_subj.append(per_subj)
        n_fit = len(per_subj)
        n_exp_wins = (per_subj["winner"] == "exponential").sum()
        n_pl_wins  = (per_subj["winner"] == "power_law").sum()
        median_d_aic = float(per_subj["delta_aic_pl_minus_exp"].median())
        print(f"    {label}: {n_fit} subjects fit")
        print(f"      exponential wins (lower RMSE): {n_exp_wins} ({n_exp_wins/n_fit:.0%})")
        print(f"      power-law wins:                {n_pl_wins} ({n_pl_wins/n_fit:.0%})")
        print(f"      median ΔAIC (power − exp):     {median_d_aic:+.2f}   "
              f"(positive ⇒ exponential preferred)")
        # Median per-subject learning rates
        print(f"      median per-subject β:  power = {per_subj['pl_b'].median():+.3f}   "
              f"exp = {per_subj['exp_b'].median():+.4f}")
        winner_rows.append({
            "measure": measure, "n_subjects_fit": n_fit,
            "n_exp_wins": int(n_exp_wins), "n_pl_wins": int(n_pl_wins),
            "pct_exp_wins": float(n_exp_wins / n_fit),
            "median_delta_aic_pl_minus_exp": median_d_aic,
            "median_pl_b": float(per_subj["pl_b"].median()),
            "median_exp_b": float(per_subj["exp_b"].median()),
        })

    per_subj_df = pd.concat(all_per_subj, ignore_index=True)
    per_subj_df.to_csv(os.path.join(args.prior_dir,
                                     "practice_curves_per_subject.csv"), index=False)
    pd.DataFrame(winner_rows).to_csv(
        os.path.join(args.prior_dir, "practice_curves_winners.csv"), index=False)

    # ── group-aggregated fits ─────────────────────────────────────────────────
    print("\n[B] group-aggregated fits (mean per trial-order)")
    group_results = []
    for measure, label in MEASURES:
        gres = group_aggregated_fits(df, measure)
        pl, exp = gres["pl"], gres["exp"]
        win = "exponential" if exp["rmse"] < pl["rmse"] else "power_law"
        print(f"    {label}:")
        print(f"      power-law:    b = {pl['b']:+.4f}, R²(log) = {pl['r2_logspace']:.3f}, RMSE = {pl['rmse']:.3f}")
        print(f"      exponential:  b = {exp['b']:+.4f}, R²(log) = {exp['r2_logspace']:.3f}, RMSE = {exp['rmse']:.3f}")
        print(f"      winner on raw-scale RMSE: {win}    (ΔAIC pl-exp = {pl['aic']-exp['aic']:+.2f})")
        group_results.append({
            "measure": measure,
            "pl_b": pl["b"], "pl_r2_log": pl["r2_logspace"],
            "pl_rmse": pl["rmse"], "pl_aic": pl["aic"],
            "exp_b": exp["b"], "exp_r2_log": exp["r2_logspace"],
            "exp_rmse": exp["rmse"], "exp_aic": exp["aic"],
            "winner_aggregated": win,
        })
    pd.DataFrame(group_results).to_csv(
        os.path.join(args.prior_dir, "practice_curves_group.csv"), index=False)

    # ── individual learning rate × accuracy / top-solver ──────────────────────
    print("\n[C] individual learning rate × accuracy / top-solver")
    diagnostic_rows = []
    for measure, label in MEASURES:
        sub = per_subj_df[per_subj_df["measure"] == measure]
        # Exponential β as the learning rate (more negative = faster learning)
        rho_acc, p_acc = spearmanr(sub["exp_b"], sub["accuracy"])
        top  = sub.loc[sub["top_solver"], "exp_b"].values
        rest = sub.loc[~sub["top_solver"], "exp_b"].values
        _, p_mw = mannwhitneyu(top, rest, alternative="two-sided")
        diff = float(np.median(top) - np.median(rest))
        print(f"    {label}:")
        print(f"      Spearman(exp learning rate β, accuracy) = {rho_acc:+.3f} (p = {p_acc:.4f})")
        print(f"      Top median β = {np.median(top):+.4f}   rest median β = {np.median(rest):+.4f}   "
              f"top − rest = {diff:+.4f}   (p = {p_mw:.4f})")
        diagnostic_rows.append({
            "measure": measure,
            "spearman_b_acc": float(rho_acc), "p_spearman": float(p_acc),
            "top_median_b": float(np.median(top)),
            "rest_median_b": float(np.median(rest)),
            "top_minus_rest_b": diff, "p_mannwhitney": float(p_mw),
        })

    # ── narrative summary ─────────────────────────────────────────────────────
    summary_lines = [
        "Practice-curve fits: power law vs exponential",
        "=" * 60,
        "",
        f"Sample: {n_subj} subjects, ≥{MIN_TRIALS_FIT}-trial fit threshold.",
        "",
    ]
    for measure, label in MEASURES:
        per_subj = per_subj_df[per_subj_df["measure"] == measure]
        wr = next(w for w in winner_rows if w["measure"] == measure)
        gr = next(g for g in group_results if g["measure"] == measure)
        dr = next(d for d in diagnostic_rows if d["measure"] == measure)
        summary_lines.extend([
            f"--- {label} ---",
            f"  Per-subject (n = {len(per_subj)}):",
            f"    exponential wins: {wr['n_exp_wins']} ({wr['pct_exp_wins']:.0%})",
            f"    power-law wins:   {wr['n_pl_wins']} ({1 - wr['pct_exp_wins']:.0%})",
            f"    median ΔAIC (pl − exp) = {wr['median_delta_aic_pl_minus_exp']:+.2f}",
            f"",
            f"  Group-aggregated mean per trial-order:",
            f"    power-law   b = {gr['pl_b']:+.4f}  R²(log) = {gr['pl_r2_log']:.3f}",
            f"    exponential b = {gr['exp_b']:+.4f}  R²(log) = {gr['exp_r2_log']:.3f}",
            f"    winner: {gr['winner_aggregated']}",
            f"",
            f"  Individual learning rate (exponential β) × outcomes:",
            f"    Spearman(β, accuracy) = {dr['spearman_b_acc']:+.3f}, p = {dr['p_spearman']:.4f}",
            f"    Top vs rest median β: {dr['top_median_b']:+.4f} vs {dr['rest_median_b']:+.4f}, p = {dr['p_mannwhitney']:.4f}",
            "",
        ])

    summary = "\n".join(summary_lines)
    with open(os.path.join(args.prior_dir, "practice_curves_summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print("[done]")


if __name__ == "__main__":
    main()
