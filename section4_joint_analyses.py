"""
Section 4: joint prediction and mediation analyses.

Uses per-subject profiles from chunking_top_solvers_merged.csv.
Pacing variable  : mean_rt_median (inter-edit RT, ms).
Chunking variable: size (mean chunk size, edits per chunk).
Outcome          : accuracy (proportion correct across tasks).

All regressions use RANK REGRESSION (OLS on rank-transformed variables),
which is the natural multivariate extension of Spearman correlation and avoids
distortion from the bounded, near-bimodal accuracy distribution and from the
non-linear size-accuracy relationship.  R² from rank regression ≈ ρ² from
Spearman in the bivariate case.

Three analyses:

    (A) Joint rank regression
        Model 1: rank(accuracy) ~ rank(size)
        Model 2: rank(accuracy) ~ rank(mean_rt_median)
        Model 3: rank(accuracy) ~ rank(size) + rank(mean_rt_median)
        Reports R², unique R² (semi-partial), standardised β, p-values.

    (B) Bootstrapped mediation (5 000 samples, bias-corrected 95 % CI)
        All variables rank-transformed before fitting paths.
        Path tested: RT (X) → chunk size (M) → accuracy (Y).

    (C) Joint top-performer profile
        Proportion of top solvers in each quadrant of the
        (chunk size, mean RT) space, using median splits.

Outputs (prior_analysis/):
    section4_regression.csv
    section4_mediation.csv
    section4_quadrant_profile.csv
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, norm
import statsmodels.api as sm


# ── helpers ───────────────────────────────────────────────────────────────────

def _rank(x: np.ndarray) -> np.ndarray:
    """Rank-transform, normalised to (0, 1) for comparable β coefficients."""
    r = rankdata(x, method="average")
    return (r - r.mean()) / r.std(ddof=1)


def _ols_fit(X_cols: list[np.ndarray], y: np.ndarray) -> "sm.regression.linear_model.RegressionResultsWrapper":
    X = sm.add_constant(np.column_stack(X_cols) if len(X_cols) > 1 else X_cols[0])
    return sm.OLS(y, X).fit()


def _ols_path(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Standardised slope and p-value from simple OLS."""
    res = _ols_fit([x], y)
    return float(res.params[1]), float(res.pvalues[1])


# ── A: joint rank regression ──────────────────────────────────────────────────

def joint_regression(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[["size", "mean_rt_median", "accuracy"]].notna().all(axis=1)
    sub = df[ok].copy()
    n = len(sub)

    sz = _rank(sub["size"].values)
    rt = _rank(sub["mean_rt_median"].values)
    y  = _rank(sub["accuracy"].values)

    m1 = _ols_fit([sz], y)
    m2 = _ols_fit([rt], y)
    m3 = _ols_fit([sz, rt], y)

    r2_size  = float(m1.rsquared)
    r2_rt    = float(m2.rsquared)
    r2_joint = float(m3.rsquared)

    unique_size = r2_joint - r2_rt
    unique_rt   = r2_joint - r2_size
    shared      = r2_size + r2_rt - r2_joint

    rows = [
        {"model": "size_only",   "R2": r2_size,    "beta_size": m1.params[1],  "p_size": m1.pvalues[1],  "beta_rt": np.nan,      "p_rt": np.nan,       "n": n},
        {"model": "rt_only",     "R2": r2_rt,      "beta_size": np.nan,        "p_size": np.nan,          "beta_rt": m2.params[1], "p_rt": m2.pvalues[1], "n": n},
        {"model": "joint",       "R2": r2_joint,   "beta_size": m3.params[1],  "p_size": m3.pvalues[1],  "beta_rt": m3.params[2], "p_rt": m3.pvalues[2], "n": n},
        {"model": "unique_size", "R2": unique_size, "beta_size": np.nan,       "p_size": np.nan,          "beta_rt": np.nan,       "p_rt": np.nan,        "n": n},
        {"model": "unique_rt",   "R2": unique_rt,   "beta_size": np.nan,       "p_size": np.nan,          "beta_rt": np.nan,       "p_rt": np.nan,        "n": n},
        {"model": "shared",      "R2": shared,      "beta_size": np.nan,       "p_size": np.nan,          "beta_rt": np.nan,       "p_rt": np.nan,        "n": n},
    ]
    return pd.DataFrame(rows)


# ── B: bootstrapped mediation (rank-based) ────────────────────────────────────

def _mediation_paths(x: np.ndarray, m: np.ndarray,
                     y: np.ndarray) -> tuple[float, float, float, float]:
    a, _ = _ols_path(x, m)
    c, _ = _ols_path(x, y)
    res  = _ols_fit([x, m], y)
    c_prime = float(res.params[1])
    b       = float(res.params[2])
    return a, b, c, c_prime


def bootstrapped_mediation(df: pd.DataFrame, x_col: str, m_col: str,
                           y_col: str, label: str,
                           n_boot: int = 5000, seed: int = 42) -> dict:
    ok  = df[[x_col, m_col, y_col]].notna().all(axis=1)
    sub = df[ok]
    n   = len(sub)

    xs = _rank(sub[x_col].values)
    ms = _rank(sub[m_col].values)
    ys = _rank(sub[y_col].values)

    a, b, c, c_prime = _mediation_paths(xs, ms, ys)
    indirect = a * b

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # re-rank within bootstrap sample for consistency
        xi = _rank(sub[x_col].values[idx])
        mi = _rank(sub[m_col].values[idx])
        yi = _rank(sub[y_col].values[idx])
        ai, bi, _, _ = _mediation_paths(xi, mi, yi)
        boot[i] = ai * bi

    # Bias-corrected CI
    z0_val = norm.ppf(np.clip(np.mean(boot < indirect), 1e-6, 1 - 1e-6))
    ci_lo = float(np.percentile(boot, norm.cdf(2 * z0_val - 1.96) * 100))
    ci_hi = float(np.percentile(boot, norm.cdf(2 * z0_val + 1.96) * 100))

    return dict(
        label=label, x=x_col, m=m_col, y=y_col, n=n, n_boot=n_boot,
        a=round(a, 4), b=round(b, 4),
        c=round(c, 4), c_prime=round(c_prime, 4),
        indirect=round(indirect, 4),
        ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
        prop_mediated=round(abs(indirect / c) if abs(c) > 1e-6 else float("nan"), 3),
        significant=(ci_lo > 0 or ci_hi < 0),
    )


# ── C: quadrant profile ───────────────────────────────────────────────────────

def quadrant_profile(df: pd.DataFrame) -> pd.DataFrame:
    ok  = df[["size", "mean_rt_median", "top_solver", "accuracy"]].notna().all(axis=1)
    sub = df[ok].copy()
    sz_med = sub["size"].median()
    rt_med = sub["mean_rt_median"].median()
    sub["quadrant"] = np.select(
        [
            (sub["size"] >= sz_med) & (sub["mean_rt_median"] <  rt_med),
            (sub["size"] >= sz_med) & (sub["mean_rt_median"] >= rt_med),
            (sub["size"] <  sz_med) & (sub["mean_rt_median"] <  rt_med),
            (sub["size"] <  sz_med) & (sub["mean_rt_median"] >= rt_med),
        ],
        ["global_fast", "global_slow", "local_fast", "local_slow"],
    )
    return (
        sub.groupby("quadrant")
        .agg(n=("top_solver","count"), n_top=("top_solver","sum"),
             pct_top=("top_solver","mean"), mean_accuracy=("accuracy","mean"))
        .reset_index()
        .sort_values("mean_accuracy", ascending=False)
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.prior_dir, "chunking_top_solvers_merged.csv"))
    print(f"[load] {len(df)} subjects")

    print("\n[A] Joint rank regression  (outcome = rank(accuracy))")
    reg = joint_regression(df)
    print(reg.round(4).to_string(index=False))
    reg.to_csv(os.path.join(args.prior_dir, "section4_regression.csv"), index=False)

    print(f"\n[B] Bootstrapped rank mediation ({args.n_boot} samples)")
    med = bootstrapped_mediation(
        df, x_col="mean_rt_median", m_col="size", y_col="accuracy",
        label="RT → chunk_size → accuracy", n_boot=args.n_boot
    )
    print(f"  a (RT→size)        = {med['a']:+.4f}")
    print(f"  b (size→acc|RT)    = {med['b']:+.4f}")
    print(f"  indirect (a×b)     = {med['indirect']:+.4f}  "
          f"95%BC-CI [{med['ci_lo']:+.4f}, {med['ci_hi']:+.4f}]  sig={med['significant']}")
    print(f"  total c            = {med['c']:+.4f}")
    print(f"  direct c'          = {med['c_prime']:+.4f}")
    print(f"  prop mediated      = {med['prop_mediated']:.1%}")
    pd.DataFrame([med]).to_csv(
        os.path.join(args.prior_dir, "section4_mediation.csv"), index=False)

    print("\n[C] Quadrant profile (median splits on size and RT)")
    quad = quadrant_profile(df)
    print(quad.to_string(index=False))
    quad.to_csv(os.path.join(args.prior_dir, "section4_quadrant_profile.csv"), index=False)

    print("\n[done]")


if __name__ == "__main__":
    main()
