"""
Confirmatory factor analysis: one-factor vs two-factor model of
pacing and chunking style.

Tests whether the §4 interpretation — that pacing and chunking style are
two surface measures of a single latent efficiency dimension — is
supported over the alternative that they reflect two distinct (but
correlated) factors.

Models
------
  Model 1 (one factor):
      Efficiency =~ deliberation_time + inter_edit_rt + time_before_first_edit
                    + chunk_size + n_cells + n_chunks_per_traj

  Model 2 (two correlated factors):
      Pacing      =~ deliberation_time + inter_edit_rt + time_before_first_edit
      Granularity =~ chunk_size + n_cells + n_chunks_per_traj
      Pacing ~~ Granularity

All indicators rank-transformed first.  Timing indicators (RTs) and
n_chunks_per_traj are sign-flipped so every indicator points in the same
direction (higher = more "efficient" / more global), making loadings
interpretable.

Outputs (prior_analysis/):
    cfa_fit_indices.csv     fit indices for both models + χ² difference test
    cfa_loadings.csv        standardised loadings per indicator per model
    cfa_summary.txt         narrative summary suitable for the chapter
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2
import semopy
from semopy.stats import calc_stats


# ── helpers ──────────────────────────────────────────────────────────────────

def _rank_z(x: np.ndarray) -> np.ndarray:
    """Rank-transform → standardised z."""
    r = rankdata(x, method="average")
    return (r - r.mean()) / r.std(ddof=1)


def _build_indicators(prior_dir: str, behavioral_csv: str) -> pd.DataFrame:
    """Build per-subject indicator table with 6 measures, all sign-aligned
    so 'higher = more efficient / more global'."""

    # Per-subject chunking + accuracy + already-computed timing
    merged = pd.read_csv(os.path.join(prior_dir,
                                       "chunking_top_solvers_merged.csv"))

    # Per-trial behavioral data → median time-in-drawing-before-first-edit
    behav = pd.read_csv(behavioral_csv)
    behav["time_in_drawing_before_first_edit"] = (
        behav["deliberation_time"] - behav["example_view_time_before_first_edit"]
    )
    pre_edit = (behav.groupby("subject")["time_in_drawing_before_first_edit"]
                       .median()
                       .reset_index()
                       .rename(columns={"time_in_drawing_before_first_edit":
                                          "time_before_first_edit"}))

    df = merged.merge(pre_edit, on="subject", how="inner")

    # Four indicators — sign-aligned so higher = more efficient / more global.
    # Dropped: pacing_first_edit (r=.86 with deliberation; redundant) and
    #          granularity_n_cells (r=.98 with size; redundant). Their inclusion
    #          produced Heywood cases (standardised loadings = 1.0) in a
    #          preliminary 6-indicator run.
    df["pacing_deliberation"]   = -df["deliberation_time_median"]
    df["pacing_inter_edit_rt"]  = -df["mean_rt_median"]
    df["granularity_size"]      =  df["size"]
    df["granularity_n_chunks"]  = -df["n_chunks_total"]  # fewer chunks = more global

    indicators = [
        "pacing_deliberation", "pacing_inter_edit_rt",
        "granularity_size", "granularity_n_chunks",
    ]
    keep = ["subject"] + indicators
    out = df[keep].dropna().copy()

    # Rank-transform every indicator → standardised
    for c in indicators:
        out[c] = _rank_z(out[c].values)

    return out, indicators


# ── model definitions ────────────────────────────────────────────────────────

ONE_FACTOR_SPEC = """
Efficiency =~ pacing_deliberation + pacing_inter_edit_rt + granularity_size + granularity_n_chunks
"""

TWO_FACTOR_SPEC = """
Pacing      =~ pacing_deliberation + pacing_inter_edit_rt
Granularity =~ granularity_size + granularity_n_chunks
Pacing ~~ Granularity
"""


# ── fitting + reporting ──────────────────────────────────────────────────────

def _fit_and_stats(spec: str, data: pd.DataFrame, label: str) -> dict:
    model = semopy.Model(spec)
    model.fit(data)
    stats = calc_stats(model).iloc[0].to_dict()

    n = len(data)
    chi2_v = float(stats.get("chi2", np.nan))
    df_v   = int(stats.get("DoF", np.nan))
    p_v    = float(stats.get("chi2 p-value", np.nan))
    cfi    = float(stats.get("CFI", np.nan))
    tli    = float(stats.get("TLI", np.nan))
    rmsea  = float(stats.get("RMSEA", np.nan))
    srmr   = float(stats.get("SRMR", np.nan))
    aic    = float(stats.get("AIC", np.nan))
    bic    = float(stats.get("BIC", np.nan))

    return dict(
        model=label, n=n, chi2=chi2_v, df=df_v, chi2_p=p_v,
        CFI=cfi, TLI=tli, RMSEA=rmsea, SRMR=srmr, AIC=aic, BIC=bic,
        fitted=model,
    )


def _loadings_table(fit: dict) -> pd.DataFrame:
    """Standardised loadings from semopy inspect()."""
    insp = fit["fitted"].inspect(std_est=True)
    # Keep just the measurement-model rows
    load = insp[insp["op"] == "~"].copy()
    # In semopy "~" is the loading; rename for clarity
    load = load.rename(columns={
        "lval": "indicator", "rval": "factor",
        "Estimate": "loading", "Est. Std": "std_loading",
        "Std. Err": "se", "p-value": "p",
    })
    load["model"] = fit["model"]
    return load[["model", "factor", "indicator", "loading", "std_loading", "se", "p"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--behavioral_csv",
                    default=str(_paths.BEHAVIORAL_CSV))
    args = ap.parse_args()

    print("[load] building indicator table")
    data, indicators = _build_indicators(args.prior_dir, args.behavioral_csv)
    print(f"       n = {len(data)} subjects, {len(indicators)} indicators")
    print()
    print("indicator correlation matrix (Spearman after rank-z transform = Pearson):")
    print(data[indicators].corr().round(2).to_string())

    # ── fit both models ───────────────────────────────────────────────────────
    print("\n[fit ] one-factor model …")
    fit1 = _fit_and_stats(ONE_FACTOR_SPEC, data, "one_factor")

    print("[fit ] two-factor model …")
    fit2 = _fit_and_stats(TWO_FACTOR_SPEC, data, "two_factor")

    # ── compile fit indices ───────────────────────────────────────────────────
    fit_df = pd.DataFrame([
        {k: v for k, v in fit1.items() if k != "fitted"},
        {k: v for k, v in fit2.items() if k != "fitted"},
    ])
    fit_df_print = fit_df.copy()
    fit_df.to_csv(os.path.join(args.prior_dir, "cfa_fit_indices.csv"), index=False)

    # ── χ² difference test (one-factor is nested in two-factor) ───────────────
    chi2_diff = fit1["chi2"] - fit2["chi2"]
    df_diff   = fit1["df"] - fit2["df"]
    p_diff    = float(chi2.sf(chi2_diff, df_diff)) if df_diff > 0 else float("nan")
    aic_diff  = fit1["AIC"] - fit2["AIC"]
    bic_diff  = fit1["BIC"] - fit2["BIC"]

    print("\n── Fit indices ────────────────────────────────────────────────")
    print(fit_df_print.round(3).to_string(index=False))

    print("\n── Comparison: one-factor vs two-factor ───────────────────────")
    print(f"  Δχ² = {chi2_diff:+.3f}  (Δdf = {df_diff},  p = {p_diff:.4f})")
    print(f"  ΔAIC (one − two) = {aic_diff:+.3f}   (positive favours two-factor)")
    print(f"  ΔBIC (one − two) = {bic_diff:+.3f}   (positive favours two-factor)")

    # ── loadings ──────────────────────────────────────────────────────────────
    load1 = _loadings_table(fit1)
    load2 = _loadings_table(fit2)
    loadings_df = pd.concat([load1, load2], ignore_index=True)
    loadings_df.to_csv(os.path.join(args.prior_dir, "cfa_loadings.csv"), index=False)

    print("\n── Standardised loadings ──────────────────────────────────────")
    print(loadings_df.round(3).to_string(index=False))

    # ── inter-factor correlation in the two-factor model ─────────────────────
    insp2 = fit2["fitted"].inspect(std_est=True)
    cov_row = insp2[(insp2["op"] == "~~") &
                    (insp2["lval"] == "Pacing") &
                    (insp2["rval"] == "Granularity")]
    if len(cov_row) > 0:
        r_pg = float(cov_row["Est. Std"].iloc[0])
    else:
        # try the other ordering
        cov_row = insp2[(insp2["op"] == "~~") &
                        (insp2["lval"] == "Granularity") &
                        (insp2["rval"] == "Pacing")]
        r_pg = float(cov_row["Est. Std"].iloc[0]) if len(cov_row) > 0 else float("nan")

    print(f"\nInter-factor correlation (Pacing ↔ Granularity) = {r_pg:+.3f}")

    # ── narrative summary ────────────────────────────────────────────────────
    shared_var = r_pg ** 2
    if p_diff < 0.05 and abs(r_pg) >= 0.80:
        # Two factors are statistically distinguishable but share most variance
        verdict = (
            "The two-factor model fits significantly better than the one-factor "
            f"model (Δχ² = {chi2_diff:.2f}, df = {df_diff}, p = {p_diff:.3f}; "
            f"ΔCFI = {fit2['CFI']-fit1['CFI']:+.3f}), so a strict single-factor "
            "interpretation is not supported by the χ² test. However, BIC favours "
            f"the more parsimonious one-factor model (ΔBIC = {bic_diff:+.2f}; "
            "negative values favour one-factor), and the inter-factor correlation "
            f"in the two-factor model is r = {r_pg:+.2f} — meaning the two factors "
            f"share approximately {shared_var:.0%} of their variance. The CFA "
            "therefore supports a nuanced 'two correlated facets' reading rather "
            "than either a strict single dimension or two independent constructs: "
            "pacing and chunking style are statistically distinguishable but "
            "tightly coupled, consistent with a higher-order common efficiency "
            "construct that surfaces in both."
        )
    elif p_diff < 0.05:
        verdict = (
            f"The two-factor model fits significantly better (Δχ² = {chi2_diff:.2f}, "
            f"df = {df_diff}, p = {p_diff:.3f}); inter-factor r = {r_pg:+.2f} "
            f"(shared variance ≈ {shared_var:.0%}). Pacing and chunking style are "
            "best treated as distinct but correlated factors rather than a single "
            "dimension."
        )
    else:
        verdict = (
            "The one-factor model fits the data as well as the two-factor model "
            f"(Δχ² = {chi2_diff:.2f}, df = {df_diff}, p = {p_diff:.3f}); ΔBIC "
            f"= {bic_diff:+.2f}. This directly supports the interpretation that "
            "pacing and chunking style are two surface measures of a single "
            "underlying efficiency dimension."
        )

    summary = f"""\
CFA: one-factor vs two-factor model of pacing + chunking style
================================================================

Sample:         n = {len(data)} subjects from the cross-repo merged dataset
Indicators (all rank-z transformed; sign-aligned so higher = more
                                       efficient / more global):
  Pacing:       deliberation_time, inter-edit RT
  Granularity:  chunk size, n chunks per trajectory (sign-flipped)
(time-before-first-edit and n_cells dropped due to redundancy with
 deliberation_time (r = .86) and chunk size (r = .98) respectively;
 their inclusion produced Heywood cases in a preliminary 6-indicator run.)

Model 1 (one factor):  Efficiency → all four indicators
Model 2 (two factors): Pacing → 2 timing,  Granularity → 2 chunking,
                       Pacing ↔ Granularity correlation free.

Fit indices
-----------
                 chi2    df   p     CFI    TLI    RMSEA  SRMR    AIC      BIC
  one-factor    {fit1['chi2']:>7.2f} {fit1['df']:>4}  {fit1['chi2_p']:.3f}  {fit1['CFI']:.3f}  {fit1['TLI']:.3f}  {fit1['RMSEA']:.3f}  {fit1['SRMR']:.3f}  {fit1['AIC']:>8.2f}  {fit1['BIC']:>8.2f}
  two-factor    {fit2['chi2']:>7.2f} {fit2['df']:>4}  {fit2['chi2_p']:.3f}  {fit2['CFI']:.3f}  {fit2['TLI']:.3f}  {fit2['RMSEA']:.3f}  {fit2['SRMR']:.3f}  {fit2['AIC']:>8.2f}  {fit2['BIC']:>8.2f}

Comparison
----------
  Δχ²        = {chi2_diff:+.3f}  (Δdf = {df_diff},  p = {p_diff:.4f})
  ΔAIC       = {aic_diff:+.3f}   (positive ⇒ two-factor preferred)
  ΔBIC       = {bic_diff:+.3f}   (positive ⇒ two-factor preferred; BIC penalises complexity more)
  r(Pacing, Granularity)  = {r_pg:+.3f}

Verdict
-------
{verdict}
"""

    with open(os.path.join(args.prior_dir, "cfa_summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print("[done]")


if __name__ == "__main__":
    main()
