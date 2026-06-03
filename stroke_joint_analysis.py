"""
§3.4 re-run with the motor chunking measure: joint prediction of accuracy from
chunk granularity (cells per stroke) and a DRAG-FREE pace measure (inter-click RT),
replacing the earlier joint analysis that used the drag-contaminated inter-edit RT.

  (1) Rank-regression variance decomposition: unique/shared R² of granularity and
      pace. (Expect granularity to carry the variance, pace ~0, little shared since
      they are nearly uncorrelated.)
  (2) Quadrant profile: median split on granularity × drag-free pace → mean accuracy.
  (3) CFA: do granularity and pacing form one factor or two? Indicators —
      Granularity: cells_per_stroke, (−)n_strokes; Pacing: (−)inter-click RT,
      (−)deliberation time. Compare 1- vs 2-factor; report the inter-factor
      correlation (the old RT-based CFA gave r = .84; with the drag-free measures it
      should be far lower if the constructs are genuinely separable).

Reads stroke_chunking_profile.csv, drag_sensitivity_profile.csv,
chunking_top_solvers_merged.csv. Outputs prior_analysis/stroke_joint_*.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def rank_r2(y, Xcols):
    ry = rankdata(y)
    X = np.column_stack([np.ones_like(ry)] + [rankdata(c) for c in Xcols])
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)
    pred = X @ beta
    return 1 - np.sum((ry - pred) ** 2) / np.sum((ry - ry.mean()) ** 2)


def main():
    sc = pd.read_csv("prior_analysis/stroke_chunking_profile.csv")[
        ["subject", "cells_per_stroke", "n_strokes"]]
    ds = pd.read_csv("prior_analysis/drag_sensitivity_profile.csv")[
        ["subject", "rt_click"]]
    mg = pd.read_csv("prior_analysis/chunking_top_solvers_merged.csv")[
        ["subject", "accuracy", "deliberation_time_median"]]
    for d in (sc, ds, mg):
        d["subject"] = d["subject"].astype(str)
    df = sc.merge(ds, on="subject").merge(mg, on="subject").dropna(
        subset=["cells_per_stroke", "rt_click", "deliberation_time_median", "accuracy"])
    acc = df["accuracy"].values
    print(f"n = {len(df)} subjects\n")

    gran = df["cells_per_stroke"].values        # chunking granularity
    pace = df["rt_click"].values                # drag-free pace (higher = slower)

    # ── (1) variance decomposition ────────────────────────────────────────────
    print("[1] Rank-regression R² for accuracy (granularity = cells/stroke; "
          "pace = drag-free inter-click RT)")
    r2_g = rank_r2(acc, [gran])
    r2_p = rank_r2(acc, [pace])
    r2_b = rank_r2(acc, [gran, pace])
    print(f"    granularity only : R² = {r2_g:.3f}")
    print(f"    pace only        : R² = {r2_p:.3f}")
    print(f"    joint            : R² = {r2_b:.3f}")
    print(f"      unique granularity = {r2_b - r2_p:.3f}")
    print(f"      unique pace        = {r2_b - r2_g:.3f}")
    print(f"      shared             = {r2_g + r2_p - r2_b:.3f}")
    print(f"    corr(granularity, pace) Spearman ρ = "
          f"{spearmanr(gran, pace).statistic:+.3f}  (≈0 ⇒ orthogonal predictors)\n")

    # ── (2) quadrant profile ──────────────────────────────────────────────────
    print("[2] Quadrant profile: median split on granularity × drag-free pace")
    gmed, pmed = np.median(gran), np.median(pace)
    df["_g"] = np.where(gran >= gmed, "coarse", "fine")
    df["_p"] = np.where(pace <= pmed, "fast", "slow")    # lower RT = faster
    rows = []
    for g in ("coarse", "fine"):
        for p in ("fast", "slow"):
            sub = df[(df["_g"] == g) & (df["_p"] == p)]
            rows.append(dict(granularity=g, pace=p, n=len(sub),
                             mean_accuracy=sub["accuracy"].mean()))
            print(f"    {g:6s} + {p:4s}: n={len(sub):3d}  "
                  f"mean accuracy = {sub['accuracy'].mean():.3f}")
    pd.DataFrame(rows).to_csv("prior_analysis/stroke_joint_quadrants.csv", index=False)
    print("    (accuracy should vary with granularity rows, not pace columns)\n")

    # ── (3) CFA: one factor vs two ────────────────────────────────────────────
    print("[3] CFA — granularity vs (drag-free) pacing: one factor or two?")
    z = lambda v: (rankdata(v) - rankdata(v).mean()) / rankdata(v).std()
    cfa = pd.DataFrame({
        "cps":   z(df["cells_per_stroke"]),
        "nstk":  z(-df["n_strokes"]),                 # higher = more global
        "iclk":  z(-df["rt_click"]),                  # higher = faster
        "delib": z(-df["deliberation_time_median"]),  # higher = faster
    })
    try:
        from semopy import Model
        m2 = Model("Granularity =~ cps + nstk\nPacing =~ iclk + delib\n"
                   "Granularity ~~ Pacing")
        m2.fit(cfa)
        ins2 = m2.inspect()
        st2 = m2.inspect(std_est=True)
        cov = ins2[(ins2["op"] == "~~") & (ins2["lval"] == "Granularity")
                   & (ins2["rval"] == "Pacing")]
        stats2 = m2.calc_stats() if hasattr(m2, "calc_stats") else None
        r_factors = float(st2[(st2["op"] == "~~") & (st2["lval"] == "Granularity")
                              & (st2["rval"] == "Pacing")]["Estimate"].iloc[0])
        print(f"    two-factor inter-factor correlation r(Granularity, Pacing) = "
              f"{r_factors:+.3f}")
        print("    (old RT-based CFA gave r = .84; a low value here = genuinely "
              "separable constructs)")
        try:
            from semopy import calc_stats
            s2 = calc_stats(m2).T
            print(f"    two-factor fit: CFI={float(s2.loc['CFI'][0]):.3f}, "
                  f"RMSEA={float(s2.loc['RMSEA'][0]):.3f}")
        except Exception as e:
            print(f"    (fit-stat calc skipped: {e})")
    except Exception as e:
        print(f"    semopy CFA failed ({e}); falling back to correlation table:")
        print(cfa.corr().round(2).to_string())

    print("\n[done]")


if __name__ == "__main__":
    main()
