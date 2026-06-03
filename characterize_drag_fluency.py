"""
Committee point #2 (characterisation): are the chunk-size and pacing accuracy
effects two faces of ONE drawing-fluency dimension?

The sensitivity analysis showed both effects are carried by drag behaviour:
accurate solvers draw in fewer, longer continuous strokes (high cells-per-stroke,
high drag fraction) rather than many discrete clicks. This script tests whether a
single fluency dimension accounts for both:

  (A) Spearman correlation matrix among the candidate measures + each vs accuracy.
  (B) Partial correlations: does chunk-size×accuracy survive controlling for
      drag-fluency?  does pacing×accuracy?  (If both collapse to ~0, one dimension.)
  (C) Rank-regression variance decomposition: unique vs shared R² of chunk size
      and pace; and whether a single fluency measure matches their joint R².
  (D) PCA on the sign-aligned rank-standardised measures: variance on PC1 and
      PC1 × accuracy.

Reads prior_analysis/drag_sensitivity_profile.csv and drag_trajectory_stats.csv.
"""

from __future__ import annotations

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def partial_spearman(x, y, z):
    """Spearman partial corr of x,y controlling for z (residualise ranks)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    rx_r = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ry_r = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = np.corrcoef(rx_r, ry_r)[0, 1]
    return float(r)


def rank_r2(y, Xcols):
    """R^2 of OLS on rank-transformed y ~ Xcols (list of arrays)."""
    ry = rankdata(y)
    X = np.column_stack([np.ones_like(ry)] + [rankdata(c) for c in Xcols])
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)
    pred = X @ beta
    ss_res = np.sum((ry - pred) ** 2)
    ss_tot = np.sum((ry - ry.mean()) ** 2)
    return 1 - ss_res / ss_tot


def main():
    prof = pd.read_csv("prior_analysis/drag_sensitivity_profile.csv")
    prof["subject"] = prof["subject"].astype(str)
    stats = pd.read_csv("prior_analysis/drag_trajectory_stats.csv")
    stats["cells_per_stroke"] = stats["n_edits"] / stats["n_strokes"].clip(lower=1)
    g = stats.groupby("subject_id").agg(
        drag_frac=("drag_frac", "mean"),
        cells_per_stroke=("cells_per_stroke", "mean"),
        strokes_per_traj=("n_strokes", "mean"),
    ).reset_index().rename(columns={"subject_id": "subject"})
    g["subject"] = g["subject"].astype(str)
    df = prof.merge(g, on="subject", how="inner")
    print(f"n subjects = {len(df)}\n")

    # measures: higher = more fluent/continuous, except rt_all (lower=faster)
    M = {
        "size_cells":       df["size_cells"].values,
        "pace_rt_all":      df["rt_all"].values,       # lower = faster
        "drag_frac":        df["drag_frac"].values,
        "cells_per_stroke": df["cells_per_stroke"].values,
        "strokes_per_traj": df["strokes_per_traj"].values,
    }
    acc = df["accuracy"].values

    # ── A: correlation matrix + vs accuracy ───────────────────────────────────
    print("[A] Spearman correlations")
    keys = list(M)
    print("    " + " ".join(f"{k[:9]:>10s}" for k in keys) + "   | accuracy")
    for k1 in keys:
        cells = []
        for k2 in keys:
            r = spearmanr(M[k1], M[k2]).statistic
            cells.append(f"{r:>10.2f}")
        ra = spearmanr(M[k1], acc)
        star = "*" if ra.pvalue < 0.05 else " "
        print(f"    {k1:16s}" + " ".join(cells) + f"   | {ra.statistic:+.3f}{star}")

    # ── B: partial correlations (control for fluency) ─────────────────────────
    print("\n[B] Does each effect survive controlling for drawing fluency?")
    for ctrl in ["cells_per_stroke", "drag_frac"]:
        ps = partial_spearman(M["size_cells"], acc, M[ctrl])
        pp = partial_spearman(M["pace_rt_all"], acc, M[ctrl])
        print(f"    controlling for {ctrl}:")
        print(f"        size_cells × accuracy : {spearmanr(M['size_cells'],acc).statistic:+.3f} → partial {ps:+.3f}")
        print(f"        pace(rt)   × accuracy : {spearmanr(M['pace_rt_all'],acc).statistic:+.3f} → partial {pp:+.3f}")

    # ── C: rank-regression variance decomposition ─────────────────────────────
    print("\n[C] Rank-regression R² for accuracy")
    r2_size = rank_r2(acc, [M["size_cells"]])
    r2_pace = rank_r2(acc, [M["pace_rt_all"]])
    r2_both = rank_r2(acc, [M["size_cells"], M["pace_rt_all"]])
    r2_frac = rank_r2(acc, [M["drag_frac"]])
    r2_cps = rank_r2(acc, [M["cells_per_stroke"]])
    print(f"    size only        R²={r2_size:.3f}")
    print(f"    pace only        R²={r2_pace:.3f}")
    print(f"    size + pace      R²={r2_both:.3f}  (unique size={r2_both-r2_pace:.3f}, "
          f"unique pace={r2_both-r2_size:.3f})")
    print(f"    drag_frac only   R²={r2_frac:.3f}   ← single fluency measure")
    print(f"    cells/stroke only R²={r2_cps:.3f}   ← single fluency measure")

    # ── D: PCA on sign-aligned rank-z measures ────────────────────────────────
    print("\n[D] PCA on rank-standardised fluency indicators "
          "(size_cells, −pace, drag_frac, cells_per_stroke)")
    cols = [M["size_cells"], -M["pace_rt_all"], M["drag_frac"], M["cells_per_stroke"]]
    Z = np.column_stack([(rankdata(c) - rankdata(c).mean()) / rankdata(c).std() for c in cols])
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var = S**2 / np.sum(S**2)
    print(f"    variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}, "
          f"PC3={var[2]:.1%}, PC4={var[3]:.1%}")
    pc1 = U[:, 0] * S[0]
    # sign-align PC1 so positive loads with drag_frac
    if spearmanr(pc1, M["drag_frac"]).statistic < 0:
        pc1 = -pc1
    print(f"    PC1 loadings (size,−pace,drag,cps): "
          f"{np.round(Vt[0] * (1 if spearmanr(U[:,0]*S[0], M['drag_frac']).statistic>0 else -1),2)}")
    rpc = spearmanr(pc1, acc)
    print(f"    PC1 (drawing-fluency factor) × accuracy: ρ={rpc.statistic:+.3f}, p={rpc.pvalue:.4f}")
    print(f"    → single fluency factor R² vs accuracy ≈ {rank_r2(acc,[pc1]):.3f}")


if __name__ == "__main__":
    main()
