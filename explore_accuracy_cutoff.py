"""
Exploratory: is the accuracy distribution bimodal, and where is the data-driven
cutoff that separates the two performance groups?  Diagnostics only — does not
modify the analysis pipeline.

    (A) Silverman's bootstrap test for the number of modes (H0: <= k modes).
    (B) GMM model selection: BIC/AIC for k = 1..4 (raw and logit-transformed).
    (C) Bootstrap stability of the data-driven cutoff (GMM crossover + Otsu).
    (D) Posterior membership uncertainty under the 2-component GMM.
    (E) Preview: Cliff's delta on chunk/pacing features for the data-driven
        high/low split vs the existing >=.95 elite split (do conclusions hold?).
"""

from __future__ import annotations

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


CHUNK_PACING = ["size", "n_cells", "n_chunks_total", "is_connected",
                "color_homogeneity", "success_iou_best",
                "mean_rt_median", "deliberation_time_median"]


# ── Silverman's test for number of modes ──────────────────────────────────────

def _count_modes(data: np.ndarray, h: float) -> int:
    lo, hi = data.min() - 3 * h, data.max() + 3 * h
    xs = np.linspace(lo, hi, 2001)
    dens = gaussian_kde(data, bw_method=h / np.std(data, ddof=1))(xs)
    return int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])))


def _critical_bw(data: np.ndarray, k: int,
                 hi: float = None, tol: float = 1e-4) -> float:
    """Smallest bandwidth giving <= k modes (bisection)."""
    lo = 1e-4
    hi = hi if hi is not None else (data.max() - data.min())
    # ensure hi gives <= k modes
    while _count_modes(data, hi) > k:
        hi *= 1.5
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _count_modes(data, mid) > k:
            lo = mid
        else:
            hi = mid
    return hi


def silverman_test(data: np.ndarray, k: int, n_boot: int = 500,
                   seed: int = 0) -> dict:
    n = len(data)
    s = np.std(data, ddof=1)
    h = _critical_bw(data, k)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        eps = rng.normal(0, h, n)
        xbar = data.mean()
        # variance-corrected smoothed bootstrap
        y = xbar + (data[idx] - xbar + eps) / np.sqrt(1 + h ** 2 / s ** 2)
        if _count_modes(y, h) > k:
            exceed += 1
    return dict(k=k, crit_bw=float(h), p_value=exceed / n_boot, n_boot=n_boot)


# ── helpers ────────────────────────────────────────────────────────────────────

def _gmm_crossover(a: np.ndarray, seed: int = 0) -> float:
    g = GaussianMixture(n_components=2, n_init=5, random_state=seed).fit(a.reshape(-1, 1))
    xs = np.linspace(a.min(), a.max(), 2001)
    hi = int(np.argmax(g.means_.ravel()))
    post = g.predict_proba(xs.reshape(-1, 1))[:, hi]
    cr = [xs[i] for i in range(1, len(xs)) if (post[i-1]-0.5)*(post[i]-0.5) < 0]
    return float(cr[0]) if cr else float("nan")


def _otsu(a: np.ndarray) -> float:
    n = len(a)
    best = (float("nan"), -1.0)
    for t in np.linspace(0.05, 0.95, 181):
        lo, hi = a[a < t], a[a >= t]
        if len(lo) < 2 or len(hi) < 2:
            continue
        bcv = (len(lo)/n)*(len(hi)/n)*(lo.mean()-hi.mean())**2
        if bcv > best[1]:
            best = (float(t), bcv)
    return best[0]


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = int(np.sum(np.subtract.outer(x, y) > 0))
    lt = int(np.sum(np.subtract.outer(x, y) < 0))
    return (gt - lt) / (len(x) * len(y))


def main():
    df = pd.read_csv("prior_analysis/chunking_top_solvers_merged.csv")
    a = df["accuracy"].dropna().values
    n = len(a)
    print(f"n = {n}\n")

    # ── A: Silverman ──────────────────────────────────────────────────────────
    print("[A] Silverman's bootstrap test for number of modes (H0: <= k modes)")
    for k in (1, 2):
        r = silverman_test(a, k, n_boot=500)
        verdict = "reject H0 (more modes)" if r["p_value"] < 0.05 else "cannot reject"
        print(f"    k={k}: crit_bw={r['crit_bw']:.4f}  p={r['p_value']:.3f}  -> {verdict}")

    # ── B: GMM model selection ─────────────────────────────────────────────────
    print("\n[B] GMM model selection (lower BIC = better)")
    X = a.reshape(-1, 1)
    for k in (1, 2, 3, 4):
        g = GaussianMixture(n_components=k, n_init=10, random_state=0).fit(X)
        print(f"    raw   k={k}: BIC={g.bic(X):8.1f}  AIC={g.aic(X):8.1f}  "
              f"means={np.round(np.sort(g.means_.ravel()),3)}")
    # logit transform (clip to avoid +/-inf at 0/1)
    ac = np.clip(a, 0.01, 0.99)
    L = np.log(ac/(1-ac)).reshape(-1, 1)
    print("    -- logit-transformed (bounded-data robustness) --")
    for k in (1, 2, 3):
        g = GaussianMixture(n_components=k, n_init=10, random_state=0).fit(L)
        print(f"    logit k={k}: BIC={g.bic(L):8.1f}  AIC={g.aic(L):8.1f}")

    # ── C: bootstrap stability of cutoff ───────────────────────────────────────
    print("\n[C] Bootstrap stability of the data-driven cutoff (1000 resamples)")
    rng = np.random.default_rng(1)
    gmm_cuts, otsu_cuts = [], []
    for b in range(1000):
        idx = rng.integers(0, n, n)
        ab = a[idx]
        try:
            gmm_cuts.append(_gmm_crossover(ab, seed=b))
        except Exception:
            pass
        otsu_cuts.append(_otsu(ab))
    gmm_cuts = np.array([c for c in gmm_cuts if np.isfinite(c)])
    otsu_cuts = np.array([c for c in otsu_cuts if np.isfinite(c)])
    for name, c in [("GMM crossover", gmm_cuts), ("Otsu", otsu_cuts)]:
        lo, med, hi = np.percentile(c, [2.5, 50, 97.5])
        print(f"    {name:14s}: median={med:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(c)})")

    # ── D: posterior membership uncertainty ─────────────────────────────────────
    print("\n[D] 2-component GMM posterior membership (point estimate on full data)")
    g2 = GaussianMixture(n_components=2, n_init=10, random_state=0).fit(X)
    hi = int(np.argmax(g2.means_.ravel()))
    p_hi = g2.predict_proba(X)[:, hi]
    cross = _gmm_crossover(a)
    ambiguous = np.sum((p_hi > 0.2) & (p_hi < 0.8))
    print(f"    crossover ~ {cross:.3f};  high-group n={int((a>=cross).sum())}, "
          f"low-group n={int((a<cross).sum())}")
    print(f"    ambiguous subjects (0.2 < P(high) < 0.8): {int(ambiguous)} "
          f"({ambiguous/n:.1%})")

    # ── E: preview feature comparisons under each split ─────────────────────────
    print("\n[E] Cliff's delta on features: data-driven (>=%.2f) vs elite (>=.95)" % cross)
    print(f"    {'feature':22s} {'delta_DD':>9s} {'delta_.95':>10s} {'same_dir':>9s}")
    for f in CHUNK_PACING:
        if f not in df.columns:
            continue
        v = df[[f, "accuracy"]].dropna()
        hi_dd = v.loc[v["accuracy"] >= cross, f].values
        lo_dd = v.loc[v["accuracy"] <  cross, f].values
        hi_95 = v.loc[v["accuracy"] >= 0.95, f].values
        lo_95 = v.loc[v["accuracy"] <  0.95, f].values
        d_dd = _cliffs_delta(hi_dd, lo_dd)
        d_95 = _cliffs_delta(hi_95, lo_95)
        same = "yes" if np.sign(d_dd) == np.sign(d_95) else "NO"
        print(f"    {f:22s} {d_dd:+9.3f} {d_95:+10.3f} {same:>9s}")


if __name__ == "__main__":
    main()
