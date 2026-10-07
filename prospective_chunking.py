"""Prospective test: does chunking measured EARLY predict accuracy on LATER, non-overlapping trials?

Motivation: the chapter's chunk-size/accuracy correlation is concurrent (both computed from the
same trials), so "predicts" is statistical rather than temporal. This uses non-overlapping windows
so the claim holds in the forecasting sense.

Trial order comes from the raw MTurk session JSONs (the session array is ordered). NOTE: the
`last_time` column in subject_task_sequence_measures_mturk.csv is NOT a session clock -- ranking
by it correlates ~-0.2 with true order. Always use the raw session order.

Usage: python prospective_chunking.py
"""
import json, glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

SUBJ = "/Users/carolineahn/Documents/GitHub/ARC-data/MTurk/mturk subject data"
MEAS = ("/Users/carolineahn/Documents/GitHub/CogARC-dataRepository/Behavioral data/"
        "subject_task_sequence_measures_mturk.csv")
STROKE = "prior_analysis/stroke_chunking_per_st.csv"
MIN_TRIALS = 40


def trial_order() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(SUBJ, "*.json")):
        sid = os.path.basename(f).replace(".json", "")
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for i, tr in enumerate(d.get("session", [])):
            task = None
            for att in tr.get("attempts", []):
                if isinstance(att, list):
                    for ev in att:
                        if ev.get("desc") == "new task":
                            task = ev.get("problem", "").replace(".json", "")
                            break
                if task:
                    break
            if task:
                rows.append({"subject": sid, "trial": task, "order": i + 1})
    return pd.DataFrame(rows)


def boot_ci(x, y, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            out.append(spearmanr(x[i], y[i])[0])
    return np.percentile(out, [2.5, 97.5])


def main():
    m = pd.read_csv(MEAS, dtype={"subject": str, "trial": str})
    st = (pd.read_csv(STROKE, dtype={"subject_id": str, "task_id": str})
            .rename(columns={"subject_id": "subject", "task_id": "trial"}))
    d = (m.merge(st[["subject", "trial", "cells_per_stroke", "n_strokes"]], on=["subject", "trial"])
           .merge(trial_order(), on=["subject", "trial"]))
    d["ok"] = (d.final_outcome.astype(str) == "success").astype(float)
    d["n_tr"] = d.groupby("subject")["trial"].transform("size")
    d = d[d.n_tr >= MIN_TRIALS]
    print(f"n subjects = {d.subject.nunique()}\n")

    for lbl, feat, cut in [("chunk size, trials 1-19 -> accuracy trials 20+", "cells_per_stroke", 19),
                           ("n strokes,  trials 1-19 -> accuracy trials 20+", "n_strokes", 19),
                           ("chunk size, 1st half     -> accuracy 2nd half", "cells_per_stroke", None)]:
        A, B = [], []
        for _, g in d.groupby("subject"):
            g = g.sort_values("order")
            if cut:
                e, l = g[g.order <= cut], g[g.order > cut]
            else:
                h = len(g) // 2
                e, l = g.iloc[:h], g.iloc[h:]
            if len(e) >= 4 and len(l) >= 4:
                A.append(e[feat].mean()); B.append(l["ok"].mean())
        r, p = spearmanr(A, B); lo, hi = boot_ci(A, B)
        print(f"{lbl:<48} n={len(A)}  rho={r:+.3f}  p={p:.4g}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    a = d.groupby("subject")["cells_per_stroke"].mean(); b = d.groupby("subject")["ok"].mean()
    r, p = spearmanr(a, b)
    print(f"\n{'concurrent benchmark (chapter reports +.24)':<48} n={len(a)}  rho={r:+.3f}  p={p:.4g}")


if __name__ == "__main__":
    main()
