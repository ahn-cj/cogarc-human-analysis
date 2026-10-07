"""Graph algorithms on the CogARC knowledge graph.

Division of labour: Kuzu does the traversal and projection (Cypher), NetworkX
does the algorithms. The graph database is not a graph-algorithms library.

Two analyses that survive the design:
  (A) keyword communities from co-selection, vs the published K-means clusters
  (B) problem-problem similarity, two ways -- keyword profile vs shared errors --
      each normalised against the assignment schedule

NOT run here, deliberately: participant or problem centrality. Participant degree
is exactly 9 for all 184 by design, and problem degree spans 92x because of how
problems were assigned (problem 00 to all 184, problem 03 to 2). Centrality on
those is a measurement of the experiment's schedule, not of the data.
"""
import os, math, itertools, collections
import kuzu, numpy as np, pandas as pd, networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.metrics import adjusted_rand_score

import paths
HERE = os.path.dirname(os.path.abspath(__file__))
DB = str(paths.DB)
KMEANS = str(paths.KMEANS_LABELS)
paths.require(paths.KMEANS_LABELS)
conn = kuzu.Connection(kuzu.Database(DB))
q = lambda s: conn.execute(s).get_as_df()

# ── (A) keyword co-selection -> NPMI -> Louvain ──────────────────────────────
def keyword_communities():
    rep = q("""MATCH (kr:KeywordReport)-[:ABOUT]->(p:Problem),
                     (kr)-[:SELECTED]->(k:Keyword)
               WHERE p.is_attention_check = false
               RETURN kr.id AS report, k.name AS keyword""")
    sets = rep.groupby("report")["keyword"].apply(set)
    N = len(sets)
    uni = collections.Counter(k for s in sets for k in s)
    joint = collections.Counter()
    for s in sets:
        for a, b in itertools.combinations(sorted(s), 2):
            joint[(a, b)] += 1

    # NPMI rather than raw co-occurrence: a raw count mostly measures base rate.
    # Color appears in 912 reports, so it co-occurs with everything.
    G = nx.Graph()
    G.add_nodes_from(uni)
    rows = []
    for (a, b), n_ab in joint.items():
        p_ab, p_a, p_b = n_ab / N, uni[a] / N, uni[b] / N
        npmi = math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)
        rows.append(dict(a=a, b=b, n=n_ab, npmi=round(npmi, 3)))
        if npmi > 0:
            G.add_edge(a, b, weight=npmi)
    pairs = pd.DataFrame(rows).sort_values("npmi", ascending=False)

    comms = louvain_communities(G, weight="weight", seed=7, resolution=1.0)
    lab = {k: i for i, c in enumerate(comms) for k in c}
    km = pd.read_csv(KMEANS).set_index("keyword")["keyword_cluster"].to_dict()
    shared = sorted(set(lab) & set(km))
    ari = adjusted_rand_score([lab[k] for k in shared], [km[k] for k in shared])

    print("=" * 72)
    print("(A) KEYWORD COMMUNITIES — Louvain on NPMI-weighted co-selection")
    print("=" * 72)
    print(f"graph: {G.number_of_nodes()} keywords, {G.number_of_edges()} positive-NPMI edges "
          f"of {len(joint)} pairs ({N} reports)\n")
    for i, c in enumerate(comms):
        print(f"  community {i}: {', '.join(sorted(c))}")
    print(f"\n  published K-means (K=6) partition, for comparison:")
    for cl in sorted(set(km.values())):
        print(f"  cluster {cl}: {', '.join(sorted(k for k, v in km.items() if v == cl))}")
    print(f"\n  adjusted Rand index (Louvain vs K-means) = {ari:+.3f}")
    print("  strongest and weakest keyword pairings by NPMI:")
    for _, r in pd.concat([pairs.head(5), pairs.tail(4)]).iterrows():
        print(f"    {r.a:<10} {r.b:<10} n={int(r.n):<4} NPMI={r.npmi:+.3f}")
    pairs.to_csv(os.path.join(HERE, "..", "queries", "03_keyword_npmi.csv"), index=False)
    return lab, ari

# ── (B) problem-problem similarity, two projections ──────────────────────────
def problem_similarity():
    prof = q("""MATCH (kr:KeywordReport)-[:ABOUT]->(p:Problem),
                      (kr)-[:SELECTED]->(k:Keyword)
                WHERE p.is_attention_check = false
                RETURN p.kw_index AS problem, k.name AS keyword,
                       count(*) AS n""")
    M = prof.pivot_table(index="problem", columns="keyword", values="n",
                         fill_value=0).astype(float)
    M = M.div(M.sum(axis=1), axis=0)                       # within-problem profile
    Z = M.values / np.linalg.norm(M.values, axis=1, keepdims=True)
    kw_sim = pd.DataFrame(Z @ Z.T, index=M.index, columns=M.index)

    # shared-error co-occurrence, normalised by co-assignment
    err = q("""MATCH (pa:Participant)-[:MADE]->(s:Submission)-[:ON_PROBLEM]->(p:Problem),
                     (s)-[:LANDED_IN]->(g:SolutionGroup)
               WHERE g.is_correct = false AND g.n_submitters >= 3
                 AND p.is_attention_check = false
               RETURN pa.id AS participant, p.kw_index AS problem""")
    allp = q("""MATCH (pa:Participant)-[:MADE]->(s:Submission)-[:ON_PROBLEM]->(p:Problem)
                WHERE p.is_attention_check = false
                RETURN pa.id AS participant, p.kw_index AS problem""")
    err_sets = err.groupby("participant")["problem"].apply(set)
    all_sets = allp.groupby("participant")["problem"].apply(set)
    obs, co_assigned = collections.Counter(), collections.Counter()
    for s in err_sets:
        for a, b in itertools.combinations(sorted(s), 2): obs[(a, b)] += 1
    for s in all_sets:
        for a, b in itertools.combinations(sorted(s), 2): co_assigned[(a, b)] += 1
    # problems where nobody made a shared error divide to NaN, not 0 -- fill them,
    # or they silently produce NaN expected counts downstream.
    denom = allp.groupby("problem").participant.nunique()
    err_rate = (err.groupby("problem").participant.nunique()
                .reindex(denom.index).fillna(0) / denom)

    rows = []
    for pair, n_both in co_assigned.items():
        a, b = pair
        if n_both < 10 or a not in err_rate.index or b not in err_rate.index: continue
        exp = n_both * err_rate[a] * err_rate[b]
        if not (exp > 0): continue
        rows.append(dict(a=a, b=b, n_co_assigned=n_both, observed=obs.get(pair, 0),
                         expected=round(exp, 2), ratio=round(obs.get(pair, 0) / exp, 2),
                         kw_profile_cos=round(float(kw_sim.loc[a, b]), 3)))
    R = pd.DataFrame(rows)

    print()
    print("=" * 72)
    print("(B) PROBLEM-PROBLEM SIMILARITY — two projections")
    print("=" * 72)
    print(f"{len(R)} problem pairs with >=10 co-assigned participants\n")
    print("  most similar problem pairs by KEYWORD PROFILE (cosine):")
    for _, r in R.nlargest(5, "kw_profile_cos").iterrows():
        print(f"    {r.a}-{r.b}  cos={r.kw_profile_cos:.3f}   shared errors obs={r.observed} exp={r.expected}")
    print("\n  problem pairs where SHARED ERRORS co-occur above chance:")
    for _, r in R.nlargest(5, "ratio").iterrows():
        print(f"    {r.a}-{r.b}  obs={r.observed} exp={r.expected} ratio={r.ratio:.2f}  "
              f"(keyword cos={r.kw_profile_cos:.3f})")
    ok = R[R.expected >= 1]
    if len(ok) > 3:
        rho = ok[["ratio", "kw_profile_cos"]].corr(method="spearman").iloc[0, 1]
        print(f"\n  do the two projections agree?  Spearman rho = {rho:+.3f} "
              f"over {len(ok)} pairs with expected >= 1")
        print("  (near zero = conceptual similarity and shared-error structure are")
        print("   different organisations of the same problems -- the interesting case)")
    print(f"\n  power note: only {int((err_sets.apply(len) >= 2).sum())} participants made a shared "
          f"error on 2+ problems, so (B) is descriptive, not a test.")
    R.to_csv(os.path.join(HERE, "..", "queries", "04_problem_similarity.csv"), index=False)

if __name__ == "__main__":
    keyword_communities()
    problem_similarity()
