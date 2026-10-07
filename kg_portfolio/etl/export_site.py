"""Export the graph to a compact JSON bundle for the explorer.

An answer group is shown when two or more people drew exactly that grid, or when
it is the correct answer. The submitter count ships with every group, so the
viewer can judge weight; the threshold is presentational, not analytic.

Grids ship as one character per cell ("0014400") rather than nested arrays,
which keeps the whole of 40 puzzles plus every answer group well under 300 KB.
"""
import os, json, math, itertools, collections
import kuzu, pandas as pd, networkx as nx
from networkx.algorithms.community import louvain_communities

import paths
HERE = os.path.dirname(os.path.abspath(__file__))
# The site lives in its own repo so it can be deployed on its own. Export writes
# there; that repo's src/build.py turns it into index.html.
OUT = str(paths.SITE_DATA)
KWT = str(paths.KEYWORD_TESTSET)
paths.require(paths.KEYWORD_TESTSET, paths.SITE)
conn = kuzu.Connection(kuzu.Database(str(paths.DB)))
q = lambda s: conn.execute(s).get_as_df()
os.makedirs(OUT, exist_ok=True)
enc = lambda g: ["".join(str(c) for c in row) for row in g]

B = {}

# ── keyword layer ───────────────────────────────────────────────────────────
rep = q("""MATCH (kr:KeywordReport)-[:ABOUT]->(p:Problem), (kr)-[:SELECTED]->(k:Keyword)
           WHERE p.is_attention_check = false
           RETURN kr.id AS report, k.name AS keyword""")
sets = rep.groupby("report")["keyword"].apply(set); N = len(sets)
uni = collections.Counter(k for s in sets for k in s)
joint = collections.Counter()
for s in sets:
    for a, b in itertools.combinations(sorted(s), 2): joint[(a, b)] += 1
G = nx.Graph(); G.add_nodes_from(uni); links = []
for (a, b), n in joint.items():
    npmi = math.log((n/N)/((uni[a]/N)*(uni[b]/N))) / -math.log(n/N)
    links.append(dict(s=a, t=b, npmi=round(npmi, 3), n=n))
    if npmi > 0: G.add_edge(a, b, weight=npmi)
comms = louvain_communities(G, weight="weight", seed=7)
cmap = {k: i for i, c in enumerate(comms) for k in c}
LABEL = {frozenset(["Color","Fill","Object","Shape","Size"]): "objectness",
         frozenset(["Contact","Direction","Distance","Path"]): "goal-directedness",
         frozenset(["Reverse","Rotation","Symmetry"]): "geometry",
         frozenset(["Number","Order","Pattern","Position"]): "number & arrangement"}
B["keywords"]   = [dict(name=k, n=uni[k], c=cmap[k]) for k in sorted(uni)]
B["links"]      = links
B["communities"] = [dict(id=i, members=sorted(c),
                         label=LABEL.get(frozenset(c), f"community {i}"))
                    for i, c in enumerate(comms)]

# ── per-puzzle payload ──────────────────────────────────────────────────────
grp = q("""MATCH (g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
           WHERE p.is_attention_check = false
           RETURN p.kw_index AS idx, p.task_id AS task, g.id AS gid,
                  g.is_correct AS ok, g.n_submitters AS n, g.grid AS grid""")
kwg = q("""MATCH (pa:Participant)-[:GAVE]->(kr:KeywordReport)-[:ABOUT]->(p:Problem),
                 (kr)-[:SELECTED]->(k:Keyword)
           MATCH (pa)-[:MADE]->(s:Submission)-[:ON_PROBLEM]->(p),
                 (s)-[:LANDED_IN]->(g:SolutionGroup)
           WHERE p.is_attention_check = false
           RETURN p.kw_index AS idx, g.id AS gid, k.name AS kw,
                  count(DISTINCT pa) AS n""")
prof = q("""MATCH (kr:KeywordReport)-[:ABOUT]->(p:Problem), (kr)-[:SELECTED]->(k:Keyword)
            WHERE p.is_attention_check = false
            RETURN p.kw_index AS idx, k.name AS kw, count(DISTINCT kr) AS n""")
nrep = q("""MATCH (kr:KeywordReport)-[:ABOUT]->(p:Problem)
            WHERE p.is_attention_check = false
            RETURN p.kw_index AS idx, count(*) AS n""").set_index("idx")["n"].to_dict()

kwg_by = collections.defaultdict(dict)
for _, r in kwg.iterrows(): kwg_by[r.gid][r.kw] = int(r.n)
prof_by = collections.defaultdict(dict)
for _, r in prof.iterrows(): prof_by[r.idx][r.kw] = int(r.n)

puzzles = []
for idx, sub in grp.groupby("idx"):
    j = json.load(open(f"{KWT}/{idx}.json"))
    sub = sub.sort_values(["ok", "n"], ascending=[False, False])
    total = int(sub.n.sum())
    gs = [dict(id=r.gid, ok=bool(r.ok), n=int(r.n), grid=enc(json.loads(r.grid)),
               kw=kwg_by.get(r.gid, {}))
          for _, r in sub.iterrows() if r.n >= 2 or r.ok]
    shown = [r for _, r in sub.iterrows() if r.n >= 2 or r.ok]
    biggest = max((int(r.n) for r in shown), default=1)
    puzzles.append(dict(
        idx=idx, task=sub.task.iloc[0], n=total,
        # how far people split: 0 = everyone drew the same grid, 1 = all different
        split=round(1 - biggest / total, 3) if total else 0,
        n_wrong=sum(1 for r in shown if not r.ok),
        acc=round(float(sub[sub.ok].n.sum()) / total, 3) if total else 0,
        ex_in=enc(j["train"][0]["input"]), ex_out=enc(j["train"][0]["output"]),
        test=enc(j["test"][0]["input"]),
        n_groups=int(len(sub)), n_reports=int(nrep.get(idx, 0)),
        profile=prof_by.get(idx, {}), groups=gs))
puzzles.sort(key=lambda p: p["idx"])
B["puzzles"] = puzzles

# ── annotated case studies ──────────────────────────────────────────────────
def diff_cells(a, b):
    return [[r, c] for r in range(len(a)) for c in range(len(a[r]))
            if a[r][c] != b[r][c]]
CASES = [
    ("26", "26_wrong2", "Reverse",
     "The completion shape and colour are both right but the positioning is wrong, "
     "caused by reflecting the figure rather than rotating it."),
    ("07", "07_wrong1", "Shape",
     "The grey shapes are fitted into the wall correctly, exactly as in the correct "
     "answer. The originals at the bottom were never cleared."),
    ("13", "13_wrong1", "Pattern",
     "Each square extends out diagonally. Both lines were drawn along the opposite "
     "diagonal: the right operation but mirrored."),
    ("16", "16_wrong1", "Color",
     "A grey path connects the three markers. The correct answer loops."),
]
cases = []
for idx, gid, kw, note in CASES:
    p = next(x for x in puzzles if x["idx"] == idx)
    good = next(g for g in p["groups"] if g["ok"])
    bad  = next(g for g in p["groups"] if g["id"] == gid)
    cases.append(dict(idx=idx, task=p["task"], group=gid, keyword=kw, note=note,
                      kw_in_group=bad["kw"].get(kw, 0), n_group=bad["n"],
                      kw_in_correct=good["kw"].get(kw, 0), n_correct=good["n"],
                      diff=diff_cells(good["grid"], bad["grid"])))
B["cases"] = cases

# ── human vs model describers ───────────────────────────────────────────────
# Comparable metric: the share of a describer's descriptions that mention each
# keyword. Raw counts are not comparable (1,658 human reports vs 40 per model).
n_hum = int(q("""MATCH (kr:KeywordReport)-[:ABOUT]->(p:Problem)
                 WHERE p.is_attention_check=false RETURN count(*) AS c""").c[0])
rate = q("""MATCH (k:Keyword)
  WITH k, COUNT { MATCH (kr:KeywordReport)-[:SELECTED]->(k),
                        (kr)-[:ABOUT]->(p:Problem) WHERE p.is_attention_check=false } AS h
  MATCH (a:Agent)
  WITH k, h, a, COUNT { MATCH (a)-[:AUTHORED]->(x:AgentReport)-[:PICKED]->(k) } AS m,
       COUNT { MATCH (a)-[:AUTHORED]->(y:AgentReport) } AS n
  RETURN k.name AS keyword, 100.0*h/""" + str(n_hum) + """ AS human,
         a.name AS agent, 100.0*m/n AS pct ORDER BY human DESC""")
agents = q("MATCH (a:Agent) RETURN a.name AS name, a.year AS year, a.blind AS blind")            .sort_values("year").to_dict("records")
rates = {}
for _, r in rate.iterrows():
    rates.setdefault(r.keyword, dict(keyword=r.keyword, human=round(r.human, 1)))
    rates[r.keyword][r.agent] = round(r.pct, 1)
rates = sorted(rates.values(), key=lambda x: (-x["human"], x["keyword"]))
for a in agents:
    a["gap"] = round(sum(abs(r[a["name"]] - r["human"]) for r in rates) / len(rates), 1)
    a["n_reports"] = int(q(f"""MATCH (:Agent {{name:'{a['name']}'}})-[:AUTHORED]->(x:AgentReport)
                               RETURN count(x) AS c""").c[0])
mp = q("""MATCH (a:Agent)-[:AUTHORED]->(x:AgentReport)-[:DESCRIBES]->(p:Problem),
                 (x)-[:PICKED]->(k:Keyword)
          RETURN p.kw_index AS idx, a.name AS agent, k.name AS kw, r.rank AS rank
          ORDER BY rank""" .replace("r.rank", "1"))
by_puzzle = collections.defaultdict(lambda: collections.defaultdict(list))
for _, r in mp.iterrows(): by_puzzle[r.idx][r.agent].append(r.kw)
B["models"] = dict(rates=rates, agents=agents, n_human_reports=n_hum,
                   by_puzzle={k: dict(v) for k, v in by_puzzle.items()})

# ── background: two worked puzzles with every example pair ──────────────────
B["intro"] = []
for idx in ("01", "13"):
    j = json.load(open(f"{KWT}/{idx}.json"))
    pz = next(x for x in puzzles if x["idx"] == idx)
    good = next(g for g in pz["groups"] if g["ok"])
    B["intro"].append(dict(idx=idx, task=pz["task"], acc=pz["acc"], n=pz["n"],
        pairs=[dict(i=enc(e["input"]), o=enc(e["output"])) for e in j["train"]],
        test=enc(j["test"][0]["input"]), answer=good["grid"]))

# ── lift table + headline facts ─────────────────────────────────────────────
lift = pd.read_csv(os.path.join(HERE, "..", "queries", "02_results.csv"))
B["lift"] = dict(rows=lift.head(10).to_dict("records"), n_tests=int(len(lift)))
cnt = {l: int(q(f"MATCH (n:{l}) RETURN count(*) AS c").c[0]) for l in
       ("Participant", "Problem", "Keyword", "KeywordReport", "Submission", "SolutionGroup")}
edg = {r: int(q(f"MATCH ()-[x:{r}]->() RETURN count(x) AS c").c[0]) for r in
       ("GAVE", "ABOUT", "SELECTED", "MADE", "ON_PROBLEM", "LANDED_IN", "FOR_PROBLEM")}
acc = q("""MATCH (s:Submission)-[:ON_PROBLEM]->(p:Problem) WHERE p.is_attention_check = false
           RETURN count(*) AS n,
           1.0*sum(CASE WHEN s.outcome='success' THEN 1 ELSE 0 END)/count(*) AS a""")
B["schema"] = dict(nodes=cnt, edges=edg)
# every count on the masthead excludes the attention check, so they agree with
# each other: one description per answer, 1,472 of each
B["facts"] = dict(participants=cnt["Participant"], puzzles=len(puzzles),
                  submissions=int(acc.n[0]), accuracy=round(float(acc.a[0]), 3),
                  answers=cnt["SolutionGroup"], reports=n_hum,
                  distinct_answers_shown=sum(len(p["groups"]) for p in puzzles))
def _grid_cells(q):
    """Cells in the stored answer grids. g.grid is a JSON string, so size() on it
    counts characters, not cells -- decode and multiply rows by columns."""
    import json as _j
    dims = [( len(G), len(G[0]) ) for G in
            (_j.loads(x) for x in q("MATCH (g:SolutionGroup) RETURN g.grid AS grid").grid)]
    dims.sort()
    return dict(grid_cells=sum(r * c for r, c in dims),
                grid_min=f"{dims[0][0]}x{dims[0][1]}",
                grid_max=f"{dims[-1][0]}x{dims[-1][1]}",
                grid_med=f"{dims[len(dims)//2][0]}x{dims[len(dims)//2][1]}")


# dataset inventory: what is in here, and how heterogeneous it is
NODE_DESC = {
 "Participant":  ("a person who took the study",
                  "cohort"),
 "Problem":      ("one ARC puzzle: example input-output grid pairs plus a test input grid",
                  "task id"),
 "Keyword":      ("one of the sixteen words participants could choose from",
                  "name"),
 "KeywordReport":("the words one person chose for one puzzle",
                  "latency, how many words chosen"),
 "Submission":   ("one person's single attempt at one puzzle",
                  "attempt number, outcome"),
 "SolutionGroup":("a distinct answer grid shared by participants",
                  "the grid itself, submitter count, whether it is correct"),
 "Agent":        ("a language model given the same keyword task",
                  "year, whether it was blind to this analysis"),
 "AgentReport":  ("the words one model chose for one puzzle",
                  "how many words chosen"),
}
# what each edge asserts, in the study's own terms
REL_MEANING = {
 "GAVE":        "this person wrote this description",
 "ABOUT":       "the description is of this puzzle",
 "SELECTED":    "the description used this word; the edge carries its rank, 1 to 5",
 "MADE":        "this person submitted this answer",
 "ON_PROBLEM":  "the answer is to this puzzle",
 "LANDED_IN":   "the answer is this exact grid",
 "FOR_PROBLEM": "this answer grid belongs to this puzzle",
 "AUTHORED":    "this model produced this description",
 "DESCRIBES":   "the model's description is of this puzzle",
 "PICKED":      "the model chose this word; the edge carries its rank",
}
REL_ORDER = list(REL_MEANING)
inv_nodes = []
for t, (what, attrs) in NODE_DESC.items():
    inv_nodes.append(dict(type=t, n=int(q(f"MATCH (n:{t}) RETURN count(*) AS c").c[0]),
                          what=what, attrs=attrs))
inv_rels = []
for r in REL_ORDER:
    conn_t = q(f"CALL SHOW_CONNECTION('{r}') RETURN *")   # endpoints from the catalog
    inv_rels.append(dict(rel=r, n=int(q(f"MATCH ()-[x:{r}]->() RETURN count(x) AS c").c[0]),
                         src=str(conn_t.iloc[0, 0]), dst=str(conn_t.iloc[0, 1]),
                         means=REL_MEANING[r]))
B["inventory"] = dict(nodes=inv_nodes, rels=inv_rels,
                      total_nodes=sum(x["n"] for x in inv_nodes),
                      total_edges=sum(x["n"] for x in inv_rels),
                      n_grids=int(q("MATCH (g:SolutionGroup) RETURN count(*) AS c").c[0]),
                      **_grid_cells(q))

# narrowing: the whole dataset down to the cases worth looking at
f = lambda c: int(q(c).c[0])
B["funnel"] = [
 dict(label="submissions", n=f("""MATCH (s:Submission)-[:ON_PROBLEM]->(p:Problem)
      WHERE p.is_attention_check=false RETURN count(*) AS c"""),
      note="one answer per person per puzzle"),
 dict(label="wrong", n=f("""MATCH (s:Submission)-[:LANDED_IN]->(g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
      WHERE p.is_attention_check=false AND g.is_correct=false RETURN count(*) AS c"""),
      note="the correct grid is one of many possible"),
 dict(label="on an answer grid someone else drew too", n=f("""MATCH (s:Submission)-[:LANDED_IN]->(g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
      WHERE p.is_attention_check=false AND g.is_correct=false AND g.n_submitters>=2 RETURN count(*) AS c"""),
      note="not independent error"),
 dict(label="distinct wrong answer grids, drawn more than once", n=f("""MATCH (g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
      WHERE p.is_attention_check=false AND g.is_correct=false AND g.n_submitters>=2
      RETURN count(DISTINCT g) AS c"""), note="each one a rule someone inferred"),
 dict(label="drawn by five or more", n=f("""MATCH (g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
      WHERE p.is_attention_check=false AND g.is_correct=false AND g.n_submitters>=5
      RETURN count(DISTINCT g) AS c"""), note="strong enough to read"),
 dict(label="worked through in error anatomy", n=4, note="the interpretable ones"),
]

# convergence: how often two people independently drew the identical wrong grid
conv = q("""MATCH (s:Submission)-[:LANDED_IN]->(g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
            WHERE p.is_attention_check = false AND g.is_correct = false
            RETURN p.kw_index AS idx, p.task_id AS task, g.id AS gid,
                   g.n_submitters AS n""")
gsz = conv.drop_duplicates("gid")
B["converge"] = dict(
    wrong_submissions=int(len(conv)),
    on_a_repeated_grid=int((conv.n >= 2).sum()),
    distinct_wrong_grids=int(len(gsz)),
    grids_2plus=int((gsz.n >= 2).sum()),
    grids_5plus=int((gsz.n >= 5).sum()),
    puzzles_with_repeat=int(conv[conv.n >= 2].idx.nunique()),
    puzzles_total=int(len(puzzles)))

# ── explorer graph ──────────────────────────────────────────────────────────
# A display projection: KeywordReport and Submission are collapsed into direct
# edges so the picture stays readable. The stored graph reifies both -- see the
# schema on the same page. Answer grids are limited to the recognised ones
# (correct, or drawn by two or more people).
EX_N, EX_IDX = [], {}
def _n(kind, key, label, sub="", ref=""):
    if (kind, key) not in EX_IDX:
        EX_IDX[(kind, key)] = len(EX_N); EX_N.append([kind, label, sub, ref])
    return EX_IDX[(kind, key)]

for _, r in q("MATCH (k:Keyword) RETURN k.name AS k").iterrows():
    _n("K", r.k, r.k, B["communities"][cmap[r.k]]["label"] if r.k in cmap else "")
for p_ in puzzles:
    _n("P", p_["idx"], p_["task"], f"{round(p_['acc']*100)}% correct", p_["idx"])
    for g in p_["groups"]:
        _n("A", g["id"], g["id"].split("_", 1)[1], f"{p_['task']} · {g['n']} people"
           + (" · correct" if g["ok"] else ""), f"{p_['idx']}|{g['id']}")
for _, r in q("MATCH (u:Participant) RETURN u.id AS u").iterrows():
    _n("U", r.u, r.u, "participant")
for a in agents:
    _n("M", a["name"], a["name"], f"model, {a['year']}")
for c in B["communities"]:
    _n("C", str(c["id"]), c["label"], f"{len(c['members'])} keywords")

# Every edge carries a weight. Thickness is drawn per relation type, since a
# description count and a submitter count are not on the same scale.
EX_REL = ["USED", "DESCRIBED", "DREW", "FOR_PUZZLE", "PROFILE", "IN_COMMUNITY",
          "PICKED", "USERS_DREW"]
EX_E, seen_e = [], {}
def _e(a, b, rel, w=1):
    k = (a, b, rel)
    if a == b: return
    if k in seen_e: seen_e[k][3] = max(seen_e[k][3], w); return
    row = [a, b, EX_REL.index(rel), w]; seen_e[k] = row; EX_E.append(row)

for _, r in q("""MATCH (u:Participant)-[:GAVE]->(kr:KeywordReport)-[:ABOUT]->(p:Problem),
                       (kr)-[:SELECTED]->(k:Keyword)
                 WHERE p.is_attention_check=false
                 RETURN u.id AS u, k.name AS k, p.kw_index AS p""").iterrows():
    _e(_n("U", r.u, r.u), _n("K", r.k, r.k), "USED")
    _e(_n("U", r.u, r.u), _n("P", r.p, ""), "DESCRIBED")
for _, r in q("""MATCH (u:Participant)-[:MADE]->(s:Submission)-[:LANDED_IN]->(g:SolutionGroup),
                       (s)-[:ON_PROBLEM]->(p:Problem)
                 WHERE p.is_attention_check=false
                 RETURN u.id AS u, g.id AS g""").iterrows():
    if ("A", r.g) in EX_IDX:
        _e(_n("U", r.u, r.u), EX_IDX[("A", r.g)], "DREW")
for p_ in puzzles:
    for g in p_["groups"]:
        _e(EX_IDX[("A", g["id"])], EX_IDX[("P", p_["idx"])], "FOR_PUZZLE", g["n"])
        # how many of this answer's drawers used each word; sparse on purpose
        for kw, n in g["kw"].items():
            if n >= 3 and ("K", kw) in EX_IDX:
                _e(EX_IDX[("K", kw)], EX_IDX[("A", g["id"])], "USERS_DREW", n)
    for kw, n in sorted(p_["profile"].items(), key=lambda x: (-x[1], x[0]))[:5]:
        _e(EX_IDX[("P", p_["idx"])], EX_IDX[("K", kw)], "PROFILE", n)
for c in B["communities"]:
    for m in c["members"]:
        _e(EX_IDX[("K", m)], EX_IDX[("C", str(c["id"]))], "IN_COMMUNITY", 1)
picked = collections.Counter()
for a in agents:
    for idx_, picks in B["models"]["by_puzzle"].items():
        for kw in picks.get(a["name"], []):
            picked[(a["name"], kw)] += 1
        if ("P", idx_) in EX_IDX:
            _e(EX_IDX[("M", a["name"])], EX_IDX[("P", idx_)], "DESCRIBED", 1)
for (an_, kw), n in picked.items():
    _e(EX_IDX[("M", an_)], EX_IDX[("K", kw)], "PICKED", n)

# Edges are appended in whatever order the queries return rows, which Kuzu does
# not guarantee. Sorting makes a rebuild byte-identical, so graph.json only
# changes in git when the data actually changes.
EX_E.sort(key=lambda e: (e[2], e[0], e[1]))
B["explorer"] = dict(nodes=EX_N, edges=EX_E, rels=EX_REL)

B["cypher"] = {f: open(os.path.join(HERE, "..", "queries", f)).read()
               for f in ("01_keyword_to_answer.cypher", "02_keyword_error_lift.cypher")}

path = os.path.join(OUT, "graph.json")
# sort_keys so a rebuild is byte-identical: several of these dicts are built
# from Kuzu rows, whose order is not guaranteed between runs. The site reads
# every one of them by key or sorts it itself, so the order here is free.
json.dump(B, open(path, "w"), separators=(",", ":"), sort_keys=True)
print(f"wrote {os.path.normpath(path)}  ({os.path.getsize(path)/1024:.0f} KB)")
print(f"  {len(puzzles)} puzzles, {sum(len(p['groups']) for p in puzzles)} answer grids, "
      f"{len(cases)} case studies")
