"""Extend the graph with non-human describers (GPT-4o, Gemini).

Run after build_layer1.py. Nothing in the existing graph is modified: two new
node types and three new relation types are added alongside, which is the point
-- the same schema accommodates a describer that is not a person.

    (:Agent)-[:AUTHORED]->(:AgentReport)-[:DESCRIBES]->(:Problem)
    (:AgentReport)-[:PICKED {rank, score}]->(:Keyword)
"""
import os, json
import kuzu, pandas as pd

import paths
DB = str(paths.DB)
K = str(paths.LLM_KEYWORDS)
HERE_DATA = str(paths.DATA)
paths.require(paths.LLM_KEYWORDS)
# (name, csv path, score column, year, blind)
# "blind" records whether the describer had any exposure to the human results.
# Both entries below were collected before this analysis existed. Anything added
# later must set blind=False unless it was genuinely run cold -- a describer that
# has seen which words the earlier ones were missing is not evidence.
# New runs: python etl/rerun_llm_keywords.py --models ... , which writes to
# data/<model>_keywords.csv in this same shape; add the row here to load it.
SRC  = [("gpt-4o",  f"{K}/llm keywords/4o_keywords_compiled.csv",     "gpt_score",    2024, True),
        ("gemini",  f"{K}/llm keywords/gemini_keywords_compiled.csv", "gemini_score", 2024, True)]
for _f in sorted(__import__("glob").glob(os.path.join(HERE_DATA, "*_keywords.csv"))):
    _n = os.path.basename(_f).replace("_keywords.csv", "")
    SRC.append((_n, _f, None, 2026, True))

conn = kuzu.Connection(kuzu.Database(DB))
# Idempotent: drop only this script's own tables (relations first), so re-running
# after a schema change does not require rebuilding the whole graph.
for t in ("PICKED", "DESCRIBES", "AUTHORED", "AgentReport", "Agent"):
    try:
        conn.execute(f"DROP TABLE {t}")
    except Exception:
        pass
have = set()
for ddl in [
    "CREATE NODE TABLE Agent(name STRING, kind STRING, year INT64, blind BOOLEAN, PRIMARY KEY(name))",
    "CREATE NODE TABLE AgentReport(id STRING, n_selected INT64, PRIMARY KEY(id))",
    "CREATE REL TABLE AUTHORED(FROM Agent TO AgentReport)",
    "CREATE REL TABLE DESCRIBES(FROM AgentReport TO Problem)",
    "CREATE REL TABLE PICKED(FROM AgentReport TO Keyword, rank INT64, score DOUBLE)",
]:
    name = ddl.split("TABLE ")[1].split("(")[0]
    if name not in have:
        conn.execute(ddl)

scores = pd.read_csv(f"{K}/keyword_scoring_comparison_clean.csv", dtype={"task": str})
scores["idx"] = scores.task.str.zfill(2)
valid = {r[0] for r in conn.execute("MATCH (p:Problem) RETURN p.kw_index").get_as_df().values}
kws   = {r[0] for r in conn.execute("MATCH (k:Keyword) RETURN k.name").get_as_df().values}

n_rep = n_pick = 0
for agent, path, score_col, year, blind in SRC:
    conn.execute("MERGE (:Agent {name:$n, kind:'model', year:$y, blind:$b})",
                 {"n": agent, "y": year, "b": blind})
    d = pd.read_csv(path, dtype={"task": str})
    d["idx"] = d.task.str.zfill(2)
    sc = scores.set_index(["idx", "keyword"])[score_col].to_dict() if score_col else {}
    for _, r in d.iterrows():
        if r.idx not in valid: continue
        picks = [r.get(f"ai_keyword{i}") for i in range(1, 6)]
        picks = [p for p in picks if isinstance(p, str) and p.strip() in kws]
        if not picks: continue
        rid = f"{agent}_{r.idx}"
        conn.execute("CREATE (:AgentReport {id:$i, n_selected:$n})",
                     {"i": rid, "n": len(picks)})
        conn.execute("MATCH (a:Agent {name:$a}),(x:AgentReport {id:$i}) "
                     "CREATE (a)-[:AUTHORED]->(x)", {"a": agent, "i": rid})
        conn.execute("MATCH (x:AgentReport {id:$i}),(p:Problem {kw_index:$k}) "
                     "CREATE (x)-[:DESCRIBES]->(p)", {"i": rid, "k": r.idx})
        for rank, kw in enumerate(picks, start=1):
            conn.execute("MATCH (x:AgentReport {id:$i}),(k:Keyword {name:$n}) "
                         "CREATE (x)-[:PICKED {rank:$r, score:$s}]->(k)",
                         {"i": rid, "n": kw.strip(), "r": rank,
                          "s": float(sc.get((r.idx, kw.strip()), 6.0 - rank))})
            n_pick += 1
        n_rep += 1

print(f"[agents] {n_rep} model reports, {n_pick} keyword picks")
for lbl in ("Agent", "AgentReport"):
    n = conn.execute(f"MATCH (n:{lbl}) RETURN count(*) AS c").get_next()[0]
    print(f"  {lbl:<12} {n}")

# the headline contrast, as a graph query
# Share of each describer's descriptions that mention the keyword -- comparable
# across describers, which raw counts are not (1,658 human reports vs 40 each).
print("\nkeyword use, as a share of that describer's descriptions:")
print(conn.execute("""
MATCH (k:Keyword)
WITH k, COUNT { MATCH (kr:KeywordReport)-[:SELECTED]->(k),
                      (kr)-[:ABOUT]->(p:Problem) WHERE p.is_attention_check=false } AS h
MATCH (a:Agent)
WITH k, h, a, COUNT { MATCH (a)-[:AUTHORED]->(x:AgentReport)-[:PICKED]->(k) } AS m
RETURN k.name AS keyword, round(100.0*h/1658, 1) AS human_pct,
       a.name AS agent, round(100.0*m/40, 1) AS agent_pct
ORDER BY human_pct DESC, agent
""").get_as_df().pivot(index=["keyword","human_pct"], columns="agent",
                       values="agent_pct").reset_index().to_string(index=False))
