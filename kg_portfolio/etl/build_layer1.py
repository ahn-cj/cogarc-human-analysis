"""Build the Layer 1 (keyword cohort) CogARC knowledge graph in Kuzu.

Sources
  keywords-8.json                 187 participants x ranked keyword reports
  submissions/<idx>.json/*.json   1,656 submitted answer grids
  keywords-testset/<idx>.json     the problems as shown (answer key withheld)
  ARC training tasks              answer keys, recovered via content matching

Modeling notes
  - Submission is a node: the relationship is 4-way (participant, problem,
    solution group, attempt), and edges are binary.
  - Every DISTINCT submitted grid is a SolutionGroup with n_submitters as a
    property. The "common error" threshold is NOT baked into the schema --
    it is a query-time filter, so changing it does not require a rebuild.
  - KeywordReport is a node: it carries a timestamp, a latency and an ORDERED
    keyword list, none of which fit on an edge.
"""
import json, glob, os, collections, shutil
import kuzu

import paths
KWD, KWT, TR = paths.KEYWORD_STUDY, paths.KEYWORD_TESTSET, paths.ARC_TRAINING
OUT = str(paths.DB)
paths.require(KWD, KWT, TR)

# ── entity resolution: keyword problem index -> ARC task id, by grid content ──
def resolve_problems():
    def pairs(p):
        d = json.load(open(p)); out = set()
        for k in ("train", "test"):
            for ex in d.get(k, []) or []:
                out.add(json.dumps([ex.get("input"), ex.get("output")], sort_keys=True))
        return out
    kw = {os.path.basename(f)[:-5]: pairs(f) for f in glob.glob(f"{KWT}/*.json")}
    tr = {os.path.basename(f)[:-5]: pairs(f) for f in glob.glob(f"{TR}/*.json")}
    mapping = {}
    for idx, P in kw.items():
        best = max(((len(P & Q) / max(1, min(len(P), len(Q))), len(P & Q), t)
                    for t, Q in tr.items()), default=(0, 0, None))
        if best[1] > 0:
            mapping[idx] = best[2]
    assert len(set(mapping.values())) == len(mapping), "mapping not 1:1"
    return mapping

# ── answer keys: the keyword test item's output, found in the ARC task ───────
def answer_keys(mapping):
    keys = {}
    for idx, task in mapping.items():
        tin = (json.load(open(f"{KWT}/{idx}.json")).get("test") or [{}])[0].get("input")
        arc = json.load(open(f"{TR}/{task}.json"))
        for k in ("train", "test"):
            for ex in arc.get(k, []) or []:
                if ex.get("input") == tin:
                    keys[idx] = json.dumps(ex.get("output")); break
            if idx in keys: break
    return keys

# Problem 00 is the attention check: shown to every participant and solved by
# every participant. It measures the PARTICIPANT, not the task -- zero variance,
# no error groups, and an atypical keyword profile (Shape +37pp, Direction -25pp
# against the other problems) that is over-weighted ~5x because everyone saw it.
# It is dropped here, at load, so no Problem, Submission, KeywordReport or
# SolutionGroup for it is ever created and the graph inventory agrees with every
# figure computed from it. `is_attention_check` stays on the Problem table and
# the analytic queries keep filtering on it, so re-admitting the puzzle is a
# one-line change that does not silently alter any result.
ATTENTION_CHECK = {"00"}


def main():
    mapping = {i: t for i, t in resolve_problems().items() if i not in ATTENTION_CHECK}
    keys = answer_keys(mapping)
    print(f"[resolve] {len(mapping)} problems mapped 1:1; {len(keys)} answer keys recovered "
          f"({len(ATTENTION_CHECK)} attention check excluded)")

    # ── submitted grids ─────────────────────────────────────────────────────
    subs = []
    for f in glob.glob(f"{KWD}/submissions/*.json/*_submission.json"):
        d = json.load(open(f))
        idx = str(d["task"]).replace(".json", "")
        if idx not in mapping:
            continue
        subs.append(dict(idx=idx, subject=d["subject"],
                         attempt=int(d.get("submission_count", 1)),
                         grid=json.dumps(d["grid_data"])))
    # one SolutionGroup per distinct grid per problem
    by_prob = collections.defaultdict(collections.Counter)
    for s in subs:
        by_prob[s["idx"]][s["grid"]] += 1
    groups = {}
    for idx, cnt in by_prob.items():
        correct = keys.get(idx)
        wrong_rank = 0
        for grid, n in cnt.most_common():
            is_corr = (grid == correct)
            if not is_corr:
                wrong_rank += 1
            groups[(idx, grid)] = dict(
                id=f"{idx}_{'success' if is_corr else f'wrong{wrong_rank}'}",
                kind="success" if is_corr else "wrong",
                group_rank=0 if is_corr else wrong_rank,
                n_submitters=n, is_correct=is_corr, grid=grid)

    # ── keyword reports ─────────────────────────────────────────────────────
    # Data quality: the primary-key constraint surfaced three issues here, all
    # recorded rather than silently absorbed.
    #   (a) one session record has subj_ID = None and empty image_name (15 rows)
    #       -- unattributable to a person or a problem, so dropped;
    #   (b) three participant x problem pairs have repeated identical reports
    #       submitted 0.3-37 s apart (one of them five times) -- submit-button
    #       double-fires; the earliest is kept;
    #   (c) counts for both cases are printed, so the cleaning is auditable
    #       rather than assumed.
    seen, reports, dropped = {}, [], collections.Counter()
    for line in open(f"{KWD}/keywords-8.json"):
        r = json.loads(line)
        if not r.get("subj_ID"):
            dropped["null_subject"] += len(r.get("submissions", [])); continue
        for s in r.get("submissions", []):
            idx = s["image_name"].replace(".json.png", "")
            if not idx.strip():
                dropped["empty_problem"] += 1; continue
            if idx in ATTENTION_CHECK:
                dropped["attention_check"] += 1; continue
            if idx not in mapping or not s.get("keywords"):
                dropped["unmapped_or_empty_kw"] += 1; continue
            key = (r["subj_ID"], idx)
            if key in seen:
                same = seen[key] == s["keywords"]
                dropped["repeat_identical" if same else "repeat_revised"] += 1
                continue
            seen[key] = s["keywords"]
            reports.append(dict(id=f"{r['subj_ID']}_{idx}", subject=r["subj_ID"], idx=idx,
                                latency=float(s.get("highResTime") or 0),
                                kws=s["keywords"]))
    print("[clean] keyword reports dropped:", dict(dropped))
    print(f"[load] {len(subs)} submissions, {len(groups)} distinct answer grids, "
          f"{len(reports)} keyword reports")

    # ── build ───────────────────────────────────────────────────────────────
    if os.path.exists(OUT):
        shutil.rmtree(OUT) if os.path.isdir(OUT) else os.remove(OUT)
    for stale in (OUT + ".wal", OUT + ".lock"):
        if os.path.exists(stale): os.remove(stale)
    db = kuzu.Database(OUT); conn = kuzu.Connection(db)
    for ddl in [
        "CREATE NODE TABLE Participant(id STRING, cohort STRING, PRIMARY KEY(id))",
        "CREATE NODE TABLE Problem(kw_index STRING, task_id STRING, is_attention_check BOOLEAN, PRIMARY KEY(kw_index))",
        "CREATE NODE TABLE Keyword(name STRING, PRIMARY KEY(name))",
        "CREATE NODE TABLE KeywordReport(id STRING, latency_ms DOUBLE, n_selected INT64, PRIMARY KEY(id))",
        "CREATE NODE TABLE Submission(id STRING, attempt_no INT64, outcome STRING, PRIMARY KEY(id))",
        "CREATE NODE TABLE SolutionGroup(id STRING, kind STRING, group_rank INT64, n_submitters INT64, is_correct BOOLEAN, grid STRING, PRIMARY KEY(id))",
        "CREATE REL TABLE GAVE(FROM Participant TO KeywordReport)",
        "CREATE REL TABLE ABOUT(FROM KeywordReport TO Problem)",
        "CREATE REL TABLE SELECTED(FROM KeywordReport TO Keyword, rank INT64)",
        "CREATE REL TABLE MADE(FROM Participant TO Submission)",
        "CREATE REL TABLE ON_PROBLEM(FROM Submission TO Problem)",
        "CREATE REL TABLE LANDED_IN(FROM Submission TO SolutionGroup)",
        "CREATE REL TABLE FOR_PROBLEM(FROM SolutionGroup TO Problem)",
    ]:
        conn.execute(ddl)

    people = sorted({s["subject"] for s in subs} | {r["subject"] for r in reports})
    for p in people:
        conn.execute("CREATE (:Participant {id: $i, cohort: 'keyword'})", {"i": p})
    for idx, task in mapping.items():
        conn.execute("CREATE (:Problem {kw_index: $i, task_id: $t, is_attention_check: $a})",
                     {"i": idx, "t": task, "a": idx in ATTENTION_CHECK})
    for k in sorted({k for r in reports for k in r["kws"]}):
        conn.execute("CREATE (:Keyword {name: $n})", {"n": k})
    for g in groups.values():
        conn.execute("CREATE (:SolutionGroup {id: $id, kind: $k, group_rank: $r, "
                     "n_submitters: $n, is_correct: $c, grid: $g})",
                     {"id": g["id"], "k": g["kind"], "r": g["group_rank"],
                      "n": g["n_submitters"], "c": g["is_correct"], "g": g["grid"]})
    for g in groups.values():
        conn.execute("MATCH (s:SolutionGroup {id:$s}),(p:Problem {kw_index:$p}) "
                     "CREATE (s)-[:FOR_PROBLEM]->(p)",
                     {"s": g["id"], "p": g["id"].rsplit("_", 1)[0]})

    for r in reports:
        conn.execute("CREATE (:KeywordReport {id:$i, latency_ms:$l, n_selected:$n})",
                     {"i": r["id"], "l": r["latency"], "n": len(r["kws"])})
        conn.execute("MATCH (p:Participant {id:$p}),(k:KeywordReport {id:$i}) "
                     "CREATE (p)-[:GAVE]->(k)", {"p": r["subject"], "i": r["id"]})
        conn.execute("MATCH (k:KeywordReport {id:$i}),(pr:Problem {kw_index:$x}) "
                     "CREATE (k)-[:ABOUT]->(pr)", {"i": r["id"], "x": r["idx"]})
        for rank, kw in enumerate(r["kws"], start=1):
            conn.execute("MATCH (k:KeywordReport {id:$i}),(w:Keyword {name:$n}) "
                         "CREATE (k)-[:SELECTED {rank:$r}]->(w)",
                         {"i": r["id"], "n": kw, "r": rank})

    for s in subs:
        g = groups[(s["idx"], s["grid"])]
        sid = f"{s['subject']}_{s['idx']}_a{s['attempt']}"
        outcome = "success" if g["is_correct"] else ("shared_error" if g["n_submitters"] >= 3
                                                    else "idiosyncratic")
        conn.execute("CREATE (:Submission {id:$i, attempt_no:$a, outcome:$o})",
                     {"i": sid, "a": s["attempt"], "o": outcome})
        conn.execute("MATCH (p:Participant {id:$p}),(x:Submission {id:$i}) "
                     "CREATE (p)-[:MADE]->(x)", {"p": s["subject"], "i": sid})
        conn.execute("MATCH (x:Submission {id:$i}),(pr:Problem {kw_index:$k}) "
                     "CREATE (x)-[:ON_PROBLEM]->(pr)", {"i": sid, "k": s["idx"]})
        conn.execute("MATCH (x:Submission {id:$i}),(g:SolutionGroup {id:$g}) "
                     "CREATE (x)-[:LANDED_IN]->(g)", {"i": sid, "g": g["id"]})

    for lbl in ("Participant", "Problem", "Keyword", "KeywordReport", "Submission", "SolutionGroup"):
        n = conn.execute(f"MATCH (n:{lbl}) RETURN count(*)").get_next()[0]
        print(f"  {lbl:<15} {n:>6}")
    print(f"[done] graph at {os.path.normpath(OUT)}")

if __name__ == "__main__":
    main()
