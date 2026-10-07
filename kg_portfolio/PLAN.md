# CogARC Knowledge Graph — portfolio plan

A knowledge-graph portfolio built on the CogARC keyword and common-solution data, aimed at an
applied-scientist interview where knowledge graphs are a stated requirement.

---

## 1. The framing problem to solve first

An interactive force-directed network is not a knowledge graph, and an interviewer who works
with knowledge graphs will know the difference within about thirty seconds. If the site is a
pretty node-link diagram of keyword co-occurrence, the likely read is "she made a network
visualization," which is weaker than saying nothing.

What makes something a knowledge graph is four things, none of which is the picture:

1. **A schema** — typed entities and typed relations, declared, with constraints. Someone can
   read it and know what is in the graph without looking at the data.
2. **Integration** — two or more sources reconciled into one vocabulary, so the graph answers
   questions that no single source answers.
3. **Entity resolution** — a defensible account of when two records denote the same thing.
4. **A query interface** — Cypher or SPARQL, not a hand-rolled traversal, so the graph is
   interrogable rather than just renderable.

The good news: your data gives you all four honestly, and #2 and #3 are where this project is
unusually strong. Most portfolio graphs are a single table reshaped into nodes and edges. Yours
integrates two participant cohorts that **do not overlap**, which forces a real modeling
decision about what can and cannot be linked. That is the most interview-relevant thing in the
whole project, and it is worth foregrounding rather than hiding.

**What interviewers actually probe.** Expect some subset of:

- "When would you choose a graph over a relational schema?" — variable-depth traversal,
  relationship-first queries, schema that keeps changing, path and reachability questions.
- "Walk me through your schema and why you modeled it that way." — the reification decisions
  below are your answer.
- "How did you do entity resolution?" — the content-hash match on problem grids.
- "How do you keep a graph from becoming a swamp?" — constraints, provenance, a vocabulary.
- "What would break at 10^8 edges?" — have an answer; see §7.
- Increasingly: "Have you done graph-based retrieval for an LLM?" — §5, Tier 3.

---

## 2. What the data supports

**Two cohorts, zero overlap.** The keyword study (187 participants) and the behavioral
MTurk study (233 participants) share no participant IDs. Verified. So there are exactly two
legitimate linkage levels, and the graph should make the distinction explicit:

| level | scope | what it licenses |
|---|---|---|
| **Problem-level (aggregate)** | all 41 keyword problems | "problems described with *Symmetry* tend to produce this error group" |
| **Submission-level (within keyword cohort)** | 8 problems, 216 submissions, 131 participants | "this person said *Symmetry* and then gave this specific answer" |

Any path in the UI from a keyword to a solution must be badged with which of the two it is.
Conflating them is the one scientific error that would undermine the whole thing, and showing
that you handled it deliberately is a strength.

**Entity resolution — already solved.** The keyword study indexes problems `00`–`40`; the
behavioral study uses ARC task IDs. I matched all **41 of 41** keyword problems to behavioral
task IDs, 1:1, by hashing the example input–output grid pairs. 27 matched on the exact
example set; the other 14 matched on 2–3 shared pairs because the keyword version held out a
different pair as the test item — same underlying ARC problem, different train/test split.
Mapping written to `kg_portfolio/kw_index_to_task.json`.

This is a genuine entity-resolution story: no shared key, resolved on content, with a
tolerance rule and a stated confidence criterion. Write it up; it is more interesting than the
graph itself.

**Inventory already computed** (so ETL is assembly, not analysis):

| source | content |
|---|---|
| `MTurk-Keywords/keywords-8.json` | 187 participants, 1,679 keyword reports, 2–5 **ranked** keywords each |
| `MTurk-Keywords/puzzles-8.json` | full ms-resolution action logs for the keyword cohort |
| `arc-error-grouping/MTurk/submission_tags.csv` | 13,572 submissions tagged `success` / `top_cluster_1-3` / `miscellaneous` |
| `arc-error-grouping/MTurk/copied_solution_groups/` | the actual answer **grids** per solution group |
| `arc-error-grouping/MTurk/visualized_solutions/` | pre-rendered PNGs of each group |
| `Keywords/filtered_submission_tags.csv` | the within-cohort join: keywords + own outcome, 8 problems |
| `Keywords/keyword_ranking_weighted_summary.csv` | **keyword × error-group over-representation, with deltas** |
| `Keywords/outputs/keyword_glove_cosine.csv` | 16×16 GloVe semantic similarity |
| `Keywords/outputs_ck_keyword_coocc/pmi_npmi.csv` | keyword × core-knowledge PMI/NPMI |
| `Keywords/keyword_task_ck_tags.csv` | core-knowledge tags per problem |
| `Keywords/keyword_cluster_labels_K6.csv`, `cluster_signature_table_K6.csv` | keyword and problem clusters |
| `Keywords/llm keywords/` | GPT-4o and Gemini keyword reports on the same problems |
| `Keywords/keyword-concreteness.xlsx` | concreteness rating per keyword |

`keyword_ranking_weighted_summary.csv` is already the edge you wanted: which keywords are
over-represented among people who produced error group X versus those who succeeded.

The 16-keyword controlled vocabulary **is an ontology** — small, closed, with semantic
similarity, co-occurrence statistics, concreteness ratings, and a mapping to a four-category
upper level (core knowledge). Say "controlled vocabulary with a formal upper ontology," because
that is what it is.

---

## 3. Proposed schema

Property graph. Reified where a relationship has its own identity.

**Node types**

```
(:Problem        {task_id, kw_index, n_examples, grid_h, grid_w,
                  difficulty, first_attempt_rate, accuracy})
(:Keyword        {name, glove_vec, concreteness, cluster})
(:CoreKnowledge  {code})              # Obj | Geo | Num | GoalDirected
(:SolutionGroup  {id, kind, grid, n_submitters, is_correct})
(:Submission     {id, attempt_no, outcome})
(:KeywordReport  {id, latency_ms, timestamp, n_selected})
(:Participant    {id, cohort})        # cohort: keyword | behavioral
(:Agent          {name, kind})        # human | gpt-4o | gemini
(:Cluster        {id, kind, label})
```

**Relation types**

```
(:Participant)-[:MADE]->(:Submission)-[:ON]->(:Problem)
(:Submission)-[:LANDED_IN]->(:SolutionGroup)
(:Submission)-[:FOLLOWED_BY]->(:Submission)          # attempt 1 -> 2 -> 3
(:Participant)-[:GAVE]->(:KeywordReport)-[:ABOUT]->(:Problem)
(:KeywordReport)-[:SELECTED {rank}]->(:Keyword)
(:Problem)-[:PROFILED_AS {weighted_score}]->(:Keyword)        # derived aggregate
(:Problem)-[:TAGGED {level}]->(:CoreKnowledge)
(:Keyword)-[:COOCCURS_WITH {npmi, count}]->(:Keyword)
(:Keyword)-[:SIMILAR_TO {glove_cosine}]->(:Keyword)
(:Keyword)-[:OVERREPRESENTED_IN {delta, weighted_rate, n}]->(:SolutionGroup)
(:Agent)-[:AUTHORED]->(:KeywordReport)
(:Keyword|:Problem)-[:IN_CLUSTER]->(:Cluster)
```

**Three modeling decisions to be ready to defend**

1. **`KeywordReport` is a node, not an edge.** A report has a timestamp, a latency, a
   *ranked* list, and an author that may be a human or a model. An edge cannot carry an
   ordered multi-valued payload or participate in further relations. This is standard
   reification and the interviewer will recognize it.
2. **`Submission` is a node.** Attempt order matters — error recovery across attempts 1→2→3 is
   a path query, which is precisely the thing a graph is better at than a table.
3. **`PROFILED_AS` is marked derived.** It is an aggregate over `SELECTED` edges. Keeping
   both the raw and the derived layer, and labelling which is which, is the provenance
   discipline that keeps a graph from rotting.

Add uniqueness constraints on every `id`, and a `source` property on every derived edge. Cheap,
and it is the answer to "how do you stop it becoming a swamp."

---

## 4. The queries that demonstrate competence

These are the portfolio. The site is the wrapper.

1. **Multi-hop with aggregation.** For keyword K, which error groups is it over-represented
   in, and do those problems share a core-knowledge category? Three hops plus a group-by —
   the canonical "why a graph" query.
2. **Provenance-aware paths.** Same question answered two ways: within-cohort
   (`Keyword ← SELECTED ← KeywordReport ← GAVE ← Participant → MADE → Submission → LANDED_IN
   → SolutionGroup`, 8 problems) and aggregate (`Keyword ← PROFILED_AS ← Problem →
   HAS_GROUP → SolutionGroup`, 41 problems). Show both and badge them. This is the single best
   thing in the project.
3. **Two graphs over the same nodes.** Community detection on `COOCCURS_WITH` (how people
   actually co-use keywords) versus `SIMILAR_TO` (GloVe semantics). Where they disagree is a
   real result: people's conceptual grouping is not lexical similarity.
4. **Link prediction.** Hold out `PROFILED_AS` edges; predict from graph structure with node2vec
   or a PyKEEN model (TransE / ComplEx / RotatE). You already have `keyword_prediction_cv.py`
   as a non-graph baseline, so you can report whether the graph structure adds anything — and
   if it does not, say so. A negative result reported honestly is a credibility win.
5. **Anomaly / bridge detection.** Problems whose keyword profile is far from their
   core-knowledge neighbours but which share an error signature with a distant problem. This
   is the "surfaces a link no single table would show" demo.
6. **Human versus model.** Same problem, human profile versus GPT-4o and Gemini profiles. Does
   human–model keyword divergence predict where models fail? Directly relevant to an applied
   scientist role, and you already have the model keywords.

---

## 5. Build, staged

Each tier is independently presentable. Stop wherever the calendar runs out.

**Tier 1 — the graph itself (the non-negotiable core)**
- `etl/` Python: load the sources, resolve entities, emit nodes/edges CSVs
- Load into Neo4j (Aura free tier) with constraints; schema diagram in the README
- A notebook with the six queries above, each with the Cypher shown and the result rendered
- README: schema, entity resolution, the two-cohort problem, what the graph answers

This alone is legitimate, defensible knowledge-graph work. If you only do this, you are fine.

**Tier 2 — the site**
- Static build (GitHub Pages), graph pre-exported to JSON so nothing can fail live
- Cytoscape.js or Sigma.js canvas; left panel selects a keyword or problem; right panel renders
  **the actual ARC solution grids** for the selected group
- Every view shows the Cypher that produced it, and a provenance badge for the linkage level
- The grids are the visual hook. Nobody else's graph portfolio has 30×30 colour grids in it.

**Tier 3 — the parts that read as current**
- Graph embeddings + link prediction (query 4), with the honest baseline comparison
- **Text-to-Cypher**: a natural-language box that generates Cypher, shows it, runs it, and
  renders the subgraph. Grounding an LLM in a graph instead of letting it answer from
  parameters is the thing people are hiring for right now.
- Optionally embed Kuzu-WASM so Cypher runs client-side with no server (verify the current
  WASM build before committing to it; the static JSON path is the fallback)

**Cut list if time is short:** Tier 3 embeddings before Tier 3 text-to-Cypher; the LLM
comparison before the anomaly query; polish before any of it. Do not cut the README.

---

## 6. Learning pointers

Ordered by return on time, assuming you want to be able to hold a conversation, not pass a
certification. Titles are given precisely so you can find current editions; verify links.

**Conceptual, read first**
- Hogan et al., *Knowledge Graphs* — ACM Computing Surveys 54(4), 2021. The canonical survey;
  also exists as a Springer book. Read §1–3 for vocabulary, skim the rest. This is where terms
  like "property graph vs RDF," "schema, identity, context," and "ontology" get fixed.
- Barrasa & Webber, *Building Knowledge Graphs* (O'Reilly, 2023). Practical, property-graph
  flavoured; free from Neo4j.
- Stanford **CS520 Knowledge Graphs** — free lectures, good on ontology design and the
  "what is a knowledge graph" debate.

**Hands-on, do in parallel**
- **Neo4j GraphAcademy** — free. Cypher Fundamentals, then Graph Data Science Fundamentals.
  A day or two, and the completion badges are legitimately mentionable.
- Needham & Hodler, *Graph Algorithms* (O'Reilly) — free from Neo4j. Skim for which algorithm
  answers which question; you need centrality, community detection, and similarity.
- `PyKEEN` for knowledge-graph embeddings. Read the TransE and ComplEx entries and run one.

**Know it exists, do not sink time**
- RDF side: W3C SPARQL 1.1, SHACL for validation, OWL for reasoning; `rdflib` and Oxigraph in
  Python. Pharma, biomedical, and library/government work are RDF-heavy; most product
  engineering is property-graph. Be able to say what the trade-off is — RDF buys you global
  identifiers, standard serialisation, and reasoning; property graphs buy you ergonomics and
  speed — and which you would pick for a given problem.
- GraphRAG: Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization* (Microsoft Research, 2024). Read the method section. One paper is enough to
  discuss it credibly.
- Wikidata as the worked example of a large public knowledge graph; look at one entity page and
  its underlying statements with qualifiers and references, because that is provenance done
  properly.

**The one question to rehearse**
"When is a graph the wrong choice?" Have a real answer: fixed-depth joins over a stable schema,
heavy aggregation over columns, anything that is really a star schema. Saying "graphs are always
better" is the fastest way to lose the room.

---

## 7. Honest risks, and what to say

- **"This is one dataset you already knew."** True, and fine — depth beats breadth in a
  portfolio. Lead with the modeling and integration decisions, not the data volume.
- **"Your graph is small."** It is: roughly 500 nodes and a few thousand edges. Say so first,
  then say what would change at scale — batch loaders instead of per-row `MERGE`, partitioning
  by problem, precomputed projections for the GDS algorithms, and a refusal to materialise
  derived edges you can compute on read. Volunteering the limits reads as seniority.
- **"Did the graph actually tell you anything?"** Have one concrete finding ready from query 3
  or 5. If it turns out the graph adds nothing over the tables, say that; it is a legitimate
  result and claiming otherwise is detectable.
- **Scope creep.** The temptation is to make the visualisation beautiful. The schema, the
  entity resolution, and the six queries are what get you hired. Budget accordingly.

---

## 8. Repository layout

Keep this out of the analysis repo so it can be public.

```
cogarc-kg/
  README.md               schema diagram, ER story, the two-cohort problem, findings
  etl/
    resolve_problems.py   content-hash entity resolution -> kw_index_to_task.json
    build_nodes.py
    build_edges.py
    load_neo4j.py         constraints + batch load
  schema/
    schema.cypher         constraints and indexes
    schema.md             entity and relation reference
  queries/
    01_keyword_to_errors.cypher   ... 06_human_vs_model.cypher
  notebooks/
    showcase.ipynb        the six queries, run, with output
    link_prediction.ipynb
  site/                   static build -> GitHub Pages
    data/graph.json
  data/                   small derived CSVs (not raw participant data)
```

Do not commit raw participant data. Derived, aggregated node/edge files only, and say in the
README that the source data is governed by the study's IRB.
