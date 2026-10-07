# CogARC Knowledge Graph

A property graph over human problem-solving behaviour on 40 abstract reasoning puzzles
(ARC-AGI-1, adapted for human administration), linking **what people said a puzzle was
about** to **the answer they actually drew**.

186 participants · 1,472 submissions · 1,658 keyword reports · 688 distinct answers · 42.3% accuracy

The data comes from my dissertation work. The graph is new: the two halves of the study
were collected separately and had never been joined.

---

## The finding

On puzzle `1b60fb0c` — complete a figure so it is rotationally symmetric — **8 of the 10
people who produced one particular wrong answer had described the puzzle with the keyword
*Reverse***. Among people who did not pick Reverse, 15 solved it and 2 landed there.

Decomposing the four competing answers shows three distinct kinds of failure:

| answer | people | shape | position | colour | reading |
|---|---|---|---|---|---|
| `26_success` | 17 | correct | correct | red | — |
| `26_wrong1` | 15 | **different** | — | red | misread which piece was missing |
| `26_wrong2` | 10 | correct | **one row high** | red | right piece, anchored by reflection |
| `26_wrong3` | 3 | correct | correct | **blue** | colour slip, not a reasoning error |

`26_wrong2` is the correct completion shape in the correct colour, displaced vertically by
exactly one row: these participants worked out which piece was missing and anchored it by
reflection rather than rotation. A single "miscellaneous error" label merges all three.

Sweeping every keyword × answer-group pair across all 40 puzzles, this is the strongest
pairing in the dataset (lift 2.19, Fisher p = .0037, next best p = .021).

**It does not survive multiple-comparison correction.** Across 39 tests, Bonferroni and the
Benjamini–Hochberg critical value both sit at .0013. Each participant saw 9 of 40 puzzles,
which was never powered for a sweep this wide. This is a ranked hypothesis set, not a
confirmed effect.

It does generate a clean confirmatory test, independent of the ranking that produced it: if
Reverse-selectors are reflecting rather than rotating, the signature should appear on the
other rotational-symmetry puzzles and not on the reflection ones.

## A third result: the 2024 models describe these puzzles differently

GPT-4o and Gemini each described all 40 puzzles in 2024 using the participants' sixteen-word
vocabulary, and both sit in the graph as `Agent` nodes alongside the people. Counts are not
comparable across describers (1,658 human descriptions against 41 each), so the metric is the
share of a describer's own descriptions using each word. Mean absolute gap from the human
rates: **GPT-4o 16.2 points, Gemini 18.7**.

The divergence is systematic. Both load onto colour and position, and barely touch the
vocabulary of movement and relation:

| keyword | people | GPT-4o | Gemini |
|---|---|---|---|
| Color | 54% | 95% | 68% |
| Position | 21% | 75% | 80% |
| Direction | 31% | 2% | 0% |
| Path | 27% | 0% | 0% |
| Distance | 12% | 0% | 0% |
| Reverse | 11% | 0% | 2% |

Three of the four *goal-directedness* keywords are in that list. The community structure
gives a vocabulary for saying *how* the descriptions differ, not just that they do.

Both were collected before this analysis existed, so neither had exposure to the human
results; the graph records that as a `blind` property on each `Agent`, because a describer
that has seen which words the others were missing is not evidence.

And it closes the loop on the case study. On `1b60fb0c` the people who attempted it described
it as Symmetry (55 of 74), Pattern, Color, Reverse. GPT-4o said Color, Pattern, Position;
Gemini said Color, Position. **The keyword that separates the people who got it wrong from
the people who got it right is one neither model ever reaches for.**

Model describers were added as two new node types and three new relation types alongside the
existing graph — no migration, nothing in the original data touched.

**These descriptions are worth re-running.** They predate the reasoning-model generation and
were produced from digit matrices, where participants saw rendered images — a fairness
problem as much as a capability one. `etl/rerun_llm_keywords.py` repeats the task against any
model, defaulting to the participants' own PNG stimuli, caching per puzzle:

```bash
export OPENROUTER_API_KEY=...
python etl/rerun_llm_keywords.py --provider openrouter \
    --models google/gemini-3-pro meta-llama/llama-4-maverick
```

Results land in `data/` and `add_agents.py` picks them up automatically. Running the same
model with `--mode text` as well separates "models improved" from "images versus matrices".

## A second result

Louvain community detection over the keyword co-selection graph — weighted by NPMI, because
*Color* appears in 912 of 1,658 reports and raw co-occurrence just measures base rate —
returns four communities that map one-to-one onto the corpus's four theoretical
core-knowledge priors, with nothing left over:

```
Color, Fill, Object, Shape, Size      → objectness
Contact, Direction, Distance, Path    → goal-directedness
Reverse, Rotation, Symmetry           → geometry / topology
Number, Order, Pattern, Position      → number, counting, arrangement
```

Recovered from co-selection behaviour alone, using no category labels. Against a published
k-means clustering of the same keywords, ARI = +0.54, and the disagreements favour this
partition — k-means placed *Reverse* with *Color*.

This also explains why the case study is coherent rather than random: *Reverse* and
*Rotation* are the two closest keywords in how people actually use them, so confusing them
produces a structured error.

---

## Schema

Six node types, seven relation types.

```
(:Participant)-[:GAVE]->(:KeywordReport)-[:ABOUT]->(:Problem)
(:KeywordReport)-[:SELECTED {rank}]->(:Keyword)
(:Participant)-[:MADE]->(:Submission)-[:ON_PROBLEM]->(:Problem)
(:Submission)-[:LANDED_IN]->(:SolutionGroup)-[:FOR_PROBLEM]->(:Problem)
```

Three decisions worth defending:

- **`Submission` is a node, not an edge.** The relationship is four-way — participant,
  puzzle, answer group, attempt number — and edges are binary. Any relationship over three
  or more entities has to be reified.
- **Every distinct submitted grid is a `SolutionGroup`**, with `n_submitters` as a property.
  The "shared error" threshold is a query-time filter, not a schema decision, so changing it
  needs no rebuild. Thresholds baked into a schema are how a graph rots.
- **`KeywordReport` is a node** because it carries a timestamp, a latency and an *ordered*
  list — none of which fit on an edge.

Derived nodes and edges carry their provenance (method, parameters, source file), so an
aggregate is never mistaken for an observation.

## Entity resolution

The two halves of the study used different puzzle identifiers and shared no key. All 41
puzzles were matched by hashing the example input–output grid pairs: 27 on an exact example
set, 14 on partial overlap, where the two versions held out a different pair as the test
item. The match is 1:1 and asserted as such in the loader.

Answer keys are withheld from the files shown to participants, so they were recovered from
the original corpus by matching each test input back to its source pair.

Both steps self-validate: the recovered keys agree with the study's own correct/incorrect
labels on **216 of 216** submissions where both exist.

## What is deliberately not here

- **No centrality.** Participant degree is exactly 9 for all 184 participants, by design,
  and puzzle degree spans 92× because of the assignment schedule. Centrality would measure
  the experiment's randomisation, not the data.
- **Cross-problem error structure is reported as a negative.** Whether someone who gives one
  shared wrong answer tends to give another is a bipartite projection of participants onto
  answer groups, normalised against co-assignment. Only 54 participants made a shared error
  on 2+ puzzles and the top pairs rest on counts of one or two. The method is demonstrated;
  the result is not interpretable at this sample size.

## Data handling

- **The attention-check puzzle is flagged, not deleted.** All 184 participants solved it, so
  it has zero variance and produces no error groups, and everyone saw it — five times the
  weight of a typical puzzle. Its keyword profile is atypical (*Shape* +37pp, *Direction*
  −25pp). It is marked `is_attention_check` and filtered from every analytic query, while
  staying in the graph so the screening statement remains supportable. Excluding it leaves
  the headline unchanged and stabilises one keyword's community assignment.
- **A primary-key constraint surfaced two data issues** during the first load: three
  participant × puzzle pairs with identical reports re-fired 0.3–37 seconds apart (submit
  double-fires), and one session record with no participant identifier. Both are dropped
  with counts printed, so the cleaning is auditable rather than assumed.

## Stack

| layer | tool | used for |
|---|---|---|
| language | Python 3.10 | ETL, analysis, page generation |
| graph store | Kuzu 0.11.3 | embedded property graph; all traversal in Cypher |
| query language | Cypher | pattern matching, correlated `COUNT {}` subqueries |
| graph algorithms | NetworkX 3.1 | Louvain community detection, bipartite projection |
| graph ML | PyTorch 2.0, PyTorch Geometric 2.8 | heterogeneous GraphSAGE for link prediction |
| statistics | SciPy 1.11.1 | Fisher's exact test, Spearman, Benjamini–Hochberg |
| | scikit-learn 1.3.0 | adjusted Rand index against the published k-means partition |
| data | pandas 2.3.3, NumPy 1.24.3 | tabular wrangling, grid comparison |
| page | HTML, CSS, vanilla JavaScript | no framework, no CDN, no webfont |
| | inline SVG | concept graph, schema diagram, bar charts |
| | CSS grid | every ARC puzzle grid, one element per cell |
| comparison | GPT-4o, Gemini | keyword descriptions of the same 40 puzzles |

### Where the boundaries are

A graph database is not a graph-algorithms library, and statistics over a result set do not
belong in the query. The projection is pulled with Cypher; the algorithms and tests run on
it. At scale the same split holds — project the subgraph you need rather than running
community detection over the whole store.

## At a larger scale

This graph is small: roughly 2,600 nodes and 9,000 edges. What would change:

- batch loaders instead of per-row `MERGE` (the current load is the slowest stage by far)
- partition by puzzle; the analytic queries are all puzzle-local
- precomputed projections for the algorithm stage, refreshed rather than recomputed
- derived edges computed on read, not materialised

## Link prediction (PyTorch Geometric)

Does the description predict the answer? Posed as ranking: for each submission the
candidates are that puzzle's recognised answers, ranked from the person's keyword report.
Split by participant; answer edges for held-out people are removed from the
message-passing graph. 725 submissions, 172 people, 3.8 candidates on average, 5 seeds.

| model | hit@1 % | MRR | knows |
|---|---|---|---|
| most common answer | **60.7 ± 3.3** | **0.752 ± 0.029** | nothing about the person |
| keyword cosine | 41.9 ± 2.8 | 0.640 ± 0.019 | their words vs each answer's profile |
| GraphSAGE (PyG) | 53.9 ± 10.0 | 0.706 ± 0.062 | two relational layers over the graph |

**The baseline wins.** Predicting the most common answer beats both the keyword similarity
and the GNN, and the GNN's spread across seeds overlaps it. A person's description does not
tell you which grid they will draw beyond which grid is popular. That is consistent with the
rest: the one reliable keyword–error association has a lift of 2.19 and does not survive
correction, which is too small to carry a per-submission prediction. Nine puzzles per person
is the binding constraint.

## The site

The site is a separate repository, `cogarc-kg-site`, so it can be deployed on its own.
`etl/export_site.py` writes `src/data/graph.json` there; that repo's `src/build.py` turns
it into a single self-contained `index.html`. Override the location with
`COGARC_SITE_REPO`.

## Running it

```bash
pip install -r requirements.txt
python run_all.py
```

Six stages: build the graph, add the model describers, run the community and projection
analyses, run the link prediction, export the bundle to the site repo. Then build the site
from that repo.
Open `site/index.html`. It is fully self-contained — no server, no CDN, no webfont — so it
renders identically offline and cannot fail on unfamiliar wifi.

Six views: **what is CogARC?** (what ARC is, with worked examples; where AI performance now
stands, with sources; why human process data; and an interactive walkthrough of the study
interface), **explore** (browse all 40 puzzles, every shared answer, and the keyword profile
behind each; pick a keyword as a lens to see which answers its users landed in), **error
anatomy** (four annotated puzzles with the differing cells ringed), **concept space** (the
keyword graph and its communities), **humans vs models**, and **method**.

```
kg_portfolio/
  run_all.py              rebuild everything
  cogarc_kg.kuzu          the graph
  etl/
    build_layer1.py       entity resolution, answer-key recovery, cleaning, load
    add_agents.py         GPT-4o and Gemini as Agent nodes (schema extension)
    rerun_llm_keywords.py re-run the keyword task against current models
    graph_algos.py        Louvain communities; bipartite projections
    link_prediction.py    PyG GraphSAGE vs baselines, split by participant
    export_site.py        graph -> site/data/graph.json
    build_site.py         injects the bundle into site/template.html
  queries/
    01_reverse_problem26.cypher    the case-study query
    02_keyword_error_lift.cypher   within-puzzle lift sweep
    02_results.csv  03_keyword_npmi.csv  04_problem_similarity.csv
  site/
    template.html         the explorer: markup, styles, client-side rendering
    data/graph.json       every puzzle, answer grid and keyword count
    index.html            generated — template with the data inlined
```

Source data is governed by the study's IRB and is not committed here; only derived,
aggregated outputs.
