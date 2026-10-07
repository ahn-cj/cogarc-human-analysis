// Which keyword -> wrong-answer pairings are over-represented, across all 40 puzzles?
//
// Conditioned WITHIN problem: both keyword use and group membership are
// problem-specific, so pooling across problems invites Simpson's paradox.
// lift = P(landed in g | selected k, on p) / P(landed in g | on p)
// Denominator population throughout = people with BOTH a keyword report and a
// submission on that problem.
MATCH (pa:Participant)-[:GAVE]->(kr:KeywordReport)-[:ABOUT]->(p:Problem),
      (kr)-[:SELECTED]->(k:Keyword)
MATCH (pa)-[:MADE]->(s:Submission)-[:ON_PROBLEM]->(p),
      (s)-[:LANDED_IN]->(g:SolutionGroup)
WHERE g.is_correct = false AND g.n_submitters >= 5
      AND p.is_attention_check = false
WITH p, k, g, count(DISTINCT pa) AS n_kg
WHERE n_kg >= 4
WITH p, k, g, n_kg,
  COUNT { MATCH (q:Participant)-[:GAVE]->(r:KeywordReport)-[:ABOUT]->(p),
                (r)-[:SELECTED]->(k),
                (q)-[:MADE]->(x:Submission)-[:ON_PROBLEM]->(p) }        AS n_k,
  COUNT { MATCH (q:Participant)-[:GAVE]->(r:KeywordReport)-[:ABOUT]->(p),
                (q)-[:MADE]->(x:Submission)-[:ON_PROBLEM]->(p),
                (x)-[:LANDED_IN]->(g) }                                 AS n_g,
  COUNT { MATCH (q:Participant)-[:GAVE]->(r:KeywordReport)-[:ABOUT]->(p),
                (q)-[:MADE]->(x:Submission)-[:ON_PROBLEM]->(p) }         AS n_p
RETURN p.kw_index AS problem, p.task_id AS task, k.name AS keyword,
       g.id AS answer_group, n_kg, n_k, n_g, n_p,
       (1.0 * n_kg / n_k) / (1.0 * n_g / n_p) AS lift
ORDER BY lift DESC
