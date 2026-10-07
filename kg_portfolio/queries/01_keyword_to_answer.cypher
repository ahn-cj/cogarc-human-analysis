// Join a description to the answer the same person drew.
// For the people who used a given keyword on a given puzzle, which answer grid
// did each of them submit? Two independent paths out of Participant -- one
// through KeywordReport, one through Submission -- rejoined on the puzzle.
MATCH (pa:Participant)-[:GAVE]->(kr:KeywordReport)-[:ABOUT]->(p:Problem {kw_index: "26"}),
      (kr)-[:SELECTED]->(:Keyword {name: "Reverse"})
MATCH (pa)-[:MADE]->(sub:Submission)-[:ON_PROBLEM]->(p),
      (sub)-[:LANDED_IN]->(sg:SolutionGroup)
RETURN sg.id AS answer_group, sg.is_correct AS correct,
       sg.n_submitters AS group_size, count(DISTINCT pa) AS n_reverse_people
ORDER BY n_reverse_people DESC
