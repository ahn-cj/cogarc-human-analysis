"""Predict which answer a person will give, from the words they used for the rule.

Task. For each submission the candidates are the answer groups of that puzzle.
Rank them; score hit@1 and MRR. This is the chapter's question posed as link
prediction: does a person's description of the rule tell you which grid they
will draw?

Split by PARTICIPANT, so no person appears in both train and test, and the
supervision edges (Submission -> SolutionGroup) for held-out people are removed
from the message-passing graph. Group sizes and group keyword profiles are
computed on the training split only.

Baselines, in increasing order of what they know:
  majority    always pick the largest answer group on that puzzle (train counts)
  profile     cosine between the person's keyword vector and each group's mean
              keyword vector (train only) -- the graph's own statistics, no fitting
  GNN         heterogeneous GraphSAGE over the knowledge graph

Only puzzles with two or more candidate groups are scored; the rest are free.

Usage: python etl/link_prediction.py [--epochs 300] [--seeds 5]
"""
from __future__ import annotations
import argparse, json, os, collections
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import kuzu
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

HERE = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(HERE, "..", "cogarc_kg.kuzu")
OUT  = os.path.join(HERE, "..", "queries", "06_link_prediction.json")
KW16 = ["Color","Contact","Direction","Distance","Fill","Number","Object","Order",
        "Path","Pattern","Position","Reverse","Rotation","Shape","Size","Symmetry"]


def load():
    conn = kuzu.Connection(kuzu.Database(DB))
    q = lambda s: conn.execute(s).get_as_df()
    sub = q("""MATCH (pa:Participant)-[:MADE]->(s:Submission)-[:ON_PROBLEM]->(p:Problem),
                     (s)-[:LANDED_IN]->(g:SolutionGroup)
               WHERE p.is_attention_check = false
               RETURN s.id AS sid, pa.id AS pid, p.kw_index AS puz, g.id AS gid""")
    kw  = q("""MATCH (pa:Participant)-[:GAVE]->(kr:KeywordReport)-[:ABOUT]->(p:Problem),
                     (kr)-[:SELECTED]->(k:Keyword)
               WHERE p.is_attention_check = false
               RETURN pa.id AS pid, p.kw_index AS puz, k.name AS kw, 1 AS r""")
    grp = q("""MATCH (g:SolutionGroup)-[:FOR_PROBLEM]->(p:Problem)
               WHERE p.is_attention_check = false
               RETURN g.id AS gid, p.kw_index AS puz, g.is_correct AS ok,
                      g.n_submitters AS n""")
    return sub, kw, grp


def build_frames(sub, kw, grp, min_submitters=2):
    # Candidates are the answers the study recognises: the correct one plus any
    # wrong grid two or more people drew. One-off answers are not a class anyone
    # could predict, and including them turns the task into "pick the biggest".
    grp = grp[(grp.n >= min_submitters) | (grp.ok)].copy()
    cand = grp.groupby("puz")["gid"].apply(list).to_dict()
    keep = {g for v in cand.values() for g in v}
    sub = sub[sub.gid.isin(keep)].copy()
    sub = sub[sub.puz.map(lambda p: len(cand.get(p, []))) >= 2].copy()

    bag = collections.defaultdict(lambda: np.zeros(len(KW16), dtype=np.float32))
    for _, r in kw.iterrows():
        if r.kw in KW16:
            bag[(r.pid, r.puz)][KW16.index(r.kw)] = 1.0
    sub["x"] = [bag[(r.pid, r.puz)] for _, r in sub.iterrows()]
    sub = sub[[v.sum() > 0 for v in sub.x]].copy()     # needs a description to predict from
    return sub, cand


def splits(sub, seed):
    rng = np.random.default_rng(seed)
    people = np.array(sorted(sub.pid.unique())); rng.shuffle(people)
    n = len(people); tr = set(people[:int(.7*n)]); va = set(people[int(.7*n):int(.85*n)])
    which = lambda p: "train" if p in tr else ("val" if p in va else "test")
    return sub.assign(split=sub.pid.map(which))


def evaluate(rank_fn, df, cand):
    hits, rr = [], []
    for _, r in df.iterrows():
        cs = cand[r.puz]
        order = sorted(cs, key=lambda g: -rank_fn(r, g))
        pos = order.index(r.gid) + 1
        hits.append(pos == 1); rr.append(1.0 / pos)
    return dict(hit1=float(np.mean(hits)), mrr=float(np.mean(rr)), n=int(len(df)))


class Net(torch.nn.Module):
    """Two relational GraphSAGE layers, one SAGEConv per edge type, summed."""
    def __init__(self, metadata, h=48):
        super().__init__()
        mk = lambda: HeteroConv({et: SAGEConv((-1, -1), h) for et in metadata[1]}, aggr="sum")
        self.c1, self.c2 = mk(), mk()
    def forward(self, x, ei):
        h = {k: F.relu(v) for k, v in self.c1(x, ei).items()}
        return self.c2(h, ei)


def run_gnn(sub, cand, grp, seed, epochs):
    torch.manual_seed(seed)
    pids = sorted(sub.pid.unique()); puzs = sorted(sub.puz.unique())
    gids = sorted({g for v in cand.values() for g in v})
    sids = list(sub.sid)
    iP, iZ, iG, iS = ({v: i for i, v in enumerate(a)} for a in (pids, puzs, gids, sids))

    d = HeteroData()
    d["sub"].x  = torch.tensor(np.stack(sub.x.values))
    d["kw"].x   = torch.eye(len(KW16))
    d["puz"].x  = torch.eye(len(puzs))
    d["grp"].x  = torch.eye(len(gids))
    d["per"].x  = torch.zeros(len(pids), 8)

    sidx = torch.tensor([iS[s] for s in sub.sid])
    d["sub","on","puz"].edge_index  = torch.stack([sidx, torch.tensor([iZ[p] for p in sub.puz])])
    d["sub","by","per"].edge_index  = torch.stack([sidx, torch.tensor([iP[p] for p in sub.pid])])
    kr, kc = [], []
    for j, (_, r) in enumerate(sub.iterrows()):
        for k in np.nonzero(r.x)[0]: kr.append(j); kc.append(int(k))
    d["sub","uses","kw"].edge_index = torch.tensor([kr, kc])
    gp = [(iG[r.gid], iZ[r.puz]) for _, r in grp.iterrows() if r.gid in iG and r.puz in iZ]
    d["grp","of","puz"].edge_index  = torch.tensor(list(zip(*gp)))
    # supervision edges: TRAIN submissions only
    trn = sub[sub.split == "train"]
    d["sub","landed","grp"].edge_index = torch.stack([
        torch.tensor([iS[s] for s in trn.sid]), torch.tensor([iG[g] for g in trn.gid])])
    data = d.clone()
    for st, rel, dt in list(data.edge_types):
        src, dst = data[st, rel, dt].edge_index
        data[dt, "rev_" + rel, st].edge_index = torch.stack([dst, src])

    model = Net(data.metadata())
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    cand_idx = {p: torch.tensor([iG[g] for g in cand[p]]) for p in puzs}
    def batch(df):
        rows = [(iS[r.sid], cand_idx[r.puz], cand[r.puz].index(r.gid)) for _, r in df.iterrows()]
        return rows
    B = {s: batch(sub[sub.split == s]) for s in ("train", "val", "test")}

    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        z = model(data.x_dict, data.edge_index_dict)
        loss = 0.0
        for si, ci, y in B["train"]:
            logits = z["grp"][ci] @ z["sub"][si]
            loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([y]))
        (loss / max(1, len(B["train"]))).backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                z = model(data.x_dict, data.edge_index_dict)
                acc = np.mean([int((z["grp"][ci] @ z["sub"][si]).argmax().item() == y)
                               for si, ci, y in B["val"]]) if B["val"] else 0
            if acc > best: best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        z = model(data.x_dict, data.edge_index_dict)
        hits, rr = [], []
        for si, ci, y in B["test"]:
            order = (z["grp"][ci] @ z["sub"][si]).argsort(descending=True).tolist()
            pos = order.index(y) + 1
            hits.append(pos == 1); rr.append(1 / pos)
    return dict(hit1=float(np.mean(hits)), mrr=float(np.mean(rr)), n=len(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    sub0, kw, grp = load()
    sub0, cand = build_frames(sub0, kw, grp)
    print(f"[data] {len(sub0)} submissions with a description, "
          f"{sub0.pid.nunique()} people, {sub0.puz.nunique()} puzzles with >=2 answers")
    print(f"       mean answers to choose between "
          f"{np.mean([len(v) for p,v in cand.items() if p in set(sub0.puz)]):.1f}")

    res = collections.defaultdict(list)
    for seed in range(a.seeds):
        sub = splits(sub0, seed)
        trn, tst = sub[sub.split == "train"], sub[sub.split == "test"]
        size = trn.groupby("gid").size().to_dict()
        prof = {g: np.mean(np.stack(d.x.values), 0) for g, d in trn.groupby("gid")}
        res["majority"].append(evaluate(lambda r, g: size.get(g, 0), tst, cand))
        def cos(r, g):
            v = prof.get(g)
            if v is None or not v.any() or not r.x.any(): return -1
            return float(r.x @ v / (np.linalg.norm(r.x) * np.linalg.norm(v)))
        res["profile"].append(evaluate(cos, tst, cand))
        res["gnn"].append(run_gnn(sub, cand, grp, seed, a.epochs))
        print(f"  seed {seed}: " + "  ".join(
            f"{k} hit@1={res[k][-1]['hit1']:.3f}" for k in ("majority","profile","gnn")))

    print(f"\n{'model':<10} {'hit@1':>14} {'MRR':>14}   (mean +- sd over {a.seeds} seeds)")
    summary = {}
    for k in ("majority", "profile", "gnn"):
        h = np.array([r["hit1"] for r in res[k]]); m = np.array([r["mrr"] for r in res[k]])
        summary[k] = dict(hit1=float(h.mean()), hit1_sd=float(h.std()),
                          mrr=float(m.mean()), mrr_sd=float(m.std()),
                          n_test=int(res[k][0]["n"]))
        print(f"{k:<10} {h.mean():>8.3f} ± {h.std():.3f} {m.mean():>8.3f} ± {m.std():.3f}")
    summary["setup"] = dict(n_submissions=int(len(sub0)), n_people=int(sub0.pid.nunique()),
                            n_puzzles=int(sub0.puz.nunique()), seeds=a.seeds,
                            mean_candidates=float(np.mean(
                                [len(v) for p, v in cand.items() if p in set(sub0.puz)])))
    json.dump(summary, open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
