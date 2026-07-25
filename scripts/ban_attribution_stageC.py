"""Stage C of the denial-vector attribution study: pick-denial counterfactual.

Spec: docs/2026-07-25-ban-attribution-spec.md. For each val-game pick, ask
the meta transformer ensemble how much the OPPONENT wanted that champion at
their immediately next pick slot, had it been left on the board:

  edit: champ token at the pick's slot -> MISSED (the pick "didn't happen"),
        availability at the opponent's next pick slot u re-opens the champion
  read: ensemble softmax prob of the champion at u; denial score = its
        percentile among the (edited) availability pool at u

v0 approximations, flagged: everything else stays teacher-forced (the actual
later draft, including phase-2 bans, is held fixed — hence "immediately next
opponent pick slot" only); a MISSED token at a pick slot is out-of-
distribution for the model. R-P5 (slot 20) has no later opponent pick and is
unscored. This is the seed of decision valuation (denial ~ demand x
strength), not the valuation itself.

Population: 54 val games x 9 scorable picks = 486. VAL ONLY; the frozen EWC
test set is never touched. Needs expcache_attr/meta_seed<S>.pt weights
(ban_attribution_train.py).

Output: data/processed/ban_attribution.json (stage_C section)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_dataset import DRAFT_SEQUENCE
from draft_transformer import MISSED, Config, Vocab, build_games, to_tensors
from experiment_v11_ban_timesignal import (
    SEEDS, MetaDraftTransformer, build_meta_matrix,
)

VAL_DAYS = 14
CACHE_DIR = DATA_PROCESSED / "expcache_attr"
CFG_BASE = dict(d_model=192, n_layers=4, n_heads=6)

PICK_SLOTS = [i for i, (_, d, _, _) in enumerate(DRAFT_SEQUENCE) if d == "pick"]
SLOT_LABEL = {
    i: f"{s[0]}-P{o}" for i, (_, d, s, o) in enumerate(DRAFT_SEQUENCE)
    if d == "pick"
}


def next_opp_pick(t: int) -> int | None:
    side = DRAFT_SEQUENCE[t][2]
    for u in PICK_SLOTS:
        if u > t and DRAFT_SEQUENCE[u][2] != side:
            return u
    return None


def main() -> None:
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate",
                 "label", "pick_rate", "ban_rate"],
    )
    cand["date"] = pd.to_datetime(cand["date"])

    def test_mask(df: pd.DataFrame) -> pd.Series:
        return ((df.league == "EWC") & (df.date.dt.month == 7)
                & (df.date.dt.year == 2026))

    cutoff = cand.loc[test_mask(cand), "date"].min()
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)

    vocab = Vocab(list(cand.candidate.unique()), list(seq.champion.unique()))
    assert vocab.size == 171
    games = build_games(seq, vocab)
    n_leagues = len(games.attrs["leagues"])
    seq_test_gids = set(seq.loc[test_mask(seq), "gameid"])
    pre = games[~games.gameid.isin(seq_test_gids) & (games.date < cutoff)]
    g_val = pre[pre.date >= val_start].reset_index(drop=True)
    assert len(g_val) == 54

    candidate_set = set(vocab.champs)
    expected_coverage = seq.groupby("gameid").series_prior.first().map(
        lambda p: len(candidate_set)
        - (len({c for c in p.split("|") if c} & candidate_set) if p else 0)
    )
    meta_all, _ = build_meta_matrix(
        cand, games, vocab, val_start, expected_coverage
    )
    all_pos = {g: i for i, g in enumerate(games.gameid)}
    val_ix = np.array([all_pos[gid] for gid in g_val.gameid])
    t_val = {**to_tensors(g_val), "meta": torch.from_numpy(meta_all[val_ix])}

    # One virtual game per scorable (game, pick slot): token at t -> MISSED,
    # champion re-opened at the opponent's next pick slot u.
    edits = []  # (game_row, t, u, champ_id)
    for gi in range(len(g_val)):
        champ = g_val.champ.iloc[gi]
        target = g_val.target.iloc[gi]
        for t in PICK_SLOTS:
            u = next_opp_pick(t)
            if u is None or target[t] == -100:
                continue
            edits.append((gi, t, u, int(champ[t])))
    print(f"{len(edits)} counterfactual pick edits over {len(g_val)} val games")

    gidx = torch.tensor([e[0] for e in edits])
    batch = {k: v[gidx].clone() for k, v in t_val.items()}
    for i, (gi, t, u, x) in enumerate(edits):
        batch["prev"][i, t + 1] = MISSED
        batch["avail"][i, u, x] = True

    id2champ = {v: k for k, v in vocab.id_of.items()}
    seed_probs = []
    for s in SEEDS:
        model = MetaDraftTransformer(Config(**CFG_BASE, seed=s), vocab.size,
                                     n_leagues)
        model.load_state_dict(
            torch.load(CACHE_DIR / f"meta_seed{s}.pt", weights_only=True)
        )
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(edits), 256):
                b = {k: v[i:i + 256] for k, v in batch.items()}
                logits = model(b).masked_fill(~b["avail"], float("-inf"))
                outs.append(torch.softmax(logits, -1).nan_to_num(0.0))
        seed_probs.append(torch.cat(outs).numpy())
        print(f"seed {s} scored", flush=True)
    probs = np.mean(seed_probs, axis=0)

    recs = []
    for i, (gi, t, u, x) in enumerate(edits):
        pool = batch["avail"][i, u].numpy()
        p = probs[i, u]
        px = p[x]
        n_pool = int(pool.sum())
        p_pool = p[pool]
        pct = 100.0 * float((p_pool <= px).sum()) / n_pool
        recs.append({
            "gameid": g_val.gameid.iloc[gi], "slot": t,
            "label": SLOT_LABEL[t], "champion": id2champ[x],
            "opp_demand_prob": float(px), "demand_pct": pct,
            "demand_rank": int(1 + (p_pool > px).sum()), "n_pool": n_pool,
        })
    df = pd.DataFrame(recs)

    def summarize(g: pd.DataFrame) -> dict:
        return {
            "n": len(g),
            "mean_demand_pct": round(float(g.demand_pct.mean()), 2),
            "median_demand_pct": round(float(g.demand_pct.median()), 2),
            "share_opp_top5": round(float((g.demand_rank <= 5).mean()), 4),
            "share_above_p90": round(float((g.demand_pct >= 90).mean()), 4),
            "mean_demand_prob": round(float(g.opp_demand_prob.mean()), 4),
        }

    slot_order = sorted(df.slot.unique())
    out = {
        "note": ("Stage C: counterfactual opponent demand for each val pick "
                 "at the opponent's immediately-next pick slot, meta "
                 "transformer 5-seed ensemble. Teacher-forced elsewhere; "
                 "MISSED-token edit is mildly out-of-distribution; seed of "
                 "valuation, not valuation."),
        "n_scored_picks": len(df),
        "overall": summarize(df),
        "by_slot": {
            f"{SLOT_LABEL[s]} (slot {s + 1})": summarize(df[df.slot == s])
            for s in slot_order
        },
        "top_denial_picks": df.nlargest(10, "demand_pct")[
            ["gameid", "label", "champion", "demand_pct", "opp_demand_prob"]
        ].to_dict("records"),
    }
    path = DATA_PROCESSED / "ban_attribution.json"
    existing = json.loads(path.read_text())
    existing["stage_C"] = out
    path.write_text(json.dumps(existing, indent=2))

    print(f"\noverall: {out['overall']}")
    for k, v in out["by_slot"].items():
        print(f"  {k}: mean pct {v['mean_demand_pct']}, "
              f"opp-top5 {v['share_opp_top5']:.0%}, >p90 {v['share_above_p90']:.0%}")


if __name__ == "__main__":
    main()
