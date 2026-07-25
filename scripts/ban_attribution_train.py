"""Train the transformer ensembles Stage B/C of the attribution study need.

Spec: docs/2026-07-25-ban-attribution-spec.md (Stages B, C). Two 5-seed
ensembles in the exact blind-test lineage (full-data vocab, games over the
full sequence table, train-split fit, val early-stop):

  baseline — no meta tensor; reproduces the v0.8 production transformer
             bit-for-bit (canary: seed-16 best val loss 3.5855). Needed
             because the blind-test cache only holds the meta condition.
  meta     — the ban time-signal model. Val probs are bit-identical to the
             blind cache (asserted); retrained here only to persist WEIGHTS,
             which Stage C's counterfactual forward passes require.

VAL ONLY: eval probs are computed for the 54 val games. The frozen EWC test
games are never scored — g_test is built solely to exclude its gameids from
the train/val pool, exactly as the blind test did.

Output: data/processed/expcache_attr/{condition}_seed<S>.npz  (val probs)
        data/processed/expcache_attr/meta_seed<S>.pt          (weights)
All outputs are untracked (npz not staged, *.pt gitignored).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_transformer import Config, Vocab, build_games, probs_for, to_tensors
from experiment_v11_ban_timesignal import SEEDS, build_meta_matrix, train_model

VAL_DAYS = 14
CACHE_DIR = DATA_PROCESSED / "expcache_attr"
BLIND_CACHE = DATA_PROCESSED / "expcache_timesignal_blind"
CFG_BASE = dict(d_model=192, n_layers=4, n_heads=6)
BASELINE_CANARY = 3.5855  # v0.8 lineage, seed 16
META_CANARY = 3.4623      # rung/blind lineage, seed 16


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
    g_train = pre[pre.date < val_start].reset_index(drop=True)
    g_val = pre[pre.date >= val_start].reset_index(drop=True)
    assert (len(g_train), len(g_val)) == (5043, 54)

    candidate_set = set(vocab.champs)
    expected_coverage = seq.groupby("gameid").series_prior.first().map(
        lambda p: len(candidate_set)
        - (len({c for c in p.split("|") if c} & candidate_set) if p else 0)
    )
    meta_all, _ = build_meta_matrix(
        cand, games, vocab, val_start, expected_coverage
    )
    all_pos = {g: i for i, g in enumerate(games.gameid)}

    def meta_for(g: pd.DataFrame) -> torch.Tensor:
        ix = np.array([all_pos[gid] for gid in g.gameid])
        return torch.from_numpy(meta_all[ix])

    t_train, t_val = to_tensors(g_train), to_tensors(g_val)
    m_train, m_val = meta_for(g_train), meta_for(g_val)

    CACHE_DIR.mkdir(exist_ok=True)
    for condition in ("baseline", "meta"):
        for seed in SEEDS:
            cache = CACHE_DIR / f"{condition}_seed{seed}.npz"
            if cache.exists():
                z = np.load(cache)
                print(f"[cached] {condition} seed={seed} "
                      f"val_loss={float(z['best_val']):.4f}")
                continue
            if condition == "meta":
                tr_t = {**t_train, "meta": m_train}
                va_t = {**t_val, "meta": m_val}
            else:
                tr_t, va_t = t_train, t_val
            cfg = Config(**CFG_BASE, seed=seed)
            model, best_val, epochs, wall = train_model(
                cfg, tr_t, va_t, vocab.size, n_leagues,
            )
            probs = probs_for(model, va_t).numpy()
            np.savez_compressed(
                cache, probs=probs, best_val=best_val, epochs=epochs
            )
            print(f"{condition} seed={seed} val_loss={best_val:.4f} "
                  f"({epochs} epochs, {wall:.0f}s)", flush=True)
            if condition == "baseline" and seed == 16 and \
                    round(best_val, 4) != BASELINE_CANARY:
                print(f"WARNING: baseline seed-16 val loss {best_val:.4f} != "
                      f"canary {BASELINE_CANARY} — lineage drifted, review")
            if condition == "meta":
                torch.save(
                    model.state_dict(), CACHE_DIR / f"meta_seed{seed}.pt"
                )
                if seed == 16 and round(best_val, 4) != META_CANARY:
                    print(f"WARNING: meta seed-16 val loss {best_val:.4f} != "
                          f"canary {META_CANARY} — lineage drifted, review")
                blind = BLIND_CACHE / f"seed{seed}.npz"
                if blind.exists():
                    bz = np.load(blind)["probs"][: len(g_val)]
                    if not np.allclose(bz, probs, atol=1e-5):
                        print(f"WARNING: meta seed={seed} val probs differ "
                              "from blind cache — lineage drifted, review")
    print("done")


if __name__ == "__main__":
    main()
