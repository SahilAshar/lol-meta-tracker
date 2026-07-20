"""v0.8 val-only experiments: transformer configs, seed ensembling, GBM blend.

Same discipline as experiment_v07*.py: every number here is computed on the
validation split (last VAL_DAYS days of the train era); the EWC July 2026 test
set is never touched. The winning configuration gets promoted into
train_draft_model_v08.py and only then blind-tested once.

Compares:
  1. transformer configs (size / dropout / time-decay) at a single seed
  2. what a 3-seed mean-probability ensemble buys over a single seed
  3. rank-average blends of the transformer ensemble with a reduced v0.7 GBM
     ensemble (2 seeds x {clf, ranker}) at several weights

Requires the multi-year dataset: draft_dataset.py --years 2024 2025 2026.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_transformer import (
    Config, Vocab, attach_scores, build_games, probs_for, to_tensors, train_model,
)
from train_draft_model import FEATURES, VAL_DAYS, fit_clf, fit_ranker, topk_accuracy

BLEND_WEIGHTS = [0.25, 0.5, 0.75]  # transformer share of the blend
SEEDS3 = [16, 17, 42]


def split_dates(ds: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    """Test = EWC July 2026 only — the year guard matters now that EWC 2024/2025
    July games are in the multi-year data."""
    is_test = (ds.league == "EWC") & (ds.date.dt.month == 7) & (ds.date.dt.year == 2026)
    cutoff = ds.loc[is_test, "date"].min()
    return is_test, cutoff, cutoff - pd.Timedelta(days=VAL_DAYS)


def fmt(r: dict) -> str:
    return "  ".join(
        f"{s}:{r[s]['top1'] * 100:.1f}/{r[s]['top3'] * 100:.1f}/{r[s]['top5'] * 100:.1f}"
        for s in ("all", "picks", "bans")
    )


def main() -> None:
    ds = pd.read_parquet(DATA_PROCESSED / "draft_decisions_multi.parquet")
    ds["date"] = pd.to_datetime(ds["date"])
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])

    is_test, cutoff, val_start = split_dates(ds)
    pre = ds[~is_test & (ds.date < cutoff)]
    train = pre[pre.date < val_start]
    val = pre[pre.date >= val_start].copy()
    print(f"cutoff={cutoff}  train {train.gameid.nunique()}g  val {val.gameid.nunique()}g")

    vocab = Vocab(
        candidate_champs=list(ds.candidate.unique()),
        extra_champs=list(seq.champion.unique()),
    )
    games = build_games(seq, vocab)
    n_leagues = len(games.attrs["leagues"])
    g_train = games[games.gameid.isin(train.gameid.unique())].reset_index(drop=True)
    g_val = games[games.gameid.isin(val.gameid.unique())].reset_index(drop=True)
    print(f"vocab={vocab.size} ({len(vocab.champs)} candidates)  "
          f"games: train={len(g_train)} val={len(g_val)}")
    t_train, t_val = to_tensors(g_train), to_tensors(g_val)
    val_pos = {g: i for i, g in enumerate(g_val.gameid)}

    def val_metrics(probs: np.ndarray, col: str) -> dict:
        val[col] = attach_scores(val, probs, val_pos, vocab)
        return topk_accuracy(val, col)

    # Baseline reference on this val split.
    val["baseline_meta"] = np.where(val.is_ban == 1, val.presence, val.pick_rate)
    print(f"{'meta baseline':34s} {fmt(topk_accuracy(val, 'baseline_meta'))}")

    # --- 1. config sweep, single seed ---
    configs = [
        Config(d_model=64, n_layers=2, n_heads=4),
        Config(d_model=128, n_layers=3, n_heads=4),
        Config(d_model=128, n_layers=3, n_heads=4, dropout=0.2),
        Config(d_model=192, n_layers=4, n_heads=6),
        Config(d_model=128, n_layers=3, n_heads=4, time_decay_tau_days=365),
    ]
    probs_by_tag: dict[str, np.ndarray] = {}
    for cfg in configs:
        model = train_model(cfg, t_train, t_val, vocab.size, n_leagues, verbose=False)
        p = probs_for(model, t_val).numpy()
        probs_by_tag[cfg.tag()] = p
        print(f"{cfg.tag():34s} {fmt(val_metrics(p, 's'))}")

    # --- 2. seed ensemble on the best-looking base config ---
    best = max(probs_by_tag, key=lambda t: topk_accuracy(
        val.assign(s=attach_scores(val, probs_by_tag[t], val_pos, vocab)), "s"
    )["all"]["top1"])
    base = next(c for c in configs if c.tag() == best)
    print(f"\nseed ensemble on {base.tag()}:")
    seed_probs = [probs_by_tag[best]]
    for s in SEEDS3[1:]:
        cfg = Config(**{**base.__dict__, "seed": s})
        model = train_model(cfg, t_train, t_val, vocab.size, n_leagues, verbose=False)
        seed_probs.append(probs_for(model, t_val).numpy())
    ens = np.mean(seed_probs, axis=0)
    print(f"{'ens3 ' + base.tag():34s} {fmt(val_metrics(ens, 's'))}")

    # --- 3. blend with a reduced v0.7 GBM ensemble ---
    print("\nfitting reduced GBM ensemble (2 seeds x clf/ranker) for blend...")
    gbm_scores = []
    for s in [16, 17]:
        m = fit_clf(FEATURES, train, s)
        gbm_scores.append(m.predict_proba(val[FEATURES])[:, 1])
        r = fit_ranker(FEATURES, train, s)
        gbm_scores.append(r.predict(val[FEATURES]))
    gbm = np.mean(
        [pd.Series(x).rank(pct=True).to_numpy() for x in gbm_scores], axis=0
    )
    val["s"] = gbm
    print(f"{'GBM v0.7-feats (reduced ens4)':34s} {fmt(topk_accuracy(val, 's'))}")
    tr = pd.Series(attach_scores(val, ens, val_pos, vocab)).rank(pct=True).to_numpy()
    for w in BLEND_WEIGHTS:
        val["s"] = w * tr + (1 - w) * gbm
        print(f"{f'blend w_transformer={w}':34s} {fmt(topk_accuracy(val, 's'))}")


if __name__ == "__main__":
    main()
