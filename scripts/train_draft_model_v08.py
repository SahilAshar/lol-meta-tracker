"""Train the v0.8 draft models on multi-year data and blind-score once.

v0.8 = a small causal transformer over the draft sequence with learned,
weight-tied champion embeddings (draft_transformer.py), trained on 2024-2026
data, 5-seed mean-probability ensemble. The configuration below (CHOSEN /
BLEND_W) is promoted from experiment_v08.py, which selects on the validation
split only; this script performs the single blind test on the EWC July 2026
main event.

For apples-to-apples comparison every feature-set lineage is refit on the SAME
multi-year train split and scored on the SAME test decisions:
  - v0.7 features, full 10-model GBM ensemble (5 seeds x clf/ranker)
  - v0.6 / v0.5 features, single classifier (their original config)
  - v0 baselines (trailing meta rates, team habit) via score_splits
The 2026-only v0.7 numbers from draft_model_metrics_v07.json are copied in
verbatim for reference — note their candidate sets differ marginally (the
multi-year build ranks every champion picked since 2024, not just 2026).

Output: data/processed/draft_model_metrics_v08.json
        data/processed/draft_model_v08_seed16.pt (first-seed weights + vocab)
        data/processed/champion_embeddings_v08.npz (for chart_embeddings_v08.py)
"""

from __future__ import annotations

import gc
import json
import platform

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_transformer import (
    Config, Vocab, attach_scores, build_games, probs_for, to_tensors, train_model,
)
from experiment_v08 import load_multi, split_dates
from train_draft_model import (
    FEATURES, FEATURES_V05, FEATURES_V06, SEEDS,
    ensemble_score, fit_clf, fit_ranker, score_splits,
)

# Promoted from experiment_v08.py (val-only selection, 2026-07-20) — do not
# tune here. d192x4 won the config sweep on val top-1; the 0.25-transformer
# rank-average blend dominated both parents on val top-1/top-3 (transformer
# carries picks, GBM carries bans).
CHOSEN = Config(d_model=192, n_layers=4, n_heads=6)
BLEND_W: float | None = 0.25  # transformer share of rank-average blend with v0.7 GBM
VERSION = "v0.8"


def main() -> None:
    ds, seq = load_multi()
    is_test, cutoff, val_start = split_dates(ds)
    pre = ds[~is_test & (ds.date < cutoff)]
    train, val, test = pre[pre.date < val_start], pre[pre.date >= val_start], ds[is_test]
    print(f"cutoff (first EWC 2026 main-event game): {cutoff}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}: {part.gameid.nunique()} games, "
              f"{part.groupby(['gameid', 'seq']).ngroups} decisions")

    vocab = Vocab(list(ds.candidate.unique()), list(seq.champion.unique()))
    games = build_games(seq, vocab)
    leagues = games.attrs["leagues"]
    n_leagues = len(leagues)
    g_train = games[games.gameid.isin(train.gameid.unique())].reset_index(drop=True)
    g_val = games[games.gameid.isin(val.gameid.unique())].reset_index(drop=True)
    t_train, t_val = to_tensors(g_train), to_tensors(g_val)
    all_pos = {g: i for i, g in enumerate(games.gameid)}
    t_all = to_tensors(games)

    # --- transformer ensemble ---
    tf_models = []
    for s in SEEDS:
        cfg = Config(**{**CHOSEN.__dict__, "seed": s})
        tf_models.append(train_model(cfg, t_train, t_val, vocab.size, n_leagues,
                                     verbose=False))
    tf_probs = np.mean([probs_for(m, t_all).numpy() for m in tf_models], axis=0)

    def tf_score(part: pd.DataFrame) -> np.ndarray:
        return attach_scores(part, tf_probs, all_pos, vocab)

    # Free everything the GBM stage doesn't need — 16GB box, sklearn casts
    # the 13M-row feature matrix to float64 internally.
    del ds, pre, t_train, t_val, t_all, g_train, g_val, games, seq
    gc.collect()

    # --- GBM lineages refit on the same multi-year train ---
    gbm = {"clf": [fit_clf(FEATURES, train, s) for s in SEEDS],
           "rank": [fit_ranker(FEATURES, train, s) for s in SEEDS]}

    def gbm_score(part: pd.DataFrame) -> np.ndarray:
        return ensemble_score(gbm, FEATURES, part)

    results = {
        "version": VERSION, "cutoff": str(cutoff),
        "data": "multi-year 2024-2026", "platform": platform.platform(),
        "transformer": {"config": CHOSEN.tag(), "seeds": SEEDS,
                        "ensemble": "mean softmax probability"},
        "blend_w_transformer": BLEND_W,
    }
    results["v0.8_transformer"] = score_splits(tf_score, val, test)
    results["v0.7_refit_multi"] = score_splits(gbm_score, val, test)

    if BLEND_W is not None:
        def blend_score(part: pd.DataFrame) -> np.ndarray:
            t = pd.Series(tf_score(part)).rank(pct=True).to_numpy()
            g = pd.Series(gbm_score(part)).rank(pct=True).to_numpy()
            return BLEND_W * t + (1 - BLEND_W) * g
        results["v0.8_blend"] = score_splits(blend_score, val, test)

    for tag, feats in [("v0.5", FEATURES_V05), ("v0.6", FEATURES_V06)]:
        m = fit_clf(feats, train, 16)
        results[f"{tag}_refit_multi"] = score_splits(
            lambda part, m=m, f=feats: m.predict_proba(part[f])[:, 1], val, test)

    v07_path = DATA_PROCESSED / "draft_model_metrics_v07.json"
    if v07_path.exists():
        stored = json.loads(v07_path.read_text())
        results["v0.7_2026only_stored"] = {
            "note": "trained on 2026 data only; candidate sets differ marginally",
            "cutoff": stored.get("cutoff"),
            "val": stored.get("val"), "test_ewc_main": stored.get("test_ewc_main"),
        }

    out = DATA_PROCESSED / "draft_model_metrics_v08.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in
                      ("v0.8_transformer", "v0.7_refit_multi")}, indent=2))

    torch.save({"state_dict": tf_models[0].state_dict(),
                "config": CHOSEN.__dict__, "vocab_champs": vocab.champs,
                "vocab_ids": vocab.id_of, "leagues": leagues},
               DATA_PROCESSED / "draft_model_v08_seed16.pt")
    emb = tf_models[0].champ_emb.weight.detach().numpy()
    ids = np.array([vocab.id_of[c] for c in vocab.champs])
    np.savez(DATA_PROCESSED / "champion_embeddings_v08.npz",
             embeddings=emb[ids], champions=np.array(vocab.champs))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
