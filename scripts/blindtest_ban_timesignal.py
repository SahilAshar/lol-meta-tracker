"""Blind test of the meta-aware transformer on the frozen EWC July-2026 set.

HONESTY FLAG: this is the project's THIRD evaluation against the EWC test set
(v0.8 on 2026-07-20 was the first, v0.8.1 the second). It was gated behind a
GO on validation (timesignal_rung.json: ban val top-1 +1.93 pts, 5/5 seeds,
loss CI excluding 0) and run only with Sahil's explicit approval on
2026-07-23. Scored on test exactly once; no further iteration against this
test set regardless of the result.

Protocol mirrors train_draft_model_v08.py's transformer lineage exactly —
vocab from the full multi-year data, games built over the full sequence
table, 5-seed mean-softmax-probability ensemble (seeds 16/17/42/7/23),
trained on the train split with early stopping on val — plus the ban
time-signal rung's meta injection (experiment_v11_ban_timesignal.py): 28-day
trailing global pick/ban rates, standardized on train games only, read
through 4 slot-type-gated output-layer scalars. Test-window rates come from
the same causal trailing windows as every other row.

Comparison baselines are the stored draft_model_metrics_v08.json blocks
(v0.8_transformer, v0.7_refit_multi, v0.8.1_blend_pertype) — same split,
same vocab construction, same metric path. split_dates/load_multi are
inlined because experiment_v08 imports lightgbm at module top (libomp
missing on this machine).

Output: data/processed/timesignal_blindtest.json
        data/processed/expcache_timesignal_blind/seed<S>.npz (eval probs)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_transformer import (
    Config, Vocab, attach_scores, build_games, probs_for, to_tensors,
)
from experiment_v11_ban_timesignal import (
    SEEDS, build_meta_matrix, topk_accuracy, train_model,
)

VAL_DAYS = 14
CACHE_DIR = DATA_PROCESSED / "expcache_timesignal_blind"
CFG_BASE = dict(d_model=192, n_layers=4, n_heads=6)
# Rung anchor: meta seed-16 best val loss (expcache_timesignal). Reproducing
# it here confirms the full-data vocab/games build changed nothing upstream.
META_CANARY_SEED16 = 3.4623


def main() -> None:
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate",
                 "label", "pick_rate", "ban_rate"],
    )
    cand["date"] = pd.to_datetime(cand["date"])

    # experiment_v08.split_dates, inlined (lightgbm import trap)
    def test_mask(df: pd.DataFrame) -> pd.Series:
        return ((df.league == "EWC") & (df.date.dt.month == 7)
                & (df.date.dt.year == 2026))

    cutoff = cand.loc[test_mask(cand), "date"].min()
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)

    # Full-data vocab and games, exactly as train_draft_model_v08.py.
    vocab = Vocab(list(cand.candidate.unique()), list(seq.champion.unique()))
    assert vocab.size == 171, (
        f"vocab {vocab.size} != 171: test era introduced new champions and "
        "stored v0.8 numbers may not be comparable — stop and review"
    )
    games = build_games(seq, vocab)
    n_leagues = len(games.attrs["leagues"])

    seq_test_gids = set(seq.loc[test_mask(seq), "gameid"])
    pre = games[~games.gameid.isin(seq_test_gids) & (games.date < cutoff)]
    g_train = pre[pre.date < val_start].reset_index(drop=True)
    g_val = pre[pre.date >= val_start].reset_index(drop=True)
    g_test = games[games.gameid.isin(seq_test_gids)].reset_index(drop=True)
    assert (len(g_train), len(g_val)) == (5043, 54)

    candidate_set = set(vocab.champs)
    expected_coverage = seq.groupby("gameid").series_prior.first().map(
        lambda p: len(candidate_set)
        - (len({c for c in p.split("|") if c} & candidate_set) if p else 0)
    )
    meta_all, meta_stats = build_meta_matrix(
        cand, games, vocab, val_start, expected_coverage
    )
    all_pos = {g: i for i, g in enumerate(games.gameid)}

    def tensors_for(g: pd.DataFrame) -> dict[str, torch.Tensor]:
        ix = np.array([all_pos[gid] for gid in g.gameid])
        return {**to_tensors(g), "meta": torch.from_numpy(meta_all[ix])}

    t_train, t_val = tensors_for(g_train), tensors_for(g_val)

    # Score only what we report: val + test games.
    g_eval = pd.concat([g_val, g_test], ignore_index=True)
    t_eval = tensors_for(g_eval)
    eval_pos = {g: i for i, g in enumerate(g_eval.gameid)}
    val_rows = cand[~test_mask(cand) & (cand.date >= val_start)
                    & (cand.date < cutoff)].copy()
    test_rows = cand[test_mask(cand)].copy()
    n_val = val_rows.groupby(["gameid", "seq"]).ngroups
    n_test = test_rows.groupby(["gameid", "seq"]).ngroups
    print(f"cutoff {cutoff.date()}: train {len(g_train)}g, val {len(g_val)}g/"
          f"{n_val}d, test {len(g_test)}g/{n_test}d, vocab {vocab.size}")
    assert n_val == 1080
    assert n_test == 1000, f"test set changed: {n_test} != 1000 decisions"

    CACHE_DIR.mkdir(exist_ok=True)
    seed_probs, seed_meta = [], []
    for seed in SEEDS:
        cache = CACHE_DIR / f"seed{seed}.npz"
        if cache.exists():
            z = np.load(cache)
            seed_probs.append(z["probs"])
            print(f"[cached] seed={seed} val_loss={float(z['best_val']):.4f}")
            seed_meta.append({"seed": seed, "best_val_loss": round(float(z["best_val"]), 4),
                              "epochs": int(z["epochs"])})
            continue
        cfg = Config(**CFG_BASE, seed=seed)
        model, best_val, epochs, wall = train_model(
            cfg, t_train, t_val, vocab.size, n_leagues,
        )
        probs = probs_for(model, t_eval).numpy()
        np.savez_compressed(cache, probs=probs, best_val=best_val, epochs=epochs)
        seed_probs.append(probs)
        seed_meta.append({"seed": seed, "best_val_loss": round(best_val, 4),
                          "epochs": epochs})
        print(f"seed={seed} val_loss={best_val:.4f} ({epochs} epochs, "
              f"{wall:.0f}s)", flush=True)
        if seed == 16 and round(best_val, 4) != META_CANARY_SEED16:
            print(f"WARNING: seed-16 meta val loss {best_val:.4f} != rung "
                  f"anchor {META_CANARY_SEED16} — full-data build drifted, "
                  "review before trusting the comparison")

    mean_probs = np.mean(seed_probs, axis=0)
    val_rows["score"] = attach_scores(val_rows, mean_probs, eval_pos, vocab)
    test_rows["score"] = attach_scores(test_rows, mean_probs, eval_pos, vocab)
    scores = {"val": topk_accuracy(val_rows, "score"),
              "test_ewc_main": topk_accuracy(test_rows, "score")}

    stored = json.loads(
        (DATA_PROCESSED / "draft_model_metrics_v08.json").read_text()
    )
    base_tf = stored["v0.8_transformer"]["test_ewc_main"]["model"]
    delta = {
        split: round(
            scores["test_ewc_main"][split]["top1"] - base_tf[split]["top1"], 4
        )
        for split in ("all", "picks", "bans")
    }

    out = {
        "note": ("Blind test of the meta-aware (ban time-signal) transformer "
                 "on the frozen EWC July-2026 set. THIRD look at this test "
                 "set; gated on the timesignal_rung GO and Sahil's explicit "
                 "approval; scored once, no further test iteration."),
        "protocol": ("train_draft_model_v08.py transformer lineage + rung "
                     "meta injection: 5-seed mean-softmax ensemble, train "
                     "split fit, val early-stop, full-data vocab"),
        "runs": seed_meta,
        "meta_transformer": scores,
        "test_top1_delta_vs_stored_v0.8_transformer": delta,
        "stored_reference": {
            "v0.8_transformer_test": base_tf,
            "v0.7_refit_multi_test":
                stored["v0.7_refit_multi"]["test_ewc_main"]["model"],
            "v0.8.1_blend_pertype_test": {
                k: stored["v0.8.1_blend_pertype"]["test_ewc_main"]["model"][k]
                for k in ("all", "picks", "bans")
            },
        },
        "provenance": {
            "seeds": SEEDS, "config": Config(**CFG_BASE).tag(),
            "standardization": {"population": "train games only", **meta_stats},
            "split_cutoff": str(cutoff.date()), "val_days": VAL_DAYS,
            "device": "cpu", "rung": "timesignal_rung.json (GO)",
        },
    }
    (DATA_PROCESSED / "timesignal_blindtest.json").write_text(
        json.dumps(out, indent=2)
    )
    t = scores["test_ewc_main"]
    print(f"\nTEST (n={t['all']['n']}): "
          f"all {t['all']['top1']:.3f}  picks {t['picks']['top1']:.3f}  "
          f"bans {t['bans']['top1']:.3f}")
    print(f"stored v0.8 transformer test: all {base_tf['all']['top1']:.3f}  "
          f"picks {base_tf['picks']['top1']:.3f}  bans {base_tf['bans']['top1']:.3f}")
    print(f"delta vs stored: {delta}")


if __name__ == "__main__":
    main()
