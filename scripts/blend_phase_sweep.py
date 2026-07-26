"""Phase-aware ban blend sweep: per-type-x-phase weights, VAL ONLY, local.

Handoff: docs/2026-07-25-next-rungs-handoff.md Mission 1. Stage B of the
attribution study (docs/2026-07-25-ban-attribution-results.md) showed the
transformer's ban deficit is entirely B1-B3 while in B4-B5 it wins or ties;
production blends bans at w=0.0 (GBM-only), discarding the phase-2 edge.
This extends the v0.8.1 per-type sweep (blend_resweep_timesignal.py) to
per-type-x-phase for bans only: one weight for ban decisions with seq 1-6
(B1-3), one for seq 13-16 (B4-5). Picks keep a single weight.

Runs locally, no lightgbm import: the GBM side is the cached codespace refit
(gbm_val_scores.parquet, original gameid keys) and the transformer side the
blind-test ensemble caches (expcache_timesignal_blind, val slice only), both
bit-identical to what the re-sweep blended. Before reporting any new cell,
every all/picks/bans row of the local sweep is hard-asserted equal to
data/processed/blend_resweep_timesignal.json — if the reconstruction
drifted, this script dies rather than reporting numbers from a bent harness.

Honesty caveat (carried into the results doc): the B1-3 / B4-5 split was
chosen after seeing Stage B's cells on these same 540 val bans. It has
a-priori motivation ("phase 2 has context to attend over") and adds one df,
but this is weaker evidence than a pre-registered split.

VAL ONLY — the frozen test set is never touched. Any promotion into
train_draft_model_v08.py needs Sahil's approval.

Output: data/processed/blend_phase_sweep.json
"""

from __future__ import annotations

import json
import platform

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from draft_transformer import Vocab, attach_scores, build_games

VAL_DAYS = 14
SEEDS = [16, 17, 42, 7, 23]
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
CACHE_DIR = DATA_PROCESSED / "expcache_timesignal_blind"
B13_SEQS = {1, 2, 3, 4, 5, 6}
B45_SEQS = {13, 14, 15, 16}


def summarize(r: pd.DataFrame) -> dict:
    return {
        "n": len(r),
        "top1": round(float((r["rank"] < 1).mean()), 4),
        "top3": round(float((r["rank"] < 3).mean()), 4),
        "top5": round(float((r["rank"] < 5).mean()), 4),
    }


def main() -> None:
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate",
                 "label"],
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
    seq_test_gids = set(seq.loc[test_mask(seq), "gameid"])
    pre = games[~games.gameid.isin(seq_test_gids) & (games.date < cutoff)]
    g_val = pre[pre.date >= val_start].reset_index(drop=True)
    g_test = games[games.gameid.isin(seq_test_gids)].reset_index(drop=True)
    assert len(g_val) == 54
    n_eval = len(g_val) + len(g_test)
    val_pos = {g: i for i, g in enumerate(g_val.gameid)}

    val = cand[~test_mask(cand) & (cand.date >= val_start)
               & (cand.date < cutoff)].copy()
    assert val.groupby(["gameid", "seq"]).ngroups == 1080

    # Transformer side: blind-test ensemble caches, val slice (first 54 games
    # of the g_val-then-g_test eval ordering — val_pos never indexes past 53).
    probs = []
    for s in SEEDS:
        z = np.load(CACHE_DIR / f"seed{s}.npz")["probs"]
        assert z.shape == (n_eval, 20, vocab.size), (
            f"seed {s} cache shape {z.shape} != ({n_eval}, 20, {vocab.size}) "
            "— eval ordering drifted"
        )
        probs.append(z)
    val["tf_raw"] = attach_scores(val, np.mean(probs, axis=0), val_pos, vocab)

    # GBM side: cached codespace refit, original gameid keys.
    gbm = pd.read_parquet(DATA_PROCESSED / "gbm_val_scores.parquet")
    val = val.merge(
        gbm[["gameid", "seq", "candidate", "gbm_score"]],
        on=["gameid", "seq", "candidate"], how="left", validate="1:1",
    )
    assert val.gbm_score.notna().all(), "GBM score join incomplete"

    # Rank-average of pct-ranks over the full val table, as the template does.
    tr = val["tf_raw"].rank(pct=True).to_numpy()
    gb = val["gbm_score"].rank(pct=True).to_numpy()

    # One rank pass per weight; cells (all/picks/bans/bans_b13/bans_b45) are
    # read off the same per-decision hit ranks, so the per-type rows are
    # directly comparable against the stored re-sweep.
    dec_meta = (val.groupby(["gameid", "seq"], sort=False)["is_ban"]
                .first().reset_index())
    dec_meta["cell"] = np.where(
        dec_meta.is_ban == 0, "picks",
        np.where(dec_meta.seq.isin(list(B13_SEQS)), "bans_b13", "bans_b45"),
    )

    sweep: dict[float, dict] = {}
    for w in BLEND_WEIGHTS:
        val["s"] = w * tr + (1 - w) * gb

        def hit_rank(g: pd.DataFrame) -> int:
            order = np.argsort(-g["s"].to_numpy(), kind="stable")
            return int(np.argmax(g["label"].to_numpy()[order]))

        ranks = (val.groupby(["gameid", "seq"], sort=False)
                 .apply(hit_rank, include_groups=False)
                 .rename("rank").reset_index()
                 .merge(dec_meta, on=["gameid", "seq"]))
        sweep[w] = {
            "all": summarize(ranks),
            "picks": summarize(ranks[ranks.cell == "picks"]),
            "bans": summarize(ranks[ranks.is_ban == 1]),
            "bans_b13": summarize(ranks[ranks.cell == "bans_b13"]),
            "bans_b45": summarize(ranks[ranks.cell == "bans_b45"]),
        }
        print(f"w={w:<5} " + "  ".join(
            f"{c}:{sweep[w][c]['top1']*100:.1f}/{sweep[w][c]['top3']*100:.1f}"
            f"/{sweep[w][c]['top5']*100:.1f}"
            for c in ("all", "picks", "bans", "bans_b13", "bans_b45")
        ))

    # Canary: the all/picks/bans rows must reproduce the stored re-sweep
    # exactly — proves the local reconstruction matches the codespace run.
    stored = json.loads(
        (DATA_PROCESSED / "blend_resweep_timesignal.json").read_text()
    )
    for w in BLEND_WEIGHTS:
        for cell in ("all", "picks", "bans"):
            for k in ("n", "top1", "top3", "top5"):
                a, b = sweep[w][cell][k], stored["sweep"][str(w)][cell][k]
                assert a == b, (
                    f"canary miss: w={w} {cell}.{k} local {a} != stored {b}"
                )
    print("canary: all 15 stored re-sweep rows reproduced exactly")

    selected = {}
    for cell in ("picks", "bans_b13", "bans_b45"):
        w_best = max(BLEND_WEIGHTS, key=lambda w: (sweep[w][cell]["top1"], -w))
        selected[cell] = {"w_transformer": w_best, **sweep[w_best][cell]}
        print(f"{cell}: w_transformer={w_best} "
              f"top1={sweep[w_best][cell]['top1']:.4f}")

    # Composite bans metric under the selected per-phase weights vs the
    # single-weight w=0.0 production row.
    n13, n45 = selected["bans_b13"]["n"], selected["bans_b45"]["n"]
    composite = {
        k: round((selected["bans_b13"][k] * n13
                  + selected["bans_b45"][k] * n45) / (n13 + n45), 4)
        for k in ("top1", "top3", "top5")
    }
    composite["n"] = n13 + n45
    print(f"composite bans (phase-aware): top1={composite['top1']:.4f} "
          f"vs w=0.0 single-weight {sweep[0.0]['bans']['top1']:.4f}")

    out = {
        "note": ("Per-type-x-phase blend sweep on VAL ONLY, run locally from "
                 "cached GBM val scores (codespace refit) and blind-test "
                 "transformer ensemble caches. all/picks/bans rows asserted "
                 "identical to blend_resweep_timesignal.json. CAVEAT: the "
                 "B1-3/B4-5 split was chosen after seeing Stage B's cells on "
                 "these same 540 bans — a-priori motivated and only 1 extra "
                 "df, but weaker evidence than a pre-registered split. No "
                 "test look; promotion needs explicit approval."),
        "sweep": {str(w): sweep[w] for w in BLEND_WEIGHTS},
        "selected_per_cell": selected,
        "composite_bans_phase_aware": composite,
        "reference": {
            "production_weights": {"picks": 0.75, "bans": 0.0},
            "single_weight_bans_w0": sweep[0.0]["bans"],
            "naive_stageB_expectation_top1_gain": "~+2.4pts (post-hoc)",
        },
        "provenance": {
            "transformer": "meta-aware, expcache_timesignal_blind mean of "
                           f"seeds {SEEDS} (bit-identical to "
                           "timesignal_blindtest.json)",
            "gbm": "gbm_val_scores.parquet (codespace refit, cached; "
                   "cross-platform caveat applies vs stored local blocks)",
            "grid": BLEND_WEIGHTS,
            "phase_split": {"bans_b13_seqs": sorted(B13_SEQS),
                            "bans_b45_seqs": sorted(B45_SEQS)},
            "split_cutoff": str(cutoff.date()),
            "platform": platform.platform(),
        },
    }
    (DATA_PROCESSED / "blend_phase_sweep.json").write_text(
        json.dumps(out, indent=2)
    )
    print("wrote blend_phase_sweep.json")


if __name__ == "__main__":
    main()
