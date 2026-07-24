"""Re-sweep the v0.8.1 per-type blend weights with the meta-aware transformer.

VAL ONLY — no test look, ever, from this script. The blind test
(timesignal_blindtest.json, third and final look) showed the meta-aware
transformer's picks now beat the GBM's (15.4 vs 14.8 test top-1); the
production pick-blend weight 0.75 was swept for the old transformer
(12.8-grade picks). This re-runs the experiment_v08.py §3-4 sweep on val
with the meta transformer to see if the optimal weights move. Any promotion
into train_draft_model_v08.py ships on this val evidence alone.

Designed to run on a Codespace (Linux), where lightgbm imports cleanly —
locally libomp is missing, which is why experiment_v08 helpers can be
imported here but not in the experiment_v11 scripts.

Transformer scores are NOT recomputed: the 5-seed eval probability caches
from the blind-test run (expcache_timesignal_blind/seed<S>.npz, val+test
games in g_val-then-g_test order) are loaded and averaged, keeping the
transformer side bit-identical to the tested lineage across machines. Only
the val slice of those probs is ever read. The GBM side is the full
production 10-model ensemble (5 seeds x clf/ranker) refit on the train
split, as train_draft_model_v08.py does.

Output: data/processed/blend_resweep_timesignal.json
"""

from __future__ import annotations

import json
import platform

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from draft_transformer import Vocab, attach_scores, build_games
from experiment_v08 import load_multi, split_dates
from train_draft_model import (
    FEATURES, SEEDS, ensemble_score, fit_clf, fit_ranker, topk_accuracy,
)

BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # same grid as experiment_v08
CACHE_DIR = DATA_PROCESSED / "expcache_timesignal_blind"


def main() -> None:
    ds, seq = load_multi()
    is_test, cutoff, val_start = split_dates(ds)
    pre = ds[~is_test & (ds.date < cutoff)]
    train = pre[pre.date < val_start]
    val = pre[pre.date >= val_start].copy()

    vocab = Vocab(list(ds.candidate.unique()), list(seq.champion.unique()))
    assert vocab.size == 171
    games = build_games(seq, vocab)

    # Reconstruct the blind-test run's eval-game ordering (g_val then g_test,
    # appearance order within each) so the cached probs line up. build_games
    # order depends only on sequence-table row order, so the int32 gameid
    # codes from load_multi don't disturb it.
    test_gids = set(seq.loc[(seq.league == "EWC") & (seq.date.dt.month == 7)
                            & (seq.date.dt.year == 2026), "gameid"])
    g_pre = games[~games.gameid.isin(test_gids) & (games.date < cutoff)]
    g_val = g_pre[g_pre.date >= val_start].reset_index(drop=True)
    g_test = games[games.gameid.isin(test_gids)].reset_index(drop=True)
    g_eval = pd.concat([g_val, g_test], ignore_index=True)
    eval_pos = {g: i for i, g in enumerate(g_eval.gameid)}
    n_val_dec = val.groupby(["gameid", "seq"]).ngroups
    print(f"cutoff {cutoff.date()}: val {len(g_val)}g/{n_val_dec}d, "
          f"eval games {len(g_eval)}")
    assert (len(g_val), n_val_dec) == (54, 1080)

    seed_probs = []
    for s in SEEDS:
        z = np.load(CACHE_DIR / f"seed{s}.npz")
        seed_probs.append(z["probs"])
        assert z["probs"].shape == (len(g_eval), 20, vocab.size), (
            f"seed {s} prob cache shape {z['probs'].shape} does not match "
            f"({len(g_eval)}, 20, {vocab.size}) — eval ordering drifted"
        )
    mean_probs = np.mean(seed_probs, axis=0)
    tf_raw = attach_scores(val, mean_probs, eval_pos, vocab)

    print(f"fitting production GBM ensemble ({len(SEEDS)} seeds x clf/ranker) "
          f"on {train.gameid.nunique()} train games...")
    gbm = {"clf": [fit_clf(FEATURES, train, s) for s in SEEDS],
           "rank": [fit_ranker(FEATURES, train, s) for s in SEEDS]}
    gbm_raw = ensemble_score(gbm, FEATURES, val)

    tr = pd.Series(tf_raw).rank(pct=True).to_numpy()
    gb = pd.Series(gbm_raw).rank(pct=True).to_numpy()

    def fmt(r: dict) -> str:
        return "  ".join(
            f"{s}:{r[s]['top1'] * 100:.1f}/{r[s]['top3'] * 100:.1f}/"
            f"{r[s]['top5'] * 100:.1f}"
            for s in ("all", "picks", "bans")
        )

    sweep: dict[float, dict] = {}
    for w in BLEND_WEIGHTS:
        val["s"] = w * tr + (1 - w) * gb
        sweep[w] = topk_accuracy(val, "s")
        print(f"w_transformer={w:<5} {fmt(sweep[w])}")

    selected = {}
    for kind in ("picks", "bans"):
        w_best = max(BLEND_WEIGHTS, key=lambda w: (sweep[w][kind]["top1"], -w))
        selected[kind] = {"w_transformer": w_best, **sweep[w_best][kind]}
        print(f"{kind}: w_transformer={w_best} "
              f"top1={sweep[w_best][kind]['top1']:.4f}")

    stored = json.loads(
        (DATA_PROCESSED / "draft_model_metrics_v08.json").read_text()
    )
    out = {
        "note": ("v0.8.1 blend re-sweep on VAL ONLY with the meta-aware "
                 "transformer (cached blind-test ensemble probs, val slice) "
                 "vs freshly refit production GBM ensemble. No test look; "
                 "any weight promotion ships on this evidence alone."),
        "sweep": {str(w): sweep[w] for w in BLEND_WEIGHTS},
        "selected_per_type": selected,
        "production_reference": {
            "current_weights": {"picks": 0.75, "bans": 0.0},
            "v0.8.1_val": stored["v0.8.1_blend_pertype"]["val"]["model"],
        },
        "provenance": {
            "transformer": "meta-aware, expcache_timesignal_blind mean of "
                           "seeds " + str(SEEDS) + " (bit-identical to "
                           "timesignal_blindtest.json)",
            "gbm": "10-model production ensemble refit on train, this run",
            "grid": BLEND_WEIGHTS, "split_cutoff": str(cutoff.date()),
            "platform": platform.platform(),
        },
    }
    (DATA_PROCESSED / "blend_resweep_timesignal.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"wrote blend_resweep_timesignal.json")


if __name__ == "__main__":
    main()
