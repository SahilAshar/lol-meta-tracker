"""Cache the production GBM ensemble's VAL scores for the attribution study.

Spec: docs/2026-07-25-ban-attribution-spec.md (Stage B infra note: "cache the
GBM val scores this time"). Fits the full production 10-model ensemble
(5 seeds x clf/ranker, train split) exactly as blend_resweep_timesignal.py
did, scores the val candidate rows once, and writes them keyed by ORIGINAL
gameid strings so any machine can join them without replaying load_multi's
int32 code mapping.

Codespace-only (lightgbm; libomp missing locally). Cross-platform caveat
carries over from the re-sweep: these Linux-refit scores are not comparable
to stored local metric blocks — Stage B only compares models within this run.

VAL ONLY. No test rows are scored.

Output: data/processed/gbm_val_scores.parquet (gitignored; tar back to local)
"""

from __future__ import annotations

import platform

import pandas as pd

from common import DATA_PROCESSED
from experiment_v08 import load_multi, split_dates
from train_draft_model import FEATURES, SEEDS, ensemble_score, fit_clf, fit_ranker


def main() -> None:
    ds, seq = load_multi()
    is_test, cutoff, val_start = split_dates(ds)
    pre = ds[~is_test & (ds.date < cutoff)]
    train = pre[pre.date < val_start]
    val = pre[pre.date >= val_start].copy()
    n_val_dec = val.groupby(["gameid", "seq"]).ngroups
    assert n_val_dec == 1080, n_val_dec

    # load_multi replaced gameids with int32 codes in seq-table order; invert.
    orig = pd.read_parquet(
        DATA_PROCESSED / "draft_sequences_multi.parquet", columns=["gameid"]
    ).gameid.unique()

    print(f"fitting production GBM ensemble ({len(SEEDS)} seeds x clf/ranker) "
          f"on {train.gameid.nunique()} train games ({platform.platform()})")
    gbm = {"clf": [fit_clf(FEATURES, train, s) for s in SEEDS],
           "rank": [fit_ranker(FEATURES, train, s) for s in SEEDS]}
    val["gbm_score"] = ensemble_score(gbm, FEATURES, val)

    out = val[["gameid", "seq", "is_ban", "candidate", "label", "gbm_score"]].copy()
    out["gameid"] = out.gameid.map(lambda c: orig[c])
    out["candidate"] = out.candidate.astype(str)
    out.to_parquet(DATA_PROCESSED / "gbm_val_scores.parquet")
    print(f"wrote gbm_val_scores.parquet: {len(out)} rows, "
          f"{out.groupby(['gameid', 'seq']).ngroups} decisions")


if __name__ == "__main__":
    main()
