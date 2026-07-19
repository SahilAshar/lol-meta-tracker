"""v0.7 follow-up (val-only): blend the classifier and ranker ensembles.

experiment_v07.py showed the LambdaRank ensemble wins top-1 while the
classifier ensemble wins top-5 — complementary errors. This evaluates
rank-averaged blends of the two 5-seed ensembles (both with pair features)
at a few mixing weights, on the validation split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from experiment_v07 import FEATURES_V07, SEEDS, fit_clf, fit_ranker
from train_draft_model import VAL_DAYS, topk_accuracy


def main() -> None:
    ds = pd.read_parquet(DATA_PROCESSED / "draft_decisions.parquet")
    ds["date"] = pd.to_datetime(ds["date"])
    is_test = (ds.league == "EWC") & (ds.date.dt.month == 7)
    cutoff = ds.loc[is_test, "date"].min()
    pre = ds[~is_test & (ds.date < cutoff)]
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)
    train, val = pre[pre.date < val_start], pre[pre.date >= val_start].copy()

    def ens(fit):
        return np.mean([
            pd.Series(fit(FEATURES_V07, train, seed)(val)).rank(pct=True).to_numpy()
            for seed in SEEDS
        ], axis=0)

    clf, rank = ens(fit_clf), ens(fit_ranker)
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        val["s"] = w * clf + (1 - w) * rank
        r = topk_accuracy(val, "s")
        line = "  ".join(
            f"{s}:{r[s]['top1']*100:.1f}/{r[s]['top3']*100:.1f}/{r[s]['top5']*100:.1f}"
            for s in ("all", "picks", "bans"))
        print(f"blend clf_w={w:.1f}   {line}")


if __name__ == "__main__":
    main()
