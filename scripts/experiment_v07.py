"""v0.7 val-only experiments: seed ensembling, pair features, ranking objective.

Compares, on the validation split ONLY (the EWC test set stays untouched):
  A. v0.6 features, HistGBM classifier, 5-seed ensemble
  B. v0.6 + pair features, classifier, 5-seed ensemble
  C. v0.6 features, LGBMRanker (lambdarank), 5-seed ensemble
  D. v0.6 + pair features, ranker, 5-seed ensemble
Single-seed (16) numbers are printed alongside to show what ensembling buys.

The winner's config gets promoted into train_draft_model.py as v0.7 and only
then blind-tested once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping
from sklearn.ensemble import HistGradientBoostingClassifier

from common import DATA_PROCESSED
from train_draft_model import FEATURES as FEATURES_V06
from train_draft_model import VAL_DAYS, topk_accuracy

FEATURES_V07 = FEATURES_V06 + ["pair_syn", "pair_ctr"]
SEEDS = [16, 17, 42, 7, 23]


def fit_clf(features: list[str], train: pd.DataFrame, seed: int):
    model = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.08, max_depth=6, min_samples_leaf=50,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=seed,
    )
    model.fit(train[features], train["label"])
    return lambda part: model.predict_proba(part[features])[:, 1]


def fit_ranker(features: list[str], train: pd.DataFrame, seed: int):
    """LambdaRank over decision groups; 10% of games held out for early stop."""
    games = train.gameid.unique()
    rng = np.random.RandomState(seed)
    holdout = set(rng.choice(games, size=len(games) // 10, replace=False))
    fit_df = train[~train.gameid.isin(holdout)].sort_values(["gameid", "seq"])
    es_df = train[train.gameid.isin(holdout)].sort_values(["gameid", "seq"])
    groups = fit_df.groupby(["gameid", "seq"], sort=False).size().to_numpy()
    es_groups = es_df.groupby(["gameid", "seq"], sort=False).size().to_numpy()
    model = LGBMRanker(
        objective="lambdarank", n_estimators=600, learning_rate=0.08,
        num_leaves=63, min_child_samples=50, random_state=seed, verbose=-1,
    )
    model.fit(
        fit_df[features], fit_df["label"], group=groups,
        eval_set=[(es_df[features], es_df["label"])], eval_group=[es_groups],
        eval_at=[5], callbacks=[early_stopping(25, verbose=False)],
    )
    return lambda part: model.predict(part[features])


def main() -> None:
    ds = pd.read_parquet(DATA_PROCESSED / "draft_decisions.parquet")
    ds["date"] = pd.to_datetime(ds["date"])
    is_test = (ds.league == "EWC") & (ds.date.dt.month == 7)
    cutoff = ds.loc[is_test, "date"].min()
    pre = ds[~is_test & (ds.date < cutoff)]
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)
    train, val = pre[pre.date < val_start], pre[pre.date >= val_start].copy()
    print(f"train: {train.gameid.nunique()} games, val: {val.gameid.nunique()} games")

    configs = [
        ("A clf v0.6-feats", fit_clf, FEATURES_V06),
        ("B clf +pair", fit_clf, FEATURES_V07),
        ("C rank v0.6-feats", fit_ranker, FEATURES_V06),
        ("D rank +pair", fit_ranker, FEATURES_V07),
    ]
    for tag, fit, feats in configs:
        scores = []
        for seed in SEEDS:
            scores.append(fit(feats, train, seed)(val))
            if seed == SEEDS[0]:
                val["s"] = scores[0]
                r = topk_accuracy(val, "s")
                line = "  ".join(
                    f"{s}:{r[s]['top1']*100:.1f}/{r[s]['top3']*100:.1f}/{r[s]['top5']*100:.1f}"
                    for s in ("all", "picks", "bans"))
                print(f"{tag:18s} seed16   {line}")
        # Rank-average across seeds: scales differ between seeds/models.
        val["s"] = np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in scores], axis=0)
        r = topk_accuracy(val, "s")
        line = "  ".join(
            f"{s}:{r[s]['top1']*100:.1f}/{r[s]['top3']*100:.1f}/{r[s]['top5']*100:.1f}"
            for s in ("all", "picks", "bans"))
        print(f"{tag:18s} ens5     {line}")


if __name__ == "__main__":
    main()
