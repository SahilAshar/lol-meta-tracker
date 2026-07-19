"""Train the v0.7 next-pick model and blind-score it on the EWC main event.

At eval time, candidates for each decision are ranked by score; we report
top-1/3/5 accuracy against the champion actually picked/banned.

v0.7 = v0.6 + champion-pair features (pair_syn, pair_ctr) + a 10-model
ensemble: 5 seeds x {HistGradientBoosting classifier, LGBMRanker lambdarank},
scores rank-averaged with equal weight between the two families. Chosen on
val only (experiment_v07*.py): the ranker family wins top-1, the classifier
family wins top-5, seeds and even platforms shift any single fit's top-1 by
+/-1.5pts, and the equal blend dominates both families. Stored v0.6/v0.5/v0
blocks are copied into the output; if the test set has grown since they were
scored, those feature sets are refit on the current data (original
single-model config) so the comparison stays apples-to-apples.

Split design (temporal, no leakage):
  - test  = EWC July main event (patch 16.13)
  - train = every game strictly before the first test game
  - val   = last 14 days of the train era (reported separately; the estimator's
    internal early stopping uses a random holdout within train)

Baselines: rank by trailing-28d meta rates (pick_rate for picks, presence for
bans) and by trailing-56d team habit (team_usage for picks, opp_usage for bans).

Output: data/processed/draft_model_metrics_v07.json,
        data/processed/draft_model_v07.joblib
"""

from __future__ import annotations

import json
import platform

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping
from sklearn.ensemble import HistGradientBoostingClassifier

from common import DATA_PROCESSED

FEATURES_V05 = [
    "pick_rate", "ban_rate", "presence", "team_usage", "opp_usage",
    "is_ban", "phase2", "is_blue", "ordinal", "fearless", "game_in_series",
    "role_need", "role_overlap_max",
]
FEATURES_V06 = FEATURES_V05 + ["player_pool", "player_wr"]
FEATURES = FEATURES_V06 + ["pair_syn", "pair_ctr"]
SEEDS = [16, 17, 42, 7, 23]
VAL_DAYS = 14
VERSION = "v0.7"


def topk_accuracy(df: pd.DataFrame, score_col: str) -> dict:
    """Mean top-k hit rate over decisions, overall and split by picks/bans."""
    def hit_rank(g: pd.DataFrame) -> int:
        order = np.argsort(-g[score_col].to_numpy(), kind="stable")
        return int(np.argmax(g["label"].to_numpy()[order]))

    grouped = df.groupby(["gameid", "seq"], sort=False)
    ranks = grouped.apply(hit_rank, include_groups=False).rename("rank").reset_index()
    ranks = ranks.merge(
        grouped["is_ban"].first().reset_index(), on=["gameid", "seq"]
    )

    def summarize(r: pd.DataFrame) -> dict:
        return {
            "n": len(r),
            "top1": round(float((r["rank"] < 1).mean()), 4),
            "top3": round(float((r["rank"] < 3).mean()), 4),
            "top5": round(float((r["rank"] < 5).mean()), 4),
        }

    return {
        "all": summarize(ranks),
        "picks": summarize(ranks[ranks.is_ban == 0]),
        "bans": summarize(ranks[ranks.is_ban == 1]),
    }


def fit_clf(features: list[str], train: pd.DataFrame, seed: int = 16):
    model = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
    )
    model.fit(train[features], train["label"])
    print(f"clf seed={seed} ({len(features)} features): {model.n_iter_} iterations")
    return model


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
    print(f"ranker seed={seed} ({len(features)} features): {model.best_iteration_} iterations")
    return model


def ensemble_score(models: dict, features: list[str], part: pd.DataFrame) -> np.ndarray:
    """Equal-weight rank-average: mean pct-rank within family, then mean of
    the two families (pct-ranks are monotone per model, so within-decision
    ordering is preserved)."""
    def family(ms, predict):
        return np.mean([
            pd.Series(predict(m)).rank(pct=True).to_numpy() for m in ms
        ], axis=0)

    clf = family(models["clf"], lambda m: m.predict_proba(part[features])[:, 1])
    rank = family(models["rank"], lambda m: m.predict(part[features]))
    return (clf + rank) / 2.0


def score_splits(score_fn, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    out = {}
    for name, part in [("val", val.copy()), ("test_ewc_main", test.copy())]:
        part["model_score"] = score_fn(part)
        part["baseline_meta"] = np.where(part.is_ban == 1, part.presence, part.pick_rate)
        part["baseline_team"] = np.where(part.is_ban == 1, part.opp_usage, part.team_usage)
        out[name] = {
            "model": topk_accuracy(part, "model_score"),
            "baseline_meta": topk_accuracy(part, "baseline_meta"),
            "baseline_team": topk_accuracy(part, "baseline_team"),
        }
    return out


def main() -> None:
    ds = pd.read_parquet(DATA_PROCESSED / "draft_decisions.parquet")
    ds["date"] = pd.to_datetime(ds["date"])

    is_test = (ds.league == "EWC") & (ds.date.dt.month == 7)
    cutoff = ds.loc[is_test, "date"].min()
    pre = ds[~is_test & (ds.date < cutoff)]
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)
    train, val, test = pre[pre.date < val_start], pre[pre.date >= val_start], ds[is_test]
    print(f"cutoff (first EWC main-event game): {cutoff}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}: {part.gameid.nunique()} games, "
              f"{part.groupby(['gameid', 'seq']).ngroups} decisions")

    models = {
        "clf": [fit_clf(FEATURES, train, s) for s in SEEDS],
        "rank": [fit_ranker(FEATURES, train, s) for s in SEEDS],
    }
    results = {
        "version": VERSION, "cutoff": str(cutoff), "features": FEATURES,
        "ensemble": {"seeds": SEEDS, "families": ["hist_gbm_clf", "lgbm_lambdarank"],
                     "blend": "equal-weight rank average"},
        "platform": platform.platform(),
    }
    results.update(score_splits(
        lambda part: ensemble_score(models, FEATURES, part), val, test))

    # Carry the v0.6/v0.5/v0 comparison blocks forward. If the test set has
    # changed since they were scored, refit those feature sets (original
    # single-model config) on the current data instead.
    v06_path = DATA_PROCESSED / "draft_model_metrics_v06.json"
    if v06_path.exists():
        stored = json.loads(v06_path.read_text())
        results["v0"] = stored.pop("v0", None)
        results["v0.5"] = stored.pop("v0.5", None)
        n_test = test.groupby(["gameid", "seq"]).ngroups
        stored_n = stored["test_ewc_main"]["model"]["all"]["n"]
        if stored_n == n_test:
            results["v0.6"] = stored
        else:
            print(f"test set changed ({stored_n} -> {n_test} decisions); "
                  "refitting v0.5/v0.6 feature sets for comparability")
            for tag, feats in [("v0.5", FEATURES_V05), ("v0.6", FEATURES_V06)]:
                m = fit_clf(feats, train, 16)
                blk = {"version": f"{tag} (refit)", "features": feats,
                       "stored_test_n": stored_n}
                blk.update(score_splits(
                    lambda part, m=m, f=feats: m.predict_proba(part[f])[:, 1],
                    val, test))
                results[tag] = blk

    out = DATA_PROCESSED / "draft_model_metrics_v07.json"
    out.write_text(json.dumps(results, indent=2))
    joblib.dump(models, DATA_PROCESSED / "draft_model_v07.joblib")
    print(json.dumps(
        {k: v for k, v in results.items() if k not in ("v0", "v0.5", "v0.6")},
        indent=2))


if __name__ == "__main__":
    main()
