"""Train the v0 next-pick model and blind-score it on the EWC main event.

Pointwise ranking: a gradient-boosted classifier over (decision, candidate) rows
from draft_dataset.py. At eval time, candidates for each decision are ranked by
score; we report top-1/3/5 accuracy against the champion actually picked/banned.

Split design (temporal, no leakage):
  - test  = EWC July main event (patch 16.13)
  - train = every game strictly before the first test game
  - val   = last 14 days of the train era (reported separately; the estimator's
    internal early stopping uses a random holdout within train)

Baselines: rank by trailing-28d meta rates (pick_rate for picks, presence for
bans) and by trailing-56d team habit (team_usage for picks, opp_usage for bans).

Output: data/processed/draft_model_metrics.json, data/processed/draft_model.joblib
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import DATA_PROCESSED

FEATURES = [
    "pick_rate", "ban_rate", "presence", "team_usage", "opp_usage",
    "is_ban", "phase2", "is_blue", "ordinal", "fearless", "game_in_series",
]
VAL_DAYS = 14


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

    model = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=16,
    )
    model.fit(train[FEATURES], train["label"])
    print(f"iterations used: {model.n_iter_}")

    results = {"cutoff": str(cutoff), "features": FEATURES}
    for name, part in [("val", val.copy()), ("test_ewc_main", test.copy())]:
        part["model_score"] = model.predict_proba(part[FEATURES])[:, 1]
        part["baseline_meta"] = np.where(part.is_ban == 1, part.presence, part.pick_rate)
        part["baseline_team"] = np.where(part.is_ban == 1, part.opp_usage, part.team_usage)
        results[name] = {
            "model": topk_accuracy(part, "model_score"),
            "baseline_meta": topk_accuracy(part, "baseline_meta"),
            "baseline_team": topk_accuracy(part, "baseline_team"),
        }

    out = DATA_PROCESSED / "draft_model_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    joblib.dump(model, DATA_PROCESSED / "draft_model.joblib")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
