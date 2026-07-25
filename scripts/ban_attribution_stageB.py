"""Stage B of the denial-vector attribution study: condition model accuracy
on the ban's dominant denial vector.

Spec: docs/2026-07-25-ban-attribution-spec.md. Per-ban top-1/top-5 hits on
the 540 val bans for three models, each conditioned on the Stage A dominant
vector:

  baseline  v0.8-lineage transformer 5-seed ensemble (expcache_attr)
  meta      ban time-signal transformer 5-seed ensemble (expcache_attr;
            bit-identical to the blind-test cache, asserted at train time)
  gbm       production 10-model GBM ensemble, codespace refit
            (gbm_val_scores.parquet; cross-platform caveat — GBM numbers
            compare only within this table, not to stored local blocks)

Hypothesis under test: the transformer's ban gap vs the GBM concentrates in
POCKET-dominant bans (it has no team/player inputs), is partially closed in
META by the time-signal rung, and is smallest in COMP.

VAL ONLY, descriptive: n is small (540 bans; smaller per cell) — directional
evidence for choosing the next accuracy rung, not promotion evidence.

Inputs: ban_attribution_perban.parquet (Stage A), expcache_attr/*.npz,
        gbm_val_scores.parquet
Output: data/processed/ban_attribution.json (stage_B section)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from draft_transformer import Vocab, attach_scores, build_games

VAL_DAYS = 14
SEEDS = [16, 17, 42, 7, 23]
CACHE_DIR = DATA_PROCESSED / "expcache_attr"


def hit_ranks(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Rank of the true target under each score column, per decision."""
    recs = []
    for (gid, s), g in df.groupby(["gameid", "seq"], sort=False):
        lab = g["label"].to_numpy()
        rec = {"gameid": gid, "seq": s}
        for c in score_cols:
            order = np.argsort(-g[c].to_numpy(), kind="stable")
            rec[c] = int(np.argmax(lab[order]))
        recs.append(rec)
    return pd.DataFrame(recs)


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
    assert len(g_val) == 54
    val_pos = {g: i for i, g in enumerate(g_val.gameid)}

    val_rows = cand[~test_mask(cand) & (cand.date >= val_start)
                    & (cand.date < cutoff)].copy()
    assert val_rows.groupby(["gameid", "seq"]).ngroups == 1080

    for cond in ("baseline", "meta"):
        probs = np.mean(
            [np.load(CACHE_DIR / f"{cond}_seed{s}.npz")["probs"] for s in SEEDS],
            axis=0,
        )
        assert probs.shape == (54, 20, vocab.size)
        val_rows[cond] = attach_scores(val_rows, probs, val_pos, vocab)

    gbm = pd.read_parquet(DATA_PROCESSED / "gbm_val_scores.parquet")
    val_rows = val_rows.merge(
        gbm[["gameid", "seq", "candidate", "gbm_score"]],
        on=["gameid", "seq", "candidate"], how="left", validate="1:1",
    )
    assert val_rows.gbm_score.notna().all(), "GBM score join incomplete"

    models = ["baseline", "meta", "gbm_score"]
    ranks = hit_ranks(val_rows, models)
    ranks = ranks.merge(
        val_rows.groupby(["gameid", "seq"], sort=False)["is_ban"]
        .first().reset_index(),
        on=["gameid", "seq"],
    )

    perban = pd.read_parquet(DATA_PROCESSED / "ban_attribution_perban.parquet")
    bans = ranks[ranks.is_ban == 1].merge(
        perban[["gameid", "seq", "ordinal", "dominant", "dominant4", "multi"]],
        on=["gameid", "seq"], how="left", validate="1:1",
    )
    n_unmatched = int(bans.dominant.isna().sum())
    bans = bans.dropna(subset=["dominant"])

    def table(df: pd.DataFrame, by: str) -> dict:
        out = {}
        for k, g in df.groupby(by, observed=True):
            out[str(k)] = {"n": len(g)}
            for m in models:
                out[str(k)][m] = {
                    "top1": round(float((g[m] < 1).mean()), 4),
                    "top5": round(float((g[m] < 5).mean()), 4),
                }
        return out

    overall = table(bans.assign(all="all_bans"), "all")["all_bans"]
    by_vec = table(bans, "dominant")
    by_vec4 = table(bans, "dominant4")
    by_phase_vec = {
        ph: table(g, "dominant")
        for ph, g in bans.groupby(bans.ordinal.le(3).map(
            {True: "B1-3", False: "B4-5"}))
    }
    by_multi = table(bans, "multi")

    out = {
        "note": ("Stage B: per-ban top-1/top-5 on the 540 val bans for "
                 "baseline transformer / meta transformer / codespace-refit "
                 "GBM, conditioned on Stage A dominant vector. Directional "
                 "(small n); GBM comparable only within this table."),
        "n_val_bans": int(len(bans)),
        "n_unmatched_dropped": n_unmatched,
        "overall": overall,
        "by_dominant_vector": by_vec,
        "by_dominant_vector_pocket_split": by_vec4,
        "by_phase_and_vector": by_phase_vec,
        "by_multi_motive": by_multi,
        "val_picks_reference": table(
            ranks[ranks.is_ban == 0].assign(all="picks"), "all"),
    }
    path = DATA_PROCESSED / "ban_attribution.json"
    existing = json.loads(path.read_text())
    existing["stage_B"] = out
    path.write_text(json.dumps(existing, indent=2))

    print(f"val bans: {len(bans)} (dropped {n_unmatched} unmatched)")
    for name, tab in [("overall", {"all": overall}),
                      ("by vector", by_vec), ("by multi", by_multi)]:
        print(f"\n{name}:")
        for k, v in tab.items():
            row = "  ".join(
                f"{m}:{v[m]['top1']*100:.1f}/{v[m]['top5']*100:.1f}"
                for m in models
            )
            print(f"  {k:>8} n={v['n']:<4} {row}")
    print("\nby phase x vector:")
    for ph, tab in by_phase_vec.items():
        for k, v in tab.items():
            row = "  ".join(
                f"{m}:{v[m]['top1']*100:.1f}/{v[m]['top5']*100:.1f}"
                for m in models
            )
            print(f"  {ph} {k:>8} n={v['n']:<4} {row}")


if __name__ == "__main__":
    main()
