"""Stage A of the denial-vector attribution study: classify historical bans.

Spec: docs/2026-07-25-ban-attribution-spec.md. Every ban is scored on three
denial vectors as within-decision percentiles across that slot's available
candidates (so vectors share one scale):

    META   = pct(presence)                     28-day global pick+ban rate
    POCKET = max(pct(opp_usage),               opponent team habit
                 pct(player_pool * max(player_wr - 0.5, 0)))
                                               opponent player pool x skill
    COMP   = pct(pair_ctr)                     counters our locked picks

Dominant vector = argmax over defined vectors; a vector is undefined (NaN)
when its raw feature is constant across the availability pool — pair features
are neutral before any picks lock, so B1-B3 can only read as META/POCKET,
which matches the game reality. Multi-motive = top two defined vectors within
eps percentile points (eps = 10, sensitivity at 5/15).

Sensitivity variants for the spec's open questions: (1) POCKET split into
TEAM (opp_usage) and PLAYER (pool x skill) sub-vectors; (3) COMP fused with
pair_syn ("slots into their comp") via max.

Population: all train+val bans (date < EWC July-2026 cutoff). Descriptive
only — no model, no test look. Ban rows are opponent-referenced by
construction (draft_dataset.py): player_pool/player_wr score the opponent's
roster, pair_ctr scores the candidate against the banning team's own locked
picks.

Output: data/processed/ban_attribution.json          (stage_A section)
        data/processed/ban_attribution_perban.parquet (per-ban assignments,
        consumed by Stage B; gitignored like all parquets)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_PROCESSED

EPS_MAIN = 10.0
EPS_SENS = [5.0, 15.0]
VECTORS = ["META", "POCKET", "COMP"]


def load_bans() -> tuple[pd.DataFrame, pd.Timestamp]:
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "ordinal",
                 "candidate", "presence", "opp_usage", "player_pool",
                 "player_wr", "pair_syn", "pair_ctr", "label"],
    )
    cand["date"] = pd.to_datetime(cand["date"])
    is_test = ((cand.league == "EWC") & (cand.date.dt.month == 7)
               & (cand.date.dt.year == 2026))
    cutoff = cand.loc[is_test, "date"].min()
    bans = cand[(cand.date < cutoff) & (cand.is_ban == 1)].copy()
    return bans, cutoff


def within_decision_pct(bans: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Percentile (0-100] of each column within its (gameid, seq) pool;
    NaN where the column is constant across the pool (vector undefined)."""
    grp = bans.groupby(["gameid", "seq"], sort=False)
    out = {}
    for c in cols:
        r = grp[c].rank(pct=True) * 100.0
        spread = grp[c].transform("max") - grp[c].transform("min")
        out[c] = r.where(spread > 0)
    return pd.DataFrame(out, index=bans.index)


def dominant(scores: pd.DataFrame, eps: float) -> tuple[pd.Series, pd.Series]:
    """Argmax vector name per row (NaN-safe) and multi-motive flag: top two
    *defined* vectors within eps points."""
    arr = scores.to_numpy(dtype=float)
    order = np.argsort(np.nan_to_num(arr, nan=-1.0), axis=1)
    top, second = order[:, -1], order[:, -2]
    rows = np.arange(len(arr))
    dom = pd.Series(
        np.array(scores.columns)[top], index=scores.index, name="dominant"
    )
    second_val = arr[rows, second]
    multi = (~np.isnan(second_val)) & (
        arr[rows, top] - second_val <= eps
    )
    return dom, pd.Series(multi, index=scores.index, name="multi")


def mixture_table(df: pd.DataFrame, by: str, col: str = "dominant") -> dict:
    tab = (df.groupby(by, observed=True)[col]
             .value_counts(normalize=True).unstack(fill_value=0.0))
    return {
        str(k): {v: round(float(tab.loc[k, v]), 4) for v in tab.columns}
        for k in tab.index
    }


def main() -> None:
    bans, cutoff = load_bans()
    n_dec = bans.groupby(["gameid", "seq"]).ngroups
    print(f"cutoff {cutoff.date()}: {n_dec} ban decisions, {len(bans)} "
          f"candidate rows, {bans.gameid.nunique()} games")

    bans["player_signal"] = (
        bans.player_pool * (bans.player_wr - 0.5).clip(lower=0)
    )
    pct = within_decision_pct(
        bans, ["presence", "opp_usage", "player_signal", "pair_ctr", "pair_syn"]
    )
    scores = pd.DataFrame({
        "META": pct.presence,
        "POCKET": pct[["opp_usage", "player_signal"]].max(axis=1),
        "COMP": pct.pair_ctr,
    })
    # Sensitivity variants (spec open questions 1 and 3)
    split4 = pd.DataFrame({
        "META": pct.presence, "TEAM": pct.opp_usage,
        "PLAYER": pct.player_signal, "COMP": pct.pair_ctr,
    })
    fused = scores.assign(COMP=pct[["pair_ctr", "pair_syn"]].max(axis=1))

    # Decisions with no label==1 row banned a champion outside the game's
    # availability pool (e.g. a fearless series-prior champ) — unattributable
    # by construction; dropped and counted.
    banned = bans[bans.label == 1].copy()
    assert banned.groupby(["gameid", "seq"]).size().eq(1).all()
    n_unattr = n_dec - len(banned)
    print(f"unattributable ban decisions (target outside pool): {n_unattr}")
    sb = scores.loc[banned.index]
    banned["dominant"], banned["multi"] = dominant(sb, EPS_MAIN)
    for eps in EPS_SENS:
        banned[f"multi_eps{int(eps)}"] = dominant(sb, eps)[1]
    banned["dominant4"] = dominant(split4.loc[banned.index], EPS_MAIN)[0]
    banned["dominant_fused"] = dominant(fused.loc[banned.index], EPS_MAIN)[0]
    for v in VECTORS:
        banned[f"pct_{v}"] = sb[v]
    banned["pct_SYN"] = pct.loc[banned.index, "pair_syn"]
    banned["month"] = banned.date.dt.strftime("%Y-%m")

    # Per-vector lift: mean percentile of the actually-banned champ on each
    # vector, vs the 50 base rate of a uniform draw from the pool.
    def lifts(g: pd.DataFrame) -> dict:
        out = {}
        for v in VECTORS + ["SYN"]:
            s = g[f"pct_{v}"].dropna()
            out[v] = {"n_defined": int(len(s)),
                      "mean_pct": round(float(s.mean()), 2) if len(s) else None,
                      "lift_vs_50": round(float(s.mean() - 50), 2) if len(s) else None}
        return out

    top_leagues = banned.league.value_counts().head(10).index
    banned["league_grp"] = np.where(
        banned.league.isin(top_leagues), banned.league, "other"
    )

    out = {
        "note": ("Stage A of the denial-vector attribution study "
                 "(docs/2026-07-25-ban-attribution-spec.md). Descriptive "
                 "classification of all train+val bans by dominant denial "
                 "vector; within-decision percentile proxies; no model, no "
                 "test look."),
        "population": {
            "cutoff": str(cutoff.date()), "n_ban_decisions": n_dec,
            "n_attributed": int(len(banned)),
            "n_unattributable_target_outside_pool": int(n_unattr),
            "n_games": int(bans.gameid.nunique()),
            "undefined_rates": {
                v: round(float(sb[v].isna().mean()), 4) for v in VECTORS
            },
        },
        "mixture_by_ban_ordinal": mixture_table(banned, "ordinal"),
        "multi_motive_rate_by_ordinal": {
            f"eps={int(e)}": {
                str(k): round(float(v), 4)
                for k, v in banned.groupby("ordinal")[c].mean().items()
            }
            for e, c in [(EPS_MAIN, "multi")] + [
                (e, f"multi_eps{int(e)}") for e in EPS_SENS
            ]
        },
        "lift_by_ordinal": {
            str(k): lifts(g) for k, g in banned.groupby("ordinal")
        },
        "lift_overall": lifts(banned),
        "mixture_by_month": mixture_table(banned, "month"),
        "mixture_by_league": mixture_table(banned, "league_grp"),
        "sensitivity": {
            "pocket_split_4vector_by_ordinal":
                mixture_table(banned, "ordinal", "dominant4"),
            "comp_fused_with_syn_by_ordinal":
                mixture_table(banned, "ordinal", "dominant_fused"),
        },
        "proxy_defs": {
            "META": "pct(presence)",
            "POCKET": "max(pct(opp_usage), pct(player_pool*max(player_wr-0.5,0)))",
            "COMP": "pct(pair_ctr); undefined before any picks lock",
            "multi_motive": f"top two defined vectors within eps={EPS_MAIN}",
        },
    }
    path = DATA_PROCESSED / "ban_attribution.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing["stage_A"] = out
    path.write_text(json.dumps(existing, indent=2))

    keep = ["gameid", "seq", "ordinal", "league", "date", "candidate",
            "dominant", "multi", "dominant4", "dominant_fused",
            "pct_META", "pct_POCKET", "pct_COMP", "pct_SYN"]
    banned[keep].to_parquet(DATA_PROCESSED / "ban_attribution_perban.parquet")

    print("\nmixture by ban ordinal (dominant vector shares):")
    print(pd.DataFrame(out["mixture_by_ban_ordinal"]).T.round(3))
    print("\nlift by ordinal (banned champ mean percentile - 50):")
    print(pd.DataFrame({
        k: {v: d[v]["lift_vs_50"] for v in VECTORS + ["SYN"]}
        for k, d in out["lift_by_ordinal"].items()
    }).T)
    print(f"\nmulti-motive rate (eps=10) by ordinal: "
          f"{out['multi_motive_rate_by_ordinal']['eps=10']}")


if __name__ == "__main__":
    main()
