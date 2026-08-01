"""Post-hoc diagnostics on the pocket-conditioning rung's cached val probs.

The rung (experiment_v13_pocket_conditioning.py) came back NO-GO on a guard
breach: bans improved on every seed while pick top-1 fell ~4 pts. This script
answers *where* each effect lives, using only the cached probability stacks in
data/processed/expcache_pocket (no training, no test look).

Reported:
  1. val NLL decomposed into ban and pick halves (the headline loss gain of
     0.162 nats is an average over both) — per seed and paired.
  2. pick top-1/top-5 by pick ordinal, so a uniform degradation can be told
     apart from one concentrated in the early, context-poor picks.
  3. argmax-composition: how often each condition's top-ranked candidate is
     also the decision's max-player_pool / max-opp_usage candidate, against
     the same rate for the champion actually chosen. This measures how far the
     injection drags the argmax toward the roster-favourite.

VAL ONLY — frozen EWC test games are dropped exactly as the rung does.

Output: data/processed/pocket_conditioning_diagnostics.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from draft_transformer import SEQ_TO_SLOT, Vocab, build_games
from experiment_v13_pocket_conditioning import (
    CACHE_DIR, FEATURES, SEEDS, VAL_DAYS,
)

CONDITIONS = ("baseline", "feat")


def load_val_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    """Rebuild the rung's val candidate rows and game ordering."""
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "ordinal",
                 "candidate", "label"] + FEATURES,
    )
    cand["date"] = pd.to_datetime(cand["date"])

    def drop_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
        is_test = ((df.league == "EWC") & (df.date.dt.month == 7)
                   & (df.date.dt.year == 2026))
        return df[~is_test & (df.date < df.loc[is_test, "date"].min())], \
            df.loc[is_test, "date"].min()

    seq, cutoff = drop_test(seq)
    cand, _ = drop_test(cand)
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)

    vocab = Vocab(list(cand.candidate.unique()), list(seq.champion.unique()))
    assert vocab.size == 171
    games = build_games(seq, vocab)
    g_val = games[~(games.date < val_start).to_numpy()].reset_index(drop=True)
    val_pos = {g: i for i, g in enumerate(g_val.gameid)}
    val_rows = cand[cand.date >= val_start].copy()
    assert (len(g_val), val_rows.groupby(["gameid", "seq"]).ngroups) == (54, 1080)

    val_rows["gi"] = val_rows.gameid.map(val_pos).to_numpy()
    val_rows["si"] = val_rows.seq.map(SEQ_TO_SLOT).to_numpy()
    val_rows["ci"] = val_rows.candidate.map(vocab.id_of).to_numpy()
    return val_rows, val_pos


def per_decision(val_rows: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """One row per val decision: NLL of the true champion and the rank the
    model gave it, plus which candidate the model ranked first."""
    p = probs[val_rows.gi.to_numpy(), val_rows.si.to_numpy(),
              val_rows.ci.to_numpy()]
    df = val_rows.assign(p=p)
    recs = []
    for (gid, s), g in df.groupby(["gameid", "seq"], sort=False):
        pv = g.p.to_numpy()
        lab = g.label.to_numpy().astype(bool)
        order = np.argsort(-pv, kind="stable")
        top = g.iloc[order[0]]
        recs.append({
            "gameid": gid, "seq": s,
            "is_ban": int(g.is_ban.iloc[0]), "ordinal": int(g.ordinal.iloc[0]),
            "nll": float(-np.log(max(pv[lab][0], 1e-12))),
            "rank": int(np.argmax(lab[order])),
            "argmax_cand": top.candidate,
            # is the model's #1 also the decision's roster-favourite?
            **{f"argmax_is_max_{f}": int(top[f] == g[f].max()) for f in FEATURES},
            # ...and is the champion actually chosen?
            **{f"truth_is_max_{f}": int(g.loc[lab, f].iloc[0] == g[f].max())
               for f in FEATURES},
        })
    return pd.DataFrame(recs)


def summarize(d: pd.DataFrame) -> dict:
    return {
        "n": len(d),
        "nll": round(float(d.nll.mean()), 4),
        "top1": round(float((d["rank"] < 1).mean()), 4),
        "top5": round(float((d["rank"] < 5).mean()), 4),
    }


def main() -> None:
    val_rows, _ = load_val_rows()
    dec = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            probs = np.load(CACHE_DIR / f"{cond}_seed{seed}_probs.npz")["probs"]
            dec[(cond, seed)] = per_decision(val_rows, probs)
        # 5-seed ensemble (mean probs, as the rung's conditional table does)
        mean_probs = np.mean(
            [np.load(CACHE_DIR / f"{cond}_seed{s}_probs.npz")["probs"]
             for s in SEEDS], axis=0)
        dec[(cond, "ens")] = per_decision(val_rows, mean_probs)

    cached = {
        (c, s): json.loads((CACHE_DIR / f"{c}_seed{s}.json").read_text())
        for c in CONDITIONS for s in SEEDS
    }

    # --- 1. loss decomposition -------------------------------------------
    loss = {"per_seed": [], "note": (
        "NLL recomputed from cached probs; 'all' reproduces the rung's "
        "best_val_loss, confirming the cache matches the reported run.")}
    for seed in SEEDS:
        row = {"seed": seed}
        for cond in CONDITIONS:
            d = dec[(cond, seed)]
            row[cond] = {
                "all": round(float(d.nll.mean()), 4),
                "bans": round(float(d[d.is_ban == 1].nll.mean()), 4),
                "picks": round(float(d[d.is_ban == 0].nll.mean()), 4),
                "reported_best_val_loss": cached[(cond, seed)]["best_val_loss"],
            }
        row["d_nll_bans"] = round(row["baseline"]["bans"] - row["feat"]["bans"], 4)
        row["d_nll_picks"] = round(row["baseline"]["picks"] - row["feat"]["picks"], 4)
        loss["per_seed"].append(row)
    for k in ("d_nll_bans", "d_nll_picks"):
        loss[f"mean_{k}"] = round(
            float(np.mean([r[k] for r in loss["per_seed"]])), 4)

    # --- 2. pick degradation by ordinal ----------------------------------
    by_ordinal = {}
    for kind, is_ban in (("picks", 0), ("bans", 1)):
        rows = {}
        for o in sorted(dec[("baseline", "ens")].query("is_ban == @is_ban").ordinal.unique()):
            entry = {}
            for cond in CONDITIONS:
                per_seed = [
                    summarize(dec[(cond, s)].query(
                        "is_ban == @is_ban and ordinal == @o"))
                    for s in SEEDS
                ]
                entry[cond] = {
                    "seed_mean_top1": round(
                        float(np.mean([x["top1"] for x in per_seed])), 4),
                    "seed_mean_top5": round(
                        float(np.mean([x["top5"] for x in per_seed])), 4),
                    "ens_top1": summarize(dec[(cond, "ens")].query(
                        "is_ban == @is_ban and ordinal == @o"))["top1"],
                }
            entry["n"] = summarize(dec[("baseline", "ens")].query(
                "is_ban == @is_ban and ordinal == @o"))["n"]
            entry["d_seed_mean_top1"] = round(
                entry["feat"]["seed_mean_top1"] - entry["baseline"]["seed_mean_top1"], 4)
            rows[f"ordinal_{int(o)}"] = entry
        by_ordinal[kind] = rows

    # --- 3. argmax composition -------------------------------------------
    argmax = {}
    for kind, is_ban in (("picks", 0), ("bans", 1)):
        e = {"truth": {}, **{c: {} for c in CONDITIONS}}
        base = dec[("baseline", "ens")].query("is_ban == @is_ban")
        for f in FEATURES:
            e["truth"][f] = round(float(base[f"truth_is_max_{f}"].mean()), 4)
            for cond in CONDITIONS:
                vals = [
                    float(dec[(cond, s)].query("is_ban == @is_ban")
                          [f"argmax_is_max_{f}"].mean()) for s in SEEDS
                ]
                e[cond][f] = round(float(np.mean(vals)), 4)
        argmax[kind] = e

    out = {
        "note": ("Diagnostics for docs/2026-07-28-pocket-conditioning-results.md. "
                 "Cached val probs only — no training, no test look."),
        "loss_decomposition": loss,
        "by_ordinal": by_ordinal,
        "argmax_composition": {
            "note": ("Share of decisions whose top-ranked candidate is also the "
                     "decision's maximum on each feature; 'truth' is the same "
                     "rate for the champion actually chosen. Seed-averaged."),
            **argmax,
        },
    }
    path = DATA_PROCESSED / "pocket_conditioning_diagnostics.json"
    path.write_text(json.dumps(out, indent=2))

    print("loss decomposition (mean over 5 seeds, >0 favors feat):")
    print(f"  d_nll bans  {loss['mean_d_nll_bans']:+.4f}")
    print(f"  d_nll picks {loss['mean_d_nll_picks']:+.4f}")
    for r in loss["per_seed"]:
        chk = all(abs(r[c]["all"] - r[c]["reported_best_val_loss"]) < 5e-3
                  for c in CONDITIONS)
        print(f"  seed {r['seed']:<3} bans {r['baseline']['bans']:.3f}->"
              f"{r['feat']['bans']:.3f}  picks {r['baseline']['picks']:.3f}->"
              f"{r['feat']['picks']:.3f}  (cache matches reported: {chk})")
    for kind in ("picks", "bans"):
        print(f"\n{kind} top-1 by ordinal (seed mean, baseline -> feat):")
        for k, v in by_ordinal[kind].items():
            print(f"  {k:<12} n={v['n']:<4} "
                  f"{v['baseline']['seed_mean_top1']*100:5.1f} -> "
                  f"{v['feat']['seed_mean_top1']*100:5.1f}  "
                  f"({v['d_seed_mean_top1']*100:+.1f})")
    print("\nargmax is the decision's feature-max candidate (share):")
    for kind in ("picks", "bans"):
        for f in FEATURES:
            a = argmax[kind]
            print(f"  {kind:<6} {f:<12} truth {a['truth'][f]*100:5.1f}  "
                  f"baseline {a['baseline'][f]*100:5.1f}  "
                  f"feat {a['feat'][f]*100:5.1f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
