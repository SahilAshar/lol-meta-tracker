"""Rung 2 arm 1 (mimic): soloq-informed candidate features in the draft GBM.

Executes docs/2026-07-25-rung2-transfer-spec.md arm 1. The regenerated
draft_decisions.parquet (draft_dataset.py) carries two new candidate-level
features from the static soloq lift tables (cut 2026-07-01, soloq coverage
2026-06-08 onward): soloq_role_wr (role-aware champ-strength prior) and
soloq_ctr_score (expected same-role counter lift vs the other side's locked
picks). Decisions before 2026-06-08 hold neutral values by construction.

Variants, each a full v0.7 ensemble (5 seeds x {hist-GBM clf, LGBM
lambdarank}, equal-weight rank blend, train_draft_model.py config):

  baseline: the production v0.7 feature set, refit here so the comparison is
            same-platform, same-data paired (stored v0.7 trained on Linux;
            single fits shift ~1.5pts top-1 across platforms).
  V1:       baseline + soloq_ctr_score + soloq_role_wr (additive).
  V2:       V1 with the thin pro pair_ctr removed (does soloq subsume it?).
  V3:       V1 + soloq_syn_score (rung 1c PASS branch — the optional duo-lift
            aggregate over the reference team's locked picks; soloq-side
            effect ~1/10 of the counter effect, expectations modest).

Gate: ensemble val top-3 improvement over baseline with a paired bootstrap
95% CI excluding 0 (10k resamples over the 1,080 val decisions, paired per
decision). Top-1/top-5 reported alongside; per-family numbers reported. The
frozen EWC test set is never loaded into any scorer.

Before training, the script verifies the regenerated parquet kept every
pre-existing column byte-identical on all pre-cutoff rows against the backup
taken before the rebuild (--backup).

Writes data/processed/rung2_gbm.json; per-fit val scores cached under
data/processed/expcache_rung2gbm/.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from soloq_lift_tables import load_tables, pro_arrays
from train_draft_model import (
    FEATURES, SEEDS, VAL_DAYS, ensemble_score, fit_clf, fit_ranker,
)

NEW_FEATURES = ["soloq_ctr_score", "soloq_role_wr"]
CONFIGS = {
    "baseline": FEATURES,
    "V1": FEATURES + NEW_FEATURES,
    "V2": [f for f in FEATURES if f != "pair_ctr"] + NEW_FEATURES,
    "V3": FEATURES + NEW_FEATURES + ["soloq_syn_score"],
}
N_BOOT = 10_000
CACHE_DIR = DATA_PROCESSED / "expcache_rung2gbm"
RNG = np.random.default_rng(20260725)


def verify_backup(ds: pd.DataFrame, backup_path: str, cutoff) -> dict:
    old = pd.read_parquet(backup_path)
    old["date"] = pd.to_datetime(old["date"])
    key = ["gameid", "seq", "candidate"]
    new_pre = ds[ds.date < cutoff].sort_values(key).reset_index(drop=True)
    old_pre = old[old.date < cutoff].sort_values(key).reset_index(drop=True)
    assert len(new_pre) == len(old_pre), \
        f"pre-cutoff row count changed: {len(old_pre)} -> {len(new_pre)}"
    assert (new_pre[key].to_numpy() == old_pre[key].to_numpy()).all(), \
        "pre-cutoff row identity changed"
    diffs = [c for c in old.columns
             if not np.array_equal(new_pre[c].to_numpy(),
                                   old_pre[c].to_numpy())]
    assert not diffs, f"pre-cutoff columns not byte-identical: {diffs}"
    print(f"backup check OK: {len(new_pre)} pre-cutoff rows, "
          f"{len(old.columns)} pre-existing columns byte-identical")
    return {"pre_cutoff_rows": len(new_pre),
            "columns_checked": len(old.columns), "byte_identical": True}


def decision_ranks(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Rank (0-based) of the actually-chosen champion per decision."""
    def hit_rank(g: pd.DataFrame) -> int:
        order = np.argsort(-g[score_col].to_numpy(), kind="stable")
        return int(np.argmax(g["label"].to_numpy()[order]))
    return df.groupby(["gameid", "seq"], sort=False).apply(
        hit_rank, include_groups=False)


def summarize(ranks: np.ndarray) -> dict:
    return {"n": len(ranks),
            "top1": round(float((ranks < 1).mean()), 4),
            "top3": round(float((ranks < 3).mean()), 4),
            "top5": round(float((ranks < 5).mean()), 4)}


def paired_bootstrap(d: np.ndarray) -> dict:
    n = len(d)
    means = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return {"mean_delta": round(float(d.mean()), 6),
            "ci95": [round(float(np.quantile(means, 0.025)), 6),
                     round(float(np.quantile(means, 0.975)), 6)],
            "p_variant_better": round(float((means > 0).mean()), 4)}


def family_scores(features: list[str], train: pd.DataFrame,
                  val: pd.DataFrame, tag: str) -> dict[str, np.ndarray]:
    """Val pct-rank scores per family, cached per (config, family, seed)."""
    CACHE_DIR.mkdir(exist_ok=True)
    out = {}
    for fam, fit, predict in (
        ("clf", fit_clf, lambda m, x: m.predict_proba(x)[:, 1]),
        ("rank", fit_ranker, lambda m, x: m.predict(x)),
    ):
        per_seed = []
        for seed in SEEDS:
            cache = CACHE_DIR / f"{tag}_{fam}_seed{seed}.npy"
            if cache.exists():
                per_seed.append(np.load(cache))
                continue
            t0 = time.time()
            model = fit(features, train, seed)
            raw = predict(model, val[features])
            pct = pd.Series(raw).rank(pct=True).to_numpy()
            np.save(cache, pct)
            per_seed.append(pct)
            print(f"  {tag}/{fam} seed={seed}: {time.time() - t0:.0f}s",
                  flush=True)
        out[fam] = np.mean(per_seed, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", default=None,
                        help="pre-rebuild parquet for the byte-identity check")
    args = parser.parse_args()
    t0 = time.time()

    ds = pd.read_parquet(DATA_PROCESSED / "draft_decisions.parquet")
    ds["date"] = pd.to_datetime(ds["date"])
    is_test = (ds.league == "EWC") & (ds.date.dt.month == 7)
    cutoff = ds.loc[is_test, "date"].min()

    backup_check = (verify_backup(ds, args.backup, cutoff)
                    if args.backup else {"skipped": True})

    # Frozen EWC test rows: dropped immediately, never scored.
    pre = ds[~is_test & (ds.date < cutoff)]
    del ds
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)
    train = pre[pre.date < val_start]
    val = pre[pre.date >= val_start].copy()
    n_val_dec = val.groupby(["gameid", "seq"]).ngroups
    print(f"cutoff {cutoff}; train {train.gameid.nunique()}g/"
          f"{train.groupby(['gameid', 'seq']).ngroups}d, "
          f"val {val.gameid.nunique()}g/{n_val_dec}d")
    assert n_val_dec == 1080, f"val decisions {n_val_dec} != 1080 (spec anchor)"

    champs = sorted(pre.candidate.unique())
    _, _, _, coverage = pro_arrays(champs, load_tables())
    print(f"bridge coverage: {coverage['matched']}/{coverage['pro_champs']} "
          f"(misses: {coverage['misses']})")

    results, rank_vecs = {}, {}
    order = None
    for tag, feats in CONFIGS.items():
        print(f"training {tag} ({len(feats)} features)...", flush=True)
        fams = family_scores(feats, train, val, tag)
        val["score_ens"] = (fams["clf"] + fams["rank"]) / 2.0
        val["score_clf"] = fams["clf"]
        val["score_rank"] = fams["rank"]
        per_scorer = {}
        for scorer in ("ens", "clf", "rank"):
            ranks = decision_ranks(val, f"score_{scorer}")
            if order is None:
                order = ranks.index
            ranks = ranks.reindex(order)
            per_scorer[scorer] = ranks.to_numpy()
        rank_vecs[tag] = per_scorer
        results[tag] = {
            "features": feats,
            "ensemble": summarize(per_scorer["ens"]),
            "clf_family": summarize(per_scorer["clf"]),
            "ranker_family": summarize(per_scorer["rank"]),
        }
        print(f"  {tag}: ens {results[tag]['ensemble']} | "
              f"clf {results[tag]['clf_family']} | "
              f"rank {results[tag]['ranker_family']}", flush=True)

    gates = {}
    base = rank_vecs["baseline"]["ens"]
    for tag in ("V1", "V2", "V3"):
        v = rank_vecs[tag]["ens"]
        gates[tag] = {
            f"top{k}": paired_bootstrap((v < k).astype(float)
                                        - (base < k).astype(float))
            for k in (1, 3, 5)}
        gates[tag]["pass"] = gates[tag]["top3"]["ci95"][0] > 0
        print(f"gate {tag} vs baseline: top3 {gates[tag]['top3']} -> "
              f"{'PASS' if gates[tag]['pass'] else 'FAIL'}", flush=True)
    # SYN increment (descriptive): V3 vs V1, per top-k.
    gates["syn_increment_V3_vs_V1"] = {
        f"top{k}": paired_bootstrap(
            (rank_vecs["V3"]["ens"] < k).astype(float)
            - (rank_vecs["V1"]["ens"] < k).astype(float))
        for k in (1, 3, 5)}
    print(f"syn increment (V3 vs V1): "
          f"{gates['syn_increment_V3_vs_V1']['top3']}", flush=True)

    arm_pass = any(gates[t]["pass"] for t in ("V1", "V2", "V3"))
    stored = json.loads(
        (DATA_PROCESSED / "draft_model_metrics_v07.json").read_text())
    out = {
        "experiment": "rung2_arm1_gbm",
        "spec": "docs/2026-07-25-rung2-transfer-spec.md",
        "question": ("do soloq-informed candidate features (role-aware "
                     "strength, lane counters) improve the v0.7 draft GBM's "
                     "val top-3?"),
        "gate": {"criterion": "ensemble val top-3 vs refit baseline, paired "
                              "bootstrap 95% CI excluding 0, either variant",
                 "pass": bool(arm_pass), "per_variant": gates},
        "models": results,
        "baseline_reproduction": {
            "stored_v07_val": stored["val"]["model"]["all"],
            "stored_platform": stored.get("platform"),
            "note": "stored v0.7 trained on Linux; baseline refit here on "
                    "this platform so the gate comparison is paired",
        },
        "timing_caveat": (
            "MANDATORY READING (pre-registered): soloq coverage starts "
            "2026-06-08; pro train decisions before that hold zero/neutral "
            "values for the new features, so the GBM can only learn them "
            "from the late-window train tail. Val is entirely post-07-01, so "
            "the gate itself is clean, but train-time sparsity may mute what "
            "the GBM learns. If the result is an ambiguous null, the "
            "follow-up is 're-run when soloq spans a full split', not a "
            "rerun of the same window."),
        "provenance": {
            "cutoff": str(cutoff),
            "val_days": VAL_DAYS,
            "n_val_decisions": n_val_dec,
            "seeds": SEEDS,
            "n_bootstrap": N_BOOT,
            "frozen_ewc_test": "never loaded into any scorer",
            "backup_check": backup_check,
            "soloq_table": {k: load_tables()[k]
                            for k in ("cutoff_iso", "clean_games",
                                      "blue_win_rate", "n_champs",
                                      "n_counter_pairs", "n_syn_pairs")},
            "bridge_coverage": coverage,
            "rung1c": ("PASS — soloq_syn_score added as V3 per the spec's "
                       "PASS branch (docs/2026-07-25-synergy-rung1c-"
                       "results.md); no LOO needed, pro games are disjoint "
                       "from the soloq table sample"),
            "soloq_feature_construction": (
                "soloq_role_wr = E_r[soloq_wr(cand@r)] under P(r) ∝ "
                "cand role-shares x ref-team open-role probs (normalized); "
                "soloq_ctr_score = sum over other side's locked picks of "
                "E[ctr lift] under P(cand@r) x P(pick@r), pick roles "
                "marginalized by permutation enumeration; soloq_syn_score = "
                "sum over the REFERENCE team's locked picks of E[duo lift] "
                "over the four priority vectors, candidate in either slot, "
                "same probability weighting; decisions before 2026-06-08 "
                "neutral (0.5 / 0.0 / 0.0)"),
            "runtime_seconds": round(time.time() - t0),
        },
    }
    path = DATA_PROCESSED / "rung2_gbm.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({round(time.time() - t0)}s total)")


if __name__ == "__main__":
    main()
