"""Ban time-signal rung: trailing meta-rates into the transformer output layer.

The transformer's ban top-1 trails the GBM's (7.6 vs 15.0 blind test),
attributed by ROADMAP to date-blindness. This injects the GBM's cheapest
causal meta features — 28-day trailing global pick/ban rates, constant per
(gameid, candidate) — as a slot-type-gated linear bias on the output logits:

    logits[g, slot, champ] += sum_f meta_w[slot_type(slot), f] * meta[g, champ, f]

That is 4 learnable scalars, created deterministically at zero AFTER base
model construction, so (a) training starts from the exact baseline model and
(b) no RNG is consumed — per seed, baseline and meta conditions see identical
init/dropout/shuffle streams and the comparison is strictly paired. Both
conditions build the same model class; the baseline simply gets no "meta"
tensor, which reproduces the current production model bit-for-bit (canary:
seed-16 baseline best val loss must be 3.5855).

Compared over 5 paired seeds on pro VALIDATION ONLY (frozen EWC test set
dropped and never touched). Primary metric: ban val top-1. Guard: pick top-1
must not degrade beyond the seed-noise band. Spec:
docs/2026-07-23-ban-timesignal-handoff.md; scaffolding:
experiment_v09_soloq_transfer.py (train_model/topk_accuracy inlined for the
same lightgbm-import reason).

Meta features are standardized with train-game statistics only; val games
reuse train mean/std. Val-window rates legitimately include train-era games
in their trailing window (causal, not leakage — same as the GBM features).

Output: data/processed/timesignal_rung.json
        data/processed/expcache_timesignal/<condition>_seed<S>.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import DATA_PROCESSED
from draft_transformer import (
    SLOT_TYPE, Config, DraftTransformer, Vocab, attach_scores, build_games,
    masked_loss, probs_for, to_tensors,
)

VAL_DAYS = 14
SEEDS = [16, 17, 42, 7, 23]  # production set, train_draft_model.SEEDS
N_BOOTSTRAP = 10_000
NOISE_BAND = 0.015  # documented seed-noise band, 1.5 top-1 points
BASELINE_CANARY = 3.5855  # seed-16 best val loss, embedding_evolution_v08_demo
CACHE_DIR = DATA_PROCESSED / "expcache_timesignal"


class MetaDraftTransformer(DraftTransformer):
    """DraftTransformer plus a slot-type-gated linear read of per-game meta
    features. meta_w starts at zero (exact baseline model) and is created
    after the base modules, consuming no RNG."""

    def __init__(self, cfg: Config, vocab_size: int, n_leagues: int):
        super().__init__(cfg, vocab_size, n_leagues)
        self.meta_w = nn.Parameter(torch.zeros(2, 2))  # (slot_type, feature)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = super().forward(batch)
        meta = batch.get("meta")  # (n_games, vocab, 2) or absent (baseline)
        if meta is not None:
            w = self.meta_w[SLOT_TYPE.to(logits.device)]  # (20, 2)
            logits = logits + torch.einsum("gvf,sf->gsv", meta, w)
        return logits


def topk_accuracy(df: pd.DataFrame, score_col: str) -> dict:
    """Mean top-k hit rate over decisions, overall and split by picks/bans.
    Copied verbatim from train_draft_model.py (lightgbm import trap)."""
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


def train_model(
    cfg: Config,
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    vocab_size: int,
    n_leagues: int,
) -> tuple[MetaDraftTransformer, float, int, float]:
    """draft_transformer.train_model plus (best_val, epochs, wall_s) in the
    return. The meta condition is expressed purely through a "meta" entry in
    the tensor dicts; when absent, meta_w never receives a gradient and AdamW
    skips it, so the baseline matches the production model bit-for-bit."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model = MetaDraftTransformer(cfg, vocab_size, n_leagues)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_t["prev"].shape[0]
    rng = np.random.RandomState(cfg.seed)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(n)
        # Displayed train loss is inf by construction: 1/100,836 train targets
        # sits outside its availability mask, an infinite constant with no
        # gradient. Cosmetic — do not "fix" masked_loss.
        for i in range(0, n, cfg.batch_size):
            ix = torch.from_numpy(order[i : i + cfg.batch_size])
            batch = {k: v[ix] for k, v in train_t.items()}
            loss = masked_loss(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(masked_loss(model(val_t), val_t))
        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, epoch + 1, time.time() - t0


def build_meta_matrix(
    cand: pd.DataFrame, games: pd.DataFrame, vocab: Vocab,
    val_start: pd.Timestamp, expected_coverage: pd.Series,
) -> tuple[np.ndarray, dict]:
    """(n_games, vocab, 2) float32 of standardized [pick_rate, ban_rate],
    aligned to `games` row order. Rates are constant per (gameid, candidate)
    across slots. A game's candidate table covers the 168 candidates minus any
    series-prior (fearless) champions — those vocab positions keep zero meta,
    which is harmless because they are never available and their logits are
    masked to -inf everywhere. Standardization stats come from train games
    only; special tokens (PAD/START/MISSED) stay at zero."""
    first = cand.drop_duplicates(["gameid", "candidate"])[
        ["gameid", "date", "candidate", "pick_rate", "ban_rate"]
    ]
    # constancy across slots: (gameid, candidate) fully determines the rates
    assert len(first) == len(
        cand.drop_duplicates(["gameid", "candidate", "pick_rate", "ban_rate"])
    ), "pick/ban rates vary across slots within a game"
    per_game = first.groupby("gameid").size()
    bad = per_game.ne(expected_coverage.reindex(per_game.index))
    assert not bad.any(), (
        f"candidate coverage mismatch in {int(bad.sum())} games, e.g. "
        f"{per_game[bad].head(3).to_dict()}"
    )

    tr = first[first.date < val_start]
    mu = tr[["pick_rate", "ban_rate"]].mean()
    sd = tr[["pick_rate", "ban_rate"]].std()

    pos = {g: i for i, g in enumerate(games.gameid)}
    m = np.zeros((len(games), vocab.size, 2), dtype=np.float32)
    gi = first.gameid.map(pos).to_numpy()
    ci = first.candidate.map(vocab.id_of).to_numpy()
    m[gi, ci, 0] = (first.pick_rate - mu.pick_rate) / sd.pick_rate
    m[gi, ci, 1] = (first.ban_rate - mu.ban_rate) / sd.ban_rate
    stats = {"train_mean": [round(float(v), 6) for v in mu],
             "train_std": [round(float(v), 6) for v in sd]}
    return m, stats


def main() -> None:
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate",
                 "label", "pick_rate", "ban_rate"],
    )
    cand["date"] = pd.to_datetime(cand["date"])

    # Same split rule as experiment_v08.split_dates; EWC July 2026 test games
    # dropped entirely, never evaluated.
    def drop_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
        is_test = ((df.league == "EWC") & (df.date.dt.month == 7)
                   & (df.date.dt.year == 2026))
        return df[~is_test & (df.date < df.loc[is_test, "date"].min())], \
            df.loc[is_test, "date"].min()

    seq, cutoff = drop_test(seq)
    cand, _ = drop_test(cand)
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)

    vocab = Vocab(list(cand.candidate.unique()), list(seq.champion.unique()))
    games = build_games(seq, vocab)
    n_leagues = len(games.attrs["leagues"])
    train_mask = (games.date < val_start).to_numpy()
    g_train = games[train_mask].reset_index(drop=True)
    g_val = games[~train_mask].reset_index(drop=True)
    t_train, t_val = to_tensors(g_train), to_tensors(g_val)
    val_pos = {g: i for i, g in enumerate(g_val.gameid)}
    val_rows = cand[cand.date >= val_start].copy()
    n_val_dec = val_rows.groupby(["gameid", "seq"]).ngroups
    n_train_dec = int((t_train["target"] != -100).sum())
    print(f"cutoff {cutoff.date()}: train {len(g_train)}g/{n_train_dec}d, "
          f"val {len(g_val)}g/{n_val_dec}d, vocab {vocab.size}")
    assert (len(g_train), n_train_dec) == (5043, 100836)
    assert (len(g_val), n_val_dec) == (54, 1080)

    # Fearless games exclude series-prior champions from the candidate table:
    # coverage = 168 - |prior bans that are candidates|.
    candidate_set = set(vocab.champs)
    expected_coverage = seq.groupby("gameid").series_prior.first().map(
        lambda p: len(candidate_set)
        - (len({c for c in p.split("|") if c} & candidate_set) if p else 0)
    )
    meta_all, meta_stats = build_meta_matrix(
        cand, games, vocab, val_start, expected_coverage
    )
    meta_train = torch.from_numpy(meta_all[train_mask])
    meta_val = torch.from_numpy(meta_all[~train_mask])
    print(f"meta matrix {meta_all.shape}, train stats {meta_stats}")

    cfg_base = dict(d_model=192, n_layers=4, n_heads=6)
    CACHE_DIR.mkdir(exist_ok=True)

    runs: list[dict] = []
    for seed in SEEDS:
        for condition in ("baseline", "meta"):
            cache = CACHE_DIR / f"{condition}_seed{seed}.json"
            if cache.exists():
                run = json.loads(cache.read_text())
                print(f"[cached] {condition} seed={seed} "
                      f"val_loss={run['best_val_loss']:.4f} "
                      f"ban_top1={run['bans']['top1']:.4f}")
                runs.append(run)
                continue
            cfg = Config(**cfg_base, seed=seed)
            if condition == "meta":
                tr_t = {**t_train, "meta": meta_train}
                va_t = {**t_val, "meta": meta_val}
            else:
                tr_t, va_t = t_train, t_val
            model, best_val, epochs, wall = train_model(
                cfg, tr_t, va_t, vocab.size, n_leagues,
            )
            probs = probs_for(model, va_t).numpy()
            val_rows["score"] = attach_scores(val_rows, probs, val_pos, vocab)
            acc = topk_accuracy(val_rows, "score")
            run = {
                "condition": condition, "seed": seed,
                "best_val_loss": round(best_val, 4),
                "top1": acc["all"]["top1"], "top3": acc["all"]["top3"],
                "top5": acc["all"]["top5"],
                "picks": acc["picks"], "bans": acc["bans"],
                "epochs": epochs, "wall_s": round(wall, 1),
            }
            if condition == "meta":
                run["meta_w"] = [
                    [round(float(v), 4) for v in row]
                    for row in model.meta_w.detach().tolist()
                ]
            cache.write_text(json.dumps(run))
            print(f"{condition} seed={seed} val_loss={best_val:.4f} "
                  f"ban_top1={acc['bans']['top1']:.4f} "
                  f"pick_top1={acc['picks']['top1']:.4f} "
                  f"({epochs} epochs, {wall:.0f}s)", flush=True)
            runs.append(run)
            if condition == "baseline" and seed == 16:
                assert round(best_val, 4) == BASELINE_CANARY, (
                    f"harness drift: seed-16 baseline val loss {best_val:.4f} "
                    f"!= canary {BASELINE_CANARY}"
                )

    by = {(r["condition"], r["seed"]): r for r in runs}
    d_loss = np.array([by[("baseline", s)]["best_val_loss"]
                       - by[("meta", s)]["best_val_loss"] for s in SEEDS])
    d_ban = np.array([by[("meta", s)]["bans"]["top1"]
                      - by[("baseline", s)]["bans"]["top1"] for s in SEEDS])
    d_pick = np.array([by[("meta", s)]["picks"]["top1"]
                       - by[("baseline", s)]["picks"]["top1"] for s in SEEDS])
    rng = np.random.default_rng(0)
    boot = rng.choice(d_loss, size=(N_BOOTSTRAP, len(d_loss))).mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    n_pos = int((d_loss > 0).sum())
    go_ban = float(d_ban.mean()) >= NOISE_BAND
    go_loss = ci[0] > 0
    pick_ok = float(d_pick.mean()) >= -NOISE_BAND
    verdict = "GO" if (go_ban or go_loss) and pick_ok else "NO-GO"

    out = {
        "note": ("Ban time-signal rung: 28-day trailing global pick/ban rates "
                 "injected at the transformer output layer (4 slot-type-gated "
                 "scalars), baseline vs meta, 5 paired seeds, pro validation "
                 "only. Primary metric: ban val top-1."),
        "runs": runs,
        "baseline_ban_top1_mean": round(
            float(np.mean([by[("baseline", s)]["bans"]["top1"] for s in SEEDS])), 4
        ),
        "meta_ban_top1_mean": round(
            float(np.mean([by[("meta", s)]["bans"]["top1"] for s in SEEDS])), 4
        ),
        "paired_deltas": {
            "seeds": SEEDS,
            "d_ban_top1 (meta - baseline, >0 favors meta)": d_ban.round(4).tolist(),
            "d_pick_top1 (meta - baseline)": d_pick.round(4).tolist(),
            "d_loss (baseline - meta, >0 favors meta)": d_loss.round(4).tolist(),
            "mean_d_ban_top1": round(float(d_ban.mean()), 4),
            "mean_d_pick_top1": round(float(d_pick.mean()), 4),
            "mean_d_loss": round(float(d_loss.mean()), 4),
            "d_loss_positive_seeds": f"{n_pos}/{len(SEEDS)}",
            "bootstrap_95ci_mean_d_loss": [round(c, 4) for c in ci],
        },
        "verdict": verdict,
        "verdict_detail": {
            "mean_d_ban_top1_ge_1.5pts": go_ban,
            "loss_ci_excludes_0_favorably": go_loss,
            "pick_top1_within_noise_band": pick_ok,
        },
        "provenance": {
            "features": "28-day trailing global pick_rate/ban_rate "
                        "(draft_decisions_multi, causal by construction)",
            "standardization": {"population": "train games only", **meta_stats},
            "injection": "output-layer bias, meta_w[slot_type, feature], "
                         "zero-init, no RNG consumed",
            "split_cutoff": str(cutoff.date()), "val_days": VAL_DAYS,
            "config": Config(**cfg_base).tag(), "device": "cpu",
            "seeds": SEEDS, "bootstrap_resamples": N_BOOTSTRAP,
            "baseline_canary_seed16": BASELINE_CANARY,
        },
    }
    (DATA_PROCESSED / "timesignal_rung.json").write_text(json.dumps(out, indent=2))
    print(f"\nbaseline ban top-1 mean: {out['baseline_ban_top1_mean']:.4f}  "
          f"meta: {out['meta_ban_top1_mean']:.4f}")
    print(f"d_ban_top1 per seed: {d_ban.round(4).tolist()}")
    print(f"d_pick_top1 per seed: {d_pick.round(4).tolist()}")
    print(f"d_loss per seed: {d_loss.round(4).tolist()}  ({n_pos}/5 positive)")
    print(f"mean d_loss={d_loss.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
