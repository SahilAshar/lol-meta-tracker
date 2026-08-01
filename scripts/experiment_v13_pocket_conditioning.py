"""Team/player conditioning rung: opponent-referenced pocket features, input-side.

Handoff: docs/2026-07-25-next-rungs-handoff.md Mission 2. Stage B of the
attribution study showed POCKET is 41% of all bans and the transformer's
worst vector (12.2 vs 5.4 GBM top-1 overall; 13.9 vs 2.6 in B1-3) — pocket
denial is structurally invisible to a model with zero roster identity. This
injects the candidate table's opponent-referenced features — opp_usage,
player_pool, player_wr — causally and train-standardized, like the meta
rates, but NOT as output scalars alone (Stage B finding 3: an output-layer
bias re-ranks the shortlist but cannot sharpen the argmax).

Unlike the meta rates these features are per (gameid, seq, candidate): ban
rows reference the opponent's roster, pick rows the own team's, and
role-open probabilities evolve through the draft (draft_dataset.py
docstring). The feature tensor is (n_games, 20, vocab, 3); positions with
no candidate row stay 0 (they are masked to -inf anyway).

Injection, all new params zero-init AFTER base construction, no RNG:

  input-side  ctx[g,s,f,:] = sum_v feat[g,s,v,f] * champ_emb[v]
              x[g,s] += reshape(ctx, 3*d) @ feat_proj          (3d x d, zeros)
  output-side logits[g,s,v] += feat[g,s,v,:] . feat_out[slot_type(s)]
                                                               (2 x 3, zeros)

The input path is the structural fix: with the weight-tied head, pushing the
hidden state toward the feature-weighted embedding centroid ("what the
opponent is dangerous on here") moves the argmax, even at B1 where there is
no draft context to attend over. The output path keeps the cheap linear
re-rank available so the projection does not have to carry it.

Protocol (experiment_v11_ban_timesignal.py scaffolding): 5 paired seeds,
baseline canary 3.5855 hard-asserted (plus a bit-identity check of baseline
val probs against expcache_attr when present), primary metric ban val
top-1, guard pick top-1 within +/-1.5pts, GO = mean d_ban >= +1.5pts or
loss CI excluding 0. Additionally reports the Stage-B-style conditional
table (ban_attribution_perban.parquet join): the rung succeeds *for the
right reason* if the gain concentrates in POCKET and B1-3 cells.

VAL ONLY — frozen EWC test games dropped, never scored.

Output: data/processed/pocket_conditioning_rung.json
        data/processed/expcache_pocket/<condition>_seed<S>.json (+ feat .pt)
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
    SEQ_TO_SLOT, SLOT_SIDE, SLOT_TYPE, Config, DraftTransformer, Vocab,
    attach_scores, build_games, masked_loss, probs_for, to_tensors,
)

VAL_DAYS = 14
SEEDS = [16, 17, 42, 7, 23]
N_BOOTSTRAP = 10_000
NOISE_BAND = 0.015
BASELINE_CANARY = 3.5855  # seed-16 v0.8-lineage best val loss
FEATURES = ["opp_usage", "player_pool", "player_wr"]
CACHE_DIR = DATA_PROCESSED / "expcache_pocket"
ATTR_CACHE = DATA_PROCESSED / "expcache_attr"


class PocketDraftTransformer(DraftTransformer):
    """DraftTransformer plus input-side and output-side reads of per-slot
    per-candidate pocket features. Both new params start at zero (exact
    baseline model) and are created after the base modules, consuming no
    RNG; without a "feat" batch entry the forward is bit-identical to the
    base model (same ops, same dropout draws)."""

    def __init__(self, cfg: Config, vocab_size: int, n_leagues: int):
        super().__init__(cfg, vocab_size, n_leagues)
        d = cfg.d_model
        self.feat_proj = nn.Parameter(torch.zeros(len(FEATURES) * d, d))
        self.feat_out = nn.Parameter(torch.zeros(2, len(FEATURES)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prev = batch["prev"]
        slots = torch.arange(20, device=prev.device)
        x = (
            self.champ_emb(prev)
            + self.pos_emb(slots)[None]
            + self.type_emb(SLOT_TYPE.to(prev.device))[None]
            + self.side_emb(SLOT_SIDE.to(prev.device))[None]
            + (
                self.league_emb(batch["league"])
                + self.fearless_emb(batch["fearless"])
                + self.gis_emb(batch["gis"])
            )[:, None, :]
        )
        feat = batch.get("feat")  # (n, 20, vocab, F) or absent (baseline)
        if feat is not None:
            ctx = torch.einsum("gsvf,vd->gsfd", feat, self.champ_emb.weight)
            x = x + ctx.reshape(*ctx.shape[:2], -1) @ self.feat_proj
        h = self.norm(self.encoder(self.drop(x), mask=self.causal))
        logits = h @ self.champ_emb.weight.T + self.out_bias
        if feat is not None:
            w = self.feat_out[SLOT_TYPE.to(logits.device)]  # (20, F)
            logits = logits + torch.einsum("gsvf,sf->gsv", feat, w)
        return logits


def topk_accuracy(df: pd.DataFrame, score_col: str) -> dict:
    """Copied verbatim from train_draft_model.py (lightgbm import trap)."""
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
) -> tuple[PocketDraftTransformer, float, int, float]:
    """experiment_v11 train_model with the pocket model class. The feat
    condition is expressed purely through a "feat" tensor-dict entry; when
    absent the new params get no gradient and AdamW skips them, so the
    baseline matches the production model bit-for-bit."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model = PocketDraftTransformer(cfg, vocab_size, n_leagues)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_t["prev"].shape[0]
    rng = np.random.RandomState(cfg.seed)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(n)
        # Displayed train loss is inf by construction (one off-mask target);
        # cosmetic — do not "fix" masked_loss.
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


def build_feat_tensor(
    cand: pd.DataFrame, games: pd.DataFrame, vocab: Vocab,
    val_start: pd.Timestamp,
) -> tuple[np.ndarray, dict]:
    """(n_games, 20, vocab, F) float32 of train-standardized pocket features,
    aligned to `games` row order. Unlike the meta rates these vary per
    (gameid, seq, candidate) — each decision's rows reference the relevant
    roster (opponent for bans, own for picks) and the evolving open-role
    probabilities — so the scatter is per slot. Positions with no candidate
    row stay exactly 0 (never available, logits masked to -inf)."""
    tr = cand[cand.date < val_start]
    mu = tr[FEATURES].mean()
    sd = tr[FEATURES].std()

    pos = {g: i for i, g in enumerate(games.gameid)}
    gi = cand.gameid.map(pos).to_numpy()
    si = cand.seq.map(SEQ_TO_SLOT).to_numpy()
    ci = cand.candidate.map(vocab.id_of).to_numpy()
    assert not (np.isnan(gi).any() or np.isnan(ci).any())
    m = np.zeros((len(games), 20, vocab.size, len(FEATURES)), dtype=np.float32)
    for k, f in enumerate(FEATURES):
        z = ((cand[f] - mu[f]) / sd[f]).to_numpy(dtype=np.float32)
        assert not np.isnan(z).any(), f"NaNs in standardized {f}"
        m[gi, si, ci, k] = z
    stats = {"train_mean": [round(float(v), 6) for v in mu],
             "train_std": [round(float(v), 6) for v in sd]}
    return m, stats


def conditional_table(val_rows: pd.DataFrame, ens: dict[str, np.ndarray]) -> dict:
    """Stage-B-style per-ban table for the two 5-seed ensemble means,
    conditioned on the Stage A dominant vector and ban phase."""
    df = val_rows.copy()
    for name, scores in ens.items():
        df[name] = scores
    models = list(ens)

    recs = []
    for (gid, s), g in df.groupby(["gameid", "seq"], sort=False):
        lab = g["label"].to_numpy()
        rec = {"gameid": gid, "seq": s, "is_ban": int(g["is_ban"].iloc[0])}
        for c in models:
            order = np.argsort(-g[c].to_numpy(), kind="stable")
            rec[c] = int(np.argmax(lab[order]))
        recs.append(rec)
    ranks = pd.DataFrame(recs)

    perban = pd.read_parquet(DATA_PROCESSED / "ban_attribution_perban.parquet")
    bans = ranks[ranks.is_ban == 1].merge(
        perban[["gameid", "seq", "ordinal", "dominant", "multi"]],
        on=["gameid", "seq"], how="left", validate="1:1",
    )
    n_unmatched = int(bans.dominant.isna().sum())
    bans = bans.dropna(subset=["dominant"])
    bans["phase"] = bans.ordinal.le(3).map({True: "B1-3", False: "B4-5"})

    def table(df: pd.DataFrame, by: str) -> dict:
        out = {}
        for k, g in df.groupby(by, observed=True):
            out[str(k)] = {"n": len(g)}
            for mname in models:
                out[str(k)][mname] = {
                    "top1": round(float((g[mname] < 1).mean()), 4),
                    "top5": round(float((g[mname] < 5).mean()), 4),
                }
        return out

    return {
        "n_val_bans": int(len(bans)),
        "n_unmatched_dropped": n_unmatched,
        "overall": table(bans.assign(all="all_bans"), "all"),
        "by_dominant_vector": table(bans, "dominant"),
        "by_phase_and_vector": {
            ph: table(g, "dominant") for ph, g in bans.groupby("phase")
        },
    }


def main() -> None:
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate",
                 "label"] + FEATURES,
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

    feat_all, feat_stats = build_feat_tensor(cand, games, vocab, val_start)
    feat_train = torch.from_numpy(feat_all[train_mask])
    feat_val = torch.from_numpy(feat_all[~train_mask])
    print(f"feat tensor {feat_all.shape} "
          f"({feat_all.nbytes / 1e6:.0f}MB), train stats {feat_stats}")

    cfg_base = dict(d_model=192, n_layers=4, n_heads=6)
    CACHE_DIR.mkdir(exist_ok=True)

    runs: list[dict] = []
    ens_probs: dict[str, list[np.ndarray]] = {"baseline": [], "feat": []}
    for seed in SEEDS:
        for condition in ("baseline", "feat"):
            cache = CACHE_DIR / f"{condition}_seed{seed}.json"
            probs_cache = CACHE_DIR / f"{condition}_seed{seed}_probs.npz"
            if cache.exists() and probs_cache.exists():
                run = json.loads(cache.read_text())
                ens_probs[condition].append(np.load(probs_cache)["probs"])
                print(f"[cached] {condition} seed={seed} "
                      f"val_loss={run['best_val_loss']:.4f} "
                      f"ban_top1={run['bans']['top1']:.4f}")
                runs.append(run)
                continue
            cfg = Config(**cfg_base, seed=seed)
            if condition == "feat":
                tr_t = {**t_train, "feat": feat_train}
                va_t = {**t_val, "feat": feat_val}
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
            if condition == "feat":
                run["feat_out"] = [
                    [round(float(v), 4) for v in row]
                    for row in model.feat_out.detach().tolist()
                ]
                d = cfg.d_model
                run["feat_proj_norm_per_channel"] = [
                    round(float(model.feat_proj[k * d:(k + 1) * d].norm()), 4)
                    for k in range(len(FEATURES))
                ]
                torch.save(model.state_dict(),
                           CACHE_DIR / f"feat_seed{seed}.pt")
            ens_probs[condition].append(probs)
            np.savez_compressed(probs_cache, probs=probs)
            cache.write_text(json.dumps(run))
            print(f"{condition} seed={seed} val_loss={best_val:.4f} "
                  f"ban_top1={acc['bans']['top1']:.4f} "
                  f"pick_top1={acc['picks']['top1']:.4f} "
                  f"({epochs} epochs, {wall:.0f}s)", flush=True)
            runs.append(run)
            if condition == "baseline":
                if seed == 16:
                    assert round(best_val, 4) == BASELINE_CANARY, (
                        f"harness drift: seed-16 baseline val loss "
                        f"{best_val:.4f} != canary {BASELINE_CANARY}"
                    )
                attr = ATTR_CACHE / f"baseline_seed{seed}.npz"
                if attr.exists():
                    ap = np.load(attr)["probs"]
                    assert np.allclose(ap, probs, atol=1e-5), (
                        f"baseline seed={seed} val probs differ from "
                        "expcache_attr — lineage drifted, stop and review"
                    )

    by = {(r["condition"], r["seed"]): r for r in runs}
    d_loss = np.array([by[("baseline", s)]["best_val_loss"]
                       - by[("feat", s)]["best_val_loss"] for s in SEEDS])
    d_ban = np.array([by[("feat", s)]["bans"]["top1"]
                      - by[("baseline", s)]["bans"]["top1"] for s in SEEDS])
    d_pick = np.array([by[("feat", s)]["picks"]["top1"]
                       - by[("baseline", s)]["picks"]["top1"] for s in SEEDS])
    rng = np.random.default_rng(0)
    boot = rng.choice(d_loss, size=(N_BOOTSTRAP, len(d_loss))).mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    n_pos = int((d_loss > 0).sum())
    go_ban = float(d_ban.mean()) >= NOISE_BAND
    go_loss = ci[0] > 0
    pick_ok = float(d_pick.mean()) >= -NOISE_BAND
    verdict = "GO" if (go_ban or go_loss) and pick_ok else "NO-GO"

    # Right-reason check: 5-seed ensemble means, Stage-B conditional cells.
    ens_scores = {}
    for condition in ("baseline", "feat"):
        mean_probs = np.mean(ens_probs[condition], axis=0)
        ens_scores[f"{condition}_ens"] = attach_scores(
            val_rows, mean_probs, val_pos, vocab
        )
    cond_table = conditional_table(val_rows, ens_scores)

    out = {
        "note": ("Team/player conditioning rung: opp_usage/player_pool/"
                 "player_wr injected input-side (feature-weighted champion-"
                 "embedding context, zero-init 3d x d projection) plus a "
                 "slot-type-gated linear output bias (6 scalars), baseline "
                 "vs feat, 5 paired seeds, pro validation only. Primary "
                 "metric: ban val top-1; right-reason check: gain should "
                 "concentrate in POCKET and B1-3 cells."),
        "runs": runs,
        "baseline_ban_top1_mean": round(
            float(np.mean([by[("baseline", s)]["bans"]["top1"] for s in SEEDS])), 4
        ),
        "feat_ban_top1_mean": round(
            float(np.mean([by[("feat", s)]["bans"]["top1"] for s in SEEDS])), 4
        ),
        "paired_deltas": {
            "seeds": SEEDS,
            "d_ban_top1 (feat - baseline, >0 favors feat)": d_ban.round(4).tolist(),
            "d_pick_top1 (feat - baseline)": d_pick.round(4).tolist(),
            "d_loss (baseline - feat, >0 favors feat)": d_loss.round(4).tolist(),
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
        "conditional_table_ensemble": cond_table,
        "provenance": {
            "features": FEATURES,
            "feature_semantics": "per (gameid, seq, candidate); ban rows "
                                 "reference the opponent's roster, pick rows "
                                 "the own team's (draft_dataset v0.6)",
            "standardization": {"population": "train candidate rows only",
                                **feat_stats},
            "injection": "input-side champ-emb-weighted context @ zero-init "
                         "(3d x d) + output bias feat_out[slot_type, F], "
                         "zero-init, no RNG consumed",
            "split_cutoff": str(cutoff.date()), "val_days": VAL_DAYS,
            "config": Config(**cfg_base).tag(), "device": "cpu",
            "seeds": SEEDS, "bootstrap_resamples": N_BOOTSTRAP,
            "baseline_canary_seed16": BASELINE_CANARY,
        },
    }
    (DATA_PROCESSED / "pocket_conditioning_rung.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"\nbaseline ban top-1 mean: {out['baseline_ban_top1_mean']:.4f}  "
          f"feat: {out['feat_ban_top1_mean']:.4f}")
    print(f"d_ban_top1 per seed: {d_ban.round(4).tolist()}")
    print(f"d_pick_top1 per seed: {d_pick.round(4).tolist()}")
    print(f"d_loss per seed: {d_loss.round(4).tolist()}  ({n_pos}/5 positive)")
    print(f"mean d_loss={d_loss.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"VERDICT: {verdict}")
    print("\nconditional cells (ensemble, top1/top5):")
    ct = cond_table
    for section in ("overall", "by_dominant_vector"):
        for k, v in ct[section].items():
            row = "  ".join(
                f"{m}:{v[m]['top1']*100:.1f}/{v[m]['top5']*100:.1f}"
                for m in ("baseline_ens", "feat_ens")
            )
            print(f"  {k:>10} n={v['n']:<4} {row}")
    for ph, tab in ct["by_phase_and_vector"].items():
        for k, v in tab.items():
            row = "  ".join(
                f"{m}:{v[m]['top1']*100:.1f}/{v[m]['top5']*100:.1f}"
                for m in ("baseline_ens", "feat_ens")
            )
            print(f"  {ph} {k:>8} n={v['n']:<4} {row}")


if __name__ == "__main__":
    main()
