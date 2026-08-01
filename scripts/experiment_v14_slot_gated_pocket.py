"""Slot-gated pocket injection — follow-up to the v13 guard breach.

Spec: docs/2026-07-28-pocket-injection-slot-gating-spec.md (pre-registered
before this file was written). Prior result:
docs/2026-07-28-pocket-conditioning-results.md.

v13 injected opp_usage/player_pool/player_wr through ONE zero-init 3d x d
projection shared by all 20 slots. Bans gained +5.3 top-1 (5/5 seeds) and the
whole 0.162-nat loss gain was ban-side (+0.326 bans, -0.002 picks), but pick
top-1 fell 4.15 pts at every ordinal, with the pick argmax over-committing to
the roster favourite. Hypothesis H1: the damage is the DIRECT injection at
pick positions, which the pick objective is indifferent to but the pick argmax
is not. H0: it is shared encoder/embedding weights reshaped by ban gradients.

Two arms against the same cached baselines:

  ban_only    (primary)   feature info reaches ban slots only — input context
                          zeroed at pick positions, output bias restricted to
                          the ban slot type (3 scalars). Pick positions get no
                          feature term; they can still read ban-slot context
                          through causal attention.
  gated_proj  (secondary) two separate zero-init input projections, one per
                          slot type; output bias as v13. Distinguishes H1 from
                          "parameter sharing specifically".

Gate (pre-registered, evaluated on ban_only only): primary mean d_ban_top1 >=
+1.5pts AND guard mean d_pick_top1 within +/-1.5pts. Both required; no
loss-CI alternative path. gated_proj is descriptive and cannot promote.

All new params are created AFTER base construction, zero-init, consuming no
RNG, so the baseline condition reproduces production bit-for-bit.

VAL ONLY — frozen EWC test games dropped, never scored.

Output: data/processed/slot_gated_pocket_rung.json
        data/processed/expcache_slotgate/<arm>_seed<S>.json (+ probs, .pt)
        baselines read from data/processed/expcache_pocket/ (v13's cache)
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
from experiment_v13_pocket_conditioning import (
    ATTR_CACHE, BASELINE_CANARY, FEATURES, NOISE_BAND, N_BOOTSTRAP, SEEDS,
    VAL_DAYS, build_feat_tensor, conditional_table, topk_accuracy,
)

ARMS = ("ban_only", "gated_proj")
BASE_CACHE = DATA_PROCESSED / "expcache_pocket"   # v13's, reused as-is
CACHE_DIR = DATA_PROCESSED / "expcache_slotgate"
BAN_TYPE = 1  # SLOT_TYPE == 1 marks ban slots


class SlotGatedPocketTransformer(DraftTransformer):
    """DraftTransformer plus a slot-type-gated read of per-slot per-candidate
    pocket features. Without a "feat" batch entry the forward is bit-identical
    to the base model (same ops, same dropout draws), and every new parameter
    is zero-init and created after the base modules, consuming no RNG."""

    def __init__(self, cfg: Config, vocab_size: int, n_leagues: int, arm: str):
        super().__init__(cfg, vocab_size, n_leagues)
        assert arm in ARMS, arm
        self.arm = arm
        d, f = cfg.d_model, len(FEATURES)
        if arm == "ban_only":
            self.feat_proj = nn.Parameter(torch.zeros(f * d, d))
            self.feat_out = nn.Parameter(torch.zeros(f))
        else:  # gated_proj: one projection per slot type, output bias as v13
            self.feat_proj = nn.Parameter(torch.zeros(2, f * d, d))
            self.feat_out = nn.Parameter(torch.zeros(2, len(FEATURES)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prev = batch["prev"]
        slots = torch.arange(20, device=prev.device)
        stype = SLOT_TYPE.to(prev.device)
        x = (
            self.champ_emb(prev)
            + self.pos_emb(slots)[None]
            + self.type_emb(stype)[None]
            + self.side_emb(SLOT_SIDE.to(prev.device))[None]
            + (
                self.league_emb(batch["league"])
                + self.fearless_emb(batch["fearless"])
                + self.gis_emb(batch["gis"])
            )[:, None, :]
        )
        feat = batch.get("feat")  # (n, 20, vocab, F) or absent (baseline)
        if feat is not None:
            if self.arm == "ban_only":
                # zero the features at pick positions: no input context, no
                # output bias, nothing for the pick argmax to be dragged by
                feat = feat * (stype == BAN_TYPE).to(feat.dtype)[None, :, None, None]
            ctx = torch.einsum("gsvf,vd->gsfd", feat, self.champ_emb.weight)
            ctx = ctx.reshape(*ctx.shape[:2], -1)  # (n, 20, F*d)
            if self.arm == "ban_only":
                x = x + ctx @ self.feat_proj
            else:
                x = x + torch.einsum("gsk,skd->gsd", ctx, self.feat_proj[stype])
        h = self.norm(self.encoder(self.drop(x), mask=self.causal))
        logits = h @ self.champ_emb.weight.T + self.out_bias
        if feat is not None:
            if self.arm == "ban_only":
                logits = logits + torch.einsum("gsvf,f->gsv", feat, self.feat_out)
            else:
                w = self.feat_out[stype]  # (20, F)
                logits = logits + torch.einsum("gsvf,sf->gsv", feat, w)
        return logits


def train_model(
    cfg: Config,
    arm: str,
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    vocab_size: int,
    n_leagues: int,
) -> tuple[SlotGatedPocketTransformer, float, int, float]:
    """v13's train_model with the slot-gated model class."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model = SlotGatedPocketTransformer(cfg, vocab_size, n_leagues, arm)
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


def nll_split(truth_rows: pd.DataFrame, probs: np.ndarray) -> dict:
    """Val NLL decomposed into ban and pick halves — the v13 diagnostic that
    showed the objective only wants these features at ban slots."""
    p = probs[truth_rows.gi.to_numpy(), truth_rows.si.to_numpy(),
              truth_rows.ci.to_numpy()]
    nll = -np.log(np.clip(p, 1e-12, None))
    return {
        "all": round(float(nll.mean()), 4),
        "bans": round(float(nll[truth_rows.is_ban.to_numpy() == 1].mean()), 4),
        "picks": round(float(nll[truth_rows.is_ban.to_numpy() == 0].mean()), 4),
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

    # one row per val decision, for the ban/pick NLL decomposition
    truth = val_rows[val_rows.label == 1].copy()
    truth["gi"] = truth.gameid.map(val_pos).to_numpy()
    truth["si"] = truth.seq.map(SEQ_TO_SLOT).to_numpy()
    truth["ci"] = truth.candidate.map(vocab.id_of).to_numpy()
    assert len(truth) == 1080

    feat_all, feat_stats = build_feat_tensor(cand, games, vocab, val_start)
    feat_train = torch.from_numpy(feat_all[train_mask])
    feat_val = torch.from_numpy(feat_all[~train_mask])
    print(f"feat tensor {feat_all.shape} "
          f"({feat_all.nbytes / 1e6:.0f}MB), train stats {feat_stats}")

    cfg_base = dict(d_model=192, n_layers=4, n_heads=6)
    CACHE_DIR.mkdir(exist_ok=True)

    runs: list[dict] = []
    ens_probs: dict[str, list[np.ndarray]] = {c: [] for c in ("baseline", *ARMS)}
    nlls: dict[tuple[str, int], dict] = {}

    for seed in SEEDS:
        for condition in ("baseline", *ARMS):
            base = condition == "baseline"
            cdir = BASE_CACHE if base else CACHE_DIR
            cache = cdir / f"{condition}_seed{seed}.json"
            probs_cache = cdir / f"{condition}_seed{seed}_probs.npz"
            if cache.exists() and probs_cache.exists():
                run = json.loads(cache.read_text())
                probs = np.load(probs_cache)["probs"]
                print(f"[cached] {condition} seed={seed} "
                      f"val_loss={run['best_val_loss']:.4f} "
                      f"ban_top1={run['bans']['top1']:.4f} "
                      f"pick_top1={run['picks']['top1']:.4f}")
            else:
                cfg = Config(**cfg_base, seed=seed)
                if base:
                    tr_t, va_t = t_train, t_val
                else:
                    tr_t = {**t_train, "feat": feat_train}
                    va_t = {**t_val, "feat": feat_val}
                arm = condition if not base else ARMS[0]
                model, best_val, epochs, wall = train_model(
                    cfg, arm, tr_t, va_t, vocab.size, n_leagues,
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
                if not base:
                    run["feat_out"] = np.round(
                        model.feat_out.detach().numpy(), 4).tolist()
                    d = cfg.d_model
                    fp = model.feat_proj.detach()
                    if arm == "ban_only":
                        run["feat_proj_norm_per_channel"] = [
                            round(float(fp[k * d:(k + 1) * d].norm()), 4)
                            for k in range(len(FEATURES))
                        ]
                    else:
                        run["feat_proj_norm_per_channel"] = {
                            ("picks" if t == 0 else "bans"): [
                                round(float(fp[t, k * d:(k + 1) * d].norm()), 4)
                                for k in range(len(FEATURES))
                            ] for t in (0, 1)
                        }
                    torch.save(model.state_dict(),
                               CACHE_DIR / f"{condition}_seed{seed}.pt")
                np.savez_compressed(probs_cache, probs=probs)
                cache.write_text(json.dumps(run))
                print(f"{condition} seed={seed} val_loss={best_val:.4f} "
                      f"ban_top1={acc['bans']['top1']:.4f} "
                      f"pick_top1={acc['picks']['top1']:.4f} "
                      f"({epochs} epochs, {wall:.0f}s)", flush=True)

            # canary + lineage identity, whether trained or loaded from cache
            if base:
                if seed == 16:
                    assert run["best_val_loss"] == BASELINE_CANARY, (
                        f"harness drift: seed-16 baseline val loss "
                        f"{run['best_val_loss']} != canary {BASELINE_CANARY}"
                    )
                attr = ATTR_CACHE / f"baseline_seed{seed}.npz"
                if attr.exists():
                    assert np.allclose(np.load(attr)["probs"], probs, atol=1e-5), (
                        f"baseline seed={seed} val probs differ from "
                        "expcache_attr — lineage drifted, stop and review"
                    )
            ens_probs[condition].append(probs)
            nlls[(condition, seed)] = nll_split(truth, probs)
            runs.append(run)

    by = {(r["condition"], r["seed"]): r for r in runs}

    def deltas(arm: str) -> dict:
        d_loss = np.array([by[("baseline", s)]["best_val_loss"]
                           - by[(arm, s)]["best_val_loss"] for s in SEEDS])
        d_ban = np.array([by[(arm, s)]["bans"]["top1"]
                          - by[("baseline", s)]["bans"]["top1"] for s in SEEDS])
        d_pick = np.array([by[(arm, s)]["picks"]["top1"]
                           - by[("baseline", s)]["picks"]["top1"] for s in SEEDS])
        rng = np.random.default_rng(0)
        boot = rng.choice(d_loss, size=(N_BOOTSTRAP, len(d_loss))).mean(axis=1)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        d_nll_ban = np.array([nlls[("baseline", s)]["bans"]
                              - nlls[(arm, s)]["bans"] for s in SEEDS])
        d_nll_pick = np.array([nlls[("baseline", s)]["picks"]
                               - nlls[(arm, s)]["picks"] for s in SEEDS])
        primary = float(d_ban.mean()) >= NOISE_BAND
        guard = abs(float(d_pick.mean())) <= NOISE_BAND
        return {
            "seeds": SEEDS,
            "d_ban_top1 (arm - baseline, >0 favors arm)": d_ban.round(4).tolist(),
            "d_pick_top1 (arm - baseline)": d_pick.round(4).tolist(),
            "d_loss (baseline - arm, >0 favors arm)": d_loss.round(4).tolist(),
            "mean_d_ban_top1": round(float(d_ban.mean()), 4),
            "mean_d_pick_top1": round(float(d_pick.mean()), 4),
            "mean_d_loss": round(float(d_loss.mean()), 4),
            "d_loss_positive_seeds": f"{int((d_loss > 0).sum())}/{len(SEEDS)}",
            "bootstrap_95ci_mean_d_loss": [round(c, 4) for c in ci],
            "mean_d_nll_bans": round(float(d_nll_ban.mean()), 4),
            "mean_d_nll_picks": round(float(d_nll_pick.mean()), 4),
            "gate": {
                "primary_mean_d_ban_top1_ge_1.5pts": primary,
                "guard_mean_d_pick_top1_within_1.5pts": guard,
                "verdict": "GO" if (primary and guard) else "NO-GO",
            },
        }

    arms = {arm: deltas(arm) for arm in ARMS}

    # Right-reason check: 5-seed ensemble means, Stage-B conditional cells.
    ens_scores = {}
    for condition in ("baseline", *ARMS):
        mean_probs = np.mean(ens_probs[condition], axis=0)
        ens_scores[f"{condition}_ens"] = attach_scores(
            val_rows, mean_probs, val_pos, vocab
        )
    cond_table = conditional_table(val_rows, ens_scores)

    out = {
        "note": ("Slot-gated pocket injection, follow-up to the v13 guard "
                 "breach. ban_only (primary, gated) vs gated_proj (secondary, "
                 "per-slot-type projections) vs the cached v13 baselines, 5 "
                 "paired seeds, pro validation only. Pre-registered gate on "
                 "ban_only: mean d_ban_top1 >= +1.5pts AND mean d_pick_top1 "
                 "within +/-1.5pts; both required. gated_proj is descriptive "
                 "and cannot promote."),
        "spec": "docs/2026-07-28-pocket-injection-slot-gating-spec.md",
        "runs": runs,
        "mean_top1": {
            c: {
                k: round(float(np.mean([by[(c, s)][k]["top1"] for s in SEEDS])), 4)
                for k in ("bans", "picks")
            } for c in ("baseline", *ARMS)
        },
        "mean_nll": {
            c: {
                k: round(float(np.mean([nlls[(c, s)][k] for s in SEEDS])), 4)
                for k in ("all", "bans", "picks")
            } for c in ("baseline", *ARMS)
        },
        "arms": arms,
        "verdict": arms["ban_only"]["gate"]["verdict"],
        "conditional_table_ensemble": cond_table,
        "provenance": {
            "features": FEATURES,
            "injection": {
                "ban_only": "features zeroed at pick slots; shared 3d x d "
                            "zero-init input projection + 3-scalar ban-only "
                            "output bias",
                "gated_proj": "two zero-init 3d x d input projections indexed "
                              "by slot type + v13's 2x3 output bias",
            },
            "standardization": {"population": "train candidate rows only",
                                **feat_stats},
            "baselines": "reused from data/processed/expcache_pocket (v13)",
            "split_cutoff": str(cutoff.date()), "val_days": VAL_DAYS,
            "config": Config(**cfg_base).tag(), "device": "cpu",
            "seeds": SEEDS, "bootstrap_resamples": N_BOOTSTRAP,
            "baseline_canary_seed16": BASELINE_CANARY,
        },
    }
    (DATA_PROCESSED / "slot_gated_pocket_rung.json").write_text(
        json.dumps(out, indent=2)
    )

    print("\nmean top-1 (bans / picks):")
    for c in ("baseline", *ARMS):
        m = out["mean_top1"][c]
        print(f"  {c:<11} {m['bans']*100:5.2f} / {m['picks']*100:5.2f}")
    for arm in ARMS:
        a = arms[arm]
        print(f"\n{arm}:")
        print(f"  d_ban_top1  per seed {a['d_ban_top1 (arm - baseline, >0 favors arm)']}"
              f"  mean {a['mean_d_ban_top1']*100:+.2f}pts")
        print(f"  d_pick_top1 per seed {a['d_pick_top1 (arm - baseline)']}"
              f"  mean {a['mean_d_pick_top1']*100:+.2f}pts")
        print(f"  d_loss mean {a['mean_d_loss']:+.4f} "
              f"95% CI {a['bootstrap_95ci_mean_d_loss']} "
              f"({a['d_loss_positive_seeds']} positive)")
        print(f"  d_nll bans {a['mean_d_nll_bans']:+.4f}  "
              f"picks {a['mean_d_nll_picks']:+.4f}")
        print(f"  gate: {a['gate']}")
    print(f"\nVERDICT (ban_only): {out['verdict']}")
    print("\nconditional cells (ensemble, top1/top5):")
    models = [f"{c}_ens" for c in ("baseline", *ARMS)]
    ct = cond_table
    for section in ("overall", "by_dominant_vector"):
        for k, v in ct[section].items():
            row = "  ".join(
                f"{m.replace('_ens',''):}:{v[m]['top1']*100:.1f}/{v[m]['top5']*100:.1f}"
                for m in models
            )
            print(f"  {k:>10} n={v['n']:<4} {row}")
    for ph, tab in ct["by_phase_and_vector"].items():
        for k, v in tab.items():
            row = "  ".join(
                f"{m.replace('_ens',''):}:{v[m]['top1']*100:.1f}/{v[m]['top5']*100:.1f}"
                for m in models
            )
            print(f"  {ph} {k:>8} n={v['n']:<4} {row}")


if __name__ == "__main__":
    main()
