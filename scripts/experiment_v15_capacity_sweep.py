"""Val-only transformer capacity sweep (d_model x n_layers x dropout).

Mission 3 of the 2026-07-28 session. Every config in this lineage has run at
d192x4L6H, dropout 0.1 since v0.7 — that shape was never swept, so any rung's
gain could be partly a capacity effect. This sweeps it, val only.

Pre-registration (written before the sweep ran):

  Stage 1 (sweep): grid below, 3 seeds (16/17/42) per config. Selection metric
    is mean best val loss — top-1 on 1,080 decisions across 3 seeds has a
    standard error near a point, loss does not, and this stage only has to
    nominate a candidate. Ties (within 0.005) break toward the smaller model.
    n_heads is tied to d_model/32 so head width is held constant.
  Stage 2 (confirm): the single winning config vs the incumbent d192x4L6H on
    the standard 5 paired seeds (16/17/42/7/23), gated on the deployment
    metric: mean d_top1 (all decisions) >= +1.0pt with neither bans nor picks
    down more than 1.5pts. Loss reported, not gating.

Hyperparameter selection on val is allowed by the rails; the frozen EWC test
set is not touched here or afterwards.

--arm selects the injection the sweep runs on top of: "none" (the v0.8-lineage
baseline) or a slot-gated arm from experiment_v14. If v14's ban_only arm GOes,
the sweep belongs on top of it, not on the old baseline.

The incumbent config at --arm none, seed 16, hard-asserts the 3.5855 canary —
it is the harness check that the sweep is measuring capacity and not drift.

Output: data/processed/capacity_sweep.json
        data/processed/expcache_capacity/<arm>_<tag>_seed<S>.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_transformer import (
    Config, Vocab, attach_scores, build_games, masked_loss, probs_for,
    to_tensors,
)
from experiment_v13_pocket_conditioning import (
    BASELINE_CANARY, FEATURES, N_BOOTSTRAP, VAL_DAYS, build_feat_tensor,
    topk_accuracy,
)
from experiment_v14_slot_gated_pocket import SlotGatedPocketTransformer

SWEEP_SEEDS = [16, 17, 42]
CONFIRM_SEEDS = [16, 17, 42, 7, 23]
INCUMBENT = dict(d_model=192, n_layers=4, dropout=0.1)
GRID = [
    dict(d_model=d, n_layers=nl, dropout=do)
    for d in (128, 192, 256)
    for nl in (3, 4, 6)
    for do in (0.1, 0.2)
]
TIE_BAND = 0.005
CACHE_DIR = DATA_PROCESSED / "expcache_capacity"


def cfg_of(spec: dict, seed: int) -> Config:
    """n_heads follows d_model so head width stays 32 across the grid."""
    return Config(d_model=spec["d_model"], n_layers=spec["n_layers"],
                  n_heads=spec["d_model"] // 32, dropout=spec["dropout"],
                  seed=seed)


def tag_of(spec: dict) -> str:
    return f"d{spec['d_model']}x{spec['n_layers']}L_do{spec['dropout']:g}"


def train_one(cfg: Config, arm: str, train_t: dict, val_t: dict,
              vocab_size: int, n_leagues: int):
    """Same loop as v13/v14; the model class is v14's so --arm can carry an
    injection. arm="none" passes no feat tensor, which makes the forward
    bit-identical to the production DraftTransformer."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model = SlotGatedPocketTransformer(
        cfg, vocab_size, n_leagues, "ban_only" if arm == "none" else arm)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    n = train_t["prev"].shape[0]
    rng = np.random.RandomState(cfg.seed)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(n)
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


def load_split():
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate",
                 "label"] + FEATURES,
    )
    cand["date"] = pd.to_datetime(cand["date"])

    def drop_test(df):
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
    train_mask = (games.date < val_start).to_numpy()
    g_train = games[train_mask].reset_index(drop=True)
    g_val = games[~train_mask].reset_index(drop=True)
    t_train, t_val = to_tensors(g_train), to_tensors(g_val)
    val_rows = cand[cand.date >= val_start].copy()
    n_train_dec = int((t_train["target"] != -100).sum())
    assert (len(g_train), n_train_dec) == (5043, 100836)
    assert (len(g_val), val_rows.groupby(["gameid", "seq"]).ngroups) == (54, 1080)
    feat_all, _ = build_feat_tensor(cand, games, vocab, val_start)
    return {
        "t_train": t_train, "t_val": t_val, "vocab": vocab,
        "n_leagues": len(games.attrs["leagues"]),
        "val_pos": {g: i for i, g in enumerate(g_val.gameid)},
        "val_rows": val_rows, "cutoff": cutoff,
        "feat_train": torch.from_numpy(feat_all[train_mask]),
        "feat_val": torch.from_numpy(feat_all[~train_mask]),
    }


def run(spec: dict, seed: int, arm: str, d: dict) -> dict:
    """One (config, seed) run, cached on disk so the sweep is resumable."""
    tag = tag_of(spec)
    cache = CACHE_DIR / f"{arm}_{tag}_seed{seed}.json"
    if cache.exists():
        rec = json.loads(cache.read_text())
        print(f"[cached] {arm} {tag} seed={seed} "
              f"loss={rec['best_val_loss']:.4f} top1={rec['top1']:.4f}")
    else:
        cfg = cfg_of(spec, seed)
        if arm == "none":
            tr_t, va_t = d["t_train"], d["t_val"]
        else:
            tr_t = {**d["t_train"], "feat": d["feat_train"]}
            va_t = {**d["t_val"], "feat": d["feat_val"]}
        model, best_val, epochs, wall = train_one(
            cfg, arm, tr_t, va_t, d["vocab"].size, d["n_leagues"])
        probs = probs_for(model, va_t).numpy()
        d["val_rows"]["score"] = attach_scores(
            d["val_rows"], probs, d["val_pos"], d["vocab"])
        acc = topk_accuracy(d["val_rows"], "score")
        rec = {
            "arm": arm, "tag": tag, "seed": seed, **spec,
            "n_heads": cfg.n_heads,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "best_val_loss": round(best_val, 4),
            "top1": acc["all"]["top1"], "top5": acc["all"]["top5"],
            "bans": acc["bans"], "picks": acc["picks"],
            "epochs": epochs, "wall_s": round(wall, 1),
        }
        cache.write_text(json.dumps(rec))
        print(f"{arm} {tag} seed={seed} loss={best_val:.4f} "
              f"top1={acc['all']['top1']:.4f} ban={acc['bans']['top1']:.4f} "
              f"pick={acc['picks']['top1']:.4f} ({epochs}ep, {wall:.0f}s)",
              flush=True)
    # Canary: the incumbent shape must reproduce a known number exactly.
    # At --arm none that is the v0.8-lineage 3.5855; at an injection arm it is
    # v14's cached run of the same (config, arm, seed), which is the stronger
    # check because it also pins the feature tensor and the injection code.
    if {k: rec[k] for k in INCUMBENT} == INCUMBENT:
        if arm == "none" and seed == 16:
            assert rec["best_val_loss"] == BASELINE_CANARY, (
                f"harness drift: incumbent seed-16 val loss "
                f"{rec['best_val_loss']} != canary {BASELINE_CANARY}")
        elif arm != "none":
            v14 = DATA_PROCESSED / "expcache_slotgate" / f"{arm}_seed{seed}.json"
            if v14.exists():
                ref = json.loads(v14.read_text())["best_val_loss"]
                assert rec["best_val_loss"] == ref, (
                    f"harness drift: incumbent {arm} seed-{seed} val loss "
                    f"{rec['best_val_loss']} != v14's {ref}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="none",
                    choices=["none", "ban_only", "gated_proj"])
    ap.add_argument("--stage", default="both", choices=["sweep", "confirm", "both"])
    ap.add_argument("--confirm-tag", default=None,
                    help="Confirm this config instead of the sweep's winner. "
                         "POST-HOC: the pre-registered selection rule is mean "
                         "val loss, so anything named here was chosen after "
                         "seeing the sweep. Writes to capacity_sweep_posthoc"
                         ".json so the pre-registered result is not clobbered.")
    args = ap.parse_args()
    CACHE_DIR.mkdir(exist_ok=True)
    d = load_split()
    print(f"cutoff {d['cutoff'].date()}  arm={args.arm}  "
          f"{len(GRID)} configs x {len(SWEEP_SEEDS)} seeds")

    out = {
        "note": ("Val-only capacity sweep. Stage 1 selects on mean best val "
                 "loss over 3 seeds (ties within 0.005 -> smaller model); "
                 "stage 2 confirms the winner vs the d192x4L6H incumbent on 5 "
                 "paired seeds, gated on mean d_top1 >= +1.0pt with neither "
                 "bans nor picks down more than 1.5pts. No test look."),
        "arm": args.arm, "grid": GRID, "sweep_seeds": SWEEP_SEEDS,
        "confirm_seeds": CONFIRM_SEEDS, "incumbent": INCUMBENT,
    }

    sweep_recs = []
    if args.stage in ("sweep", "both"):
        for spec in GRID:
            for seed in SWEEP_SEEDS:
                sweep_recs.append(run(spec, seed, args.arm, d))
        sw = pd.DataFrame(sweep_recs)
        agg = (sw.groupby("tag")
                 .agg(n_params=("n_params", "first"),
                      mean_loss=("best_val_loss", "mean"),
                      mean_top1=("top1", "mean"),
                      mean_ban_top1=("bans", lambda s: np.mean([x["top1"] for x in s])),
                      mean_pick_top1=("picks", lambda s: np.mean([x["top1"] for x in s])),
                      mean_epochs=("epochs", "mean"),
                      mean_wall_s=("wall_s", "mean"))
                 .sort_values("mean_loss").round(4))
        best_loss = agg.mean_loss.min()
        tied = agg[agg.mean_loss <= best_loss + TIE_BAND]
        winner = tied.sort_values("n_params").index[0]  # tie -> smaller model
        out["sweep"] = {
            "runs": sweep_recs,
            "by_config": json.loads(agg.reset_index().to_json(orient="records")),
            "tied_within_band": list(tied.index),
            "winner": winner,
        }
        print("\nsweep (sorted by mean val loss):")
        print(agg.to_string())
        print(f"\nwinner: {winner} (tied within {TIE_BAND}: {list(tied.index)})")

    if args.stage in ("confirm", "both"):
        winner = args.confirm_tag or out.get("sweep", {}).get("winner")
        assert winner, "confirm stage needs a sweep winner or --confirm-tag"
        if args.confirm_tag:
            out["posthoc"] = (
                "Config named with --confirm-tag, i.e. chosen AFTER seeing the "
                "sweep. Not the pre-registered selection; descriptive only and "
                "not promotable on this run.")
        wspec = next(s for s in GRID if tag_of(s) == winner)
        if wspec == INCUMBENT:
            out["confirm"] = {"skipped": "winner is the incumbent config"}
            print("\nwinner is the incumbent — nothing to confirm.")
        else:
            recs = {}
            for name, spec in (("incumbent", INCUMBENT), ("winner", wspec)):
                for seed in CONFIRM_SEEDS:
                    recs[(name, seed)] = run(spec, seed, args.arm, d)
            def darr(key):
                return np.array([
                    recs[("winner", s)][key]["top1"] - recs[("incumbent", s)][key]["top1"]
                    if isinstance(recs[("winner", s)][key], dict)
                    else recs[("winner", s)][key] - recs[("incumbent", s)][key]
                    for s in CONFIRM_SEEDS])
            d_top1, d_ban, d_pick = darr("top1"), darr("bans"), darr("picks")
            d_loss = np.array([recs[("incumbent", s)]["best_val_loss"]
                               - recs[("winner", s)]["best_val_loss"]
                               for s in CONFIRM_SEEDS])
            rng = np.random.default_rng(0)
            boot = rng.choice(d_loss, size=(N_BOOTSTRAP, len(d_loss))).mean(axis=1)
            ci = [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]
            gate = {
                "primary_mean_d_top1_ge_1.0pt": float(d_top1.mean()) >= 0.010,
                "bans_not_down_gt_1.5pts": float(d_ban.mean()) >= -0.015,
                "picks_not_down_gt_1.5pts": float(d_pick.mean()) >= -0.015,
            }
            out["confirm"] = {
                "winner_tag": winner, "seeds": CONFIRM_SEEDS,
                "d_top1": d_top1.round(4).tolist(),
                "d_ban_top1": d_ban.round(4).tolist(),
                "d_pick_top1": d_pick.round(4).tolist(),
                "d_loss (incumbent - winner, >0 favors winner)": d_loss.round(4).tolist(),
                "mean_d_top1": round(float(d_top1.mean()), 4),
                "mean_d_ban_top1": round(float(d_ban.mean()), 4),
                "mean_d_pick_top1": round(float(d_pick.mean()), 4),
                "mean_d_loss": round(float(d_loss.mean()), 4),
                "bootstrap_95ci_mean_d_loss": ci,
                "gate": gate,
                "verdict": "GO" if all(gate.values()) else "NO-GO",
            }
            print(f"\nconfirm {winner} vs incumbent (5 paired seeds):")
            print(f"  mean d_top1 {d_top1.mean()*100:+.2f}pts  "
                  f"bans {d_ban.mean()*100:+.2f}  picks {d_pick.mean()*100:+.2f}")
            print(f"  mean d_loss {d_loss.mean():+.4f} 95% CI {ci}")
            print(f"  VERDICT: {out['confirm']['verdict']}")

    name = "capacity_sweep_posthoc" if args.confirm_tag else "capacity_sweep"
    (DATA_PROCESSED / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {DATA_PROCESSED / f'{name}.json'}")


if __name__ == "__main__":
    main()
