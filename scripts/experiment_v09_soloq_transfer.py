"""Step 3 of the soloq embedding rung: does soloq-transfer init beat random?

Initializes the pro draft transformer's champ_emb rows from the rung-0 soloq
embedding (projected 128->192 with a fixed orthonormal map, rescaled to the
fresh-init row norm 0.02*sqrt(192)) and compares against random init on pro
validation over 5 paired seeds. Spec: docs/2026-07-22-soloq-data-research.md
paragraph 6; execution notes: docs/2026-07-23-step3-execution-handoff.md.

Same data, split, vocab, and config as embedding_evolution_v08.py (production
d192x4L6H, cutoff 2026-07-15, 14-day val window, frozen EWC test set dropped
and never touched). Conditions share torch's RNG stream exactly — injection
uses only numpy — so per seed the ONLY difference is the 168 champion rows.

train_model and topk_accuracy are inlined rather than imported: train_model
needs the champ_init hook plus epoch/wall-time bookkeeping, and importing
train_draft_model drags in lightgbm, whose libomp isn't installed here
(precedent: embedding_evolution_v08.py inlines VAL_DAYS).

Output: data/processed/soloq_transfer_step3.json
        data/processed/expcache_step3/<condition>_seed<S>.json (per-run cache)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from common import DATA_PROCESSED
from draft_transformer import (
    Config, DraftTransformer, Vocab, attach_scores, build_games, masked_loss,
    probs_for, to_tensors,
)

VAL_DAYS = 14
SEEDS = [16, 17, 42, 7, 23]  # production set, train_draft_model.SEEDS
PROJECTION_SEED = 1234
N_BOOTSTRAP = 10_000
CACHE_DIR = DATA_PROCESSED / "expcache_step3"

# 21 pro names differ from soloq internal names; 168/168 pro vocab champions
# covered (verified 2026-07-23). Coverage is asserted at runtime regardless.
PRO_TO_SOLOQ = {
    "Aurelion Sol": "AurelionSol", "Bel'Veth": "Belveth", "Cho'Gath": "Chogath",
    "Dr. Mundo": "DrMundo", "Fiddlesticks": "FiddleSticks",
    "Jarvan IV": "JarvanIV", "K'Sante": "KSante", "Kai'Sa": "Kaisa",
    "Kha'Zix": "Khazix", "Kog'Maw": "KogMaw", "LeBlanc": "Leblanc",
    "Lee Sin": "LeeSin", "Miss Fortune": "MissFortune",
    "Nunu & Willump": "Nunu", "Rek'Sai": "RekSai", "Renata Glasc": "Renata",
    "Tahm Kench": "TahmKench", "Twisted Fate": "TwistedFate",
    "Vel'Koz": "Velkoz", "Wukong": "MonkeyKing", "Xin Zhao": "XinZhao",
}


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
    champ_init: dict[int, np.ndarray] | None = None,
) -> tuple[DraftTransformer, float, int, float]:
    """draft_transformer.train_model plus: champ_init injection (numpy only —
    must not touch torch's RNG so both conditions see identical dropout and
    shuffle streams), and (best_val, epochs, wall_s) in the return."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model = DraftTransformer(cfg, vocab_size, n_leagues)
    if champ_init is not None:
        with torch.no_grad():
            for cid, v in champ_init.items():
                model.champ_emb.weight[cid] = torch.from_numpy(v).float()
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


def build_champ_init(vocab: Vocab) -> tuple[dict[int, np.ndarray], float]:
    soloq = np.load(DATA_PROCESSED / "soloq_champ_embeddings.npz",
                    allow_pickle=True)
    soloq_row = {c: i for i, c in enumerate(soloq["champs"])}
    rng = np.random.RandomState(PROJECTION_SEED)
    Q, _ = np.linalg.qr(rng.randn(128, 192).T)  # 192x128 orthonormal columns
    P = Q.T                                     # 128->192, norm-preserving
    target_norm = 0.02 * np.sqrt(192)
    init: dict[int, np.ndarray] = {}
    for pro_name, cid in vocab.id_of.items():
        row = soloq_row.get(PRO_TO_SOLOQ.get(pro_name, pro_name))
        if row is None:
            continue
        v = soloq["emb"][row] @ P
        init[cid] = (v * (target_norm / np.linalg.norm(v))).astype(np.float32)
    assert len(init) == len(vocab.id_of), (
        f"name bridge incomplete: {len(init)}/{len(vocab.id_of)} matched"
    )
    return init, float(target_norm)


def main() -> None:
    seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences_multi.parquet")
    seq["date"] = pd.to_datetime(seq["date"])
    cand = pd.read_parquet(
        DATA_PROCESSED / "draft_decisions_multi.parquet",
        columns=["gameid", "date", "league", "seq", "is_ban", "candidate", "label"],
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
    g_train = games[games.date < val_start].reset_index(drop=True)
    g_val = games[games.date >= val_start].reset_index(drop=True)
    t_train, t_val = to_tensors(g_train), to_tensors(g_val)
    val_pos = {g: i for i, g in enumerate(g_val.gameid)}
    val_rows = cand[cand.date >= val_start].copy()
    n_val_dec = val_rows.groupby(["gameid", "seq"]).ngroups
    n_train_dec = int((t_train["target"] != -100).sum())
    print(f"cutoff {cutoff.date()}: train {len(g_train)}g/{n_train_dec}d, "
          f"val {len(g_val)}g/{n_val_dec}d, vocab {vocab.size}")
    assert (len(g_train), n_train_dec) == (5043, 100836)
    assert (len(g_val), n_val_dec) == (54, 1080)

    champ_init, target_norm = build_champ_init(vocab)
    cfg_base = dict(d_model=192, n_layers=4, n_heads=6)
    CACHE_DIR.mkdir(exist_ok=True)

    runs: list[dict] = []
    for seed in SEEDS:
        for condition in ("random", "soloq"):
            cache = CACHE_DIR / f"{condition}_seed{seed}.json"
            if cache.exists():
                run = json.loads(cache.read_text())
                print(f"[cached] {condition} seed={seed} "
                      f"val_loss={run['best_val_loss']:.4f} top1={run['top1']:.4f}")
                runs.append(run)
                continue
            cfg = Config(**cfg_base, seed=seed)
            model, best_val, epochs, wall = train_model(
                cfg, t_train, t_val, vocab.size, n_leagues,
                champ_init=champ_init if condition == "soloq" else None,
            )
            probs = probs_for(model, t_val).numpy()
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
            cache.write_text(json.dumps(run))
            print(f"{condition} seed={seed} val_loss={best_val:.4f} "
                  f"top1={acc['all']['top1']:.4f} "
                  f"({epochs} epochs, {wall:.0f}s)", flush=True)
            runs.append(run)

    by = {(r["condition"], r["seed"]): r for r in runs}
    d_loss = np.array([by[("random", s)]["best_val_loss"]
                       - by[("soloq", s)]["best_val_loss"] for s in SEEDS])
    d_top1 = np.array([by[("soloq", s)]["top1"]
                       - by[("random", s)]["top1"] for s in SEEDS])
    rng = np.random.default_rng(0)
    boot = rng.choice(d_loss, size=(N_BOOTSTRAP, len(d_loss))).mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    n_pos = int((d_loss > 0).sum())
    go_loss = ci[0] > 0
    go_top1 = float(d_top1.mean()) >= 0.015
    verdict = "GO" if (go_loss or go_top1) else "NO-GO"

    out = {
        "note": ("Step 3 soloq->pro embedding transfer: soloq-init champ_emb "
                 "vs random init, 5 paired seeds, pro validation only"),
        "runs": runs,
        "paired_deltas": {
            "seeds": SEEDS,
            "d_loss (random - soloq, >0 favors soloq)": d_loss.round(4).tolist(),
            "d_top1 (soloq - random, >0 favors soloq)": d_top1.round(4).tolist(),
            "mean_d_loss": round(float(d_loss.mean()), 4),
            "mean_d_top1": round(float(d_top1.mean()), 4),
            "d_loss_positive_seeds": f"{n_pos}/{len(SEEDS)}",
            "bootstrap_95ci_mean_d_loss": [round(c, 4) for c in ci],
        },
        "verdict": verdict,
        "verdict_detail": {"ci_excludes_0_favorably": go_loss,
                           "mean_d_top1_ge_1.5pts": go_top1},
        "provenance": {
            "projection_seed": PROJECTION_SEED, "target_norm": target_norm,
            "mapping_size": len(PRO_TO_SOLOQ), "champs_injected": len(champ_init),
            "split_cutoff": str(cutoff.date()), "val_days": VAL_DAYS,
            "config": Config(**cfg_base).tag(), "device": "cpu",
            "soloq_purity": 0.7225, "bootstrap_resamples": N_BOOTSTRAP,
        },
    }
    (DATA_PROCESSED / "soloq_transfer_step3.json").write_text(json.dumps(out, indent=2))
    print(f"\nd_loss per seed: {d_loss.round(4).tolist()}  ({n_pos}/5 positive)")
    print(f"d_top1 per seed: {d_top1.round(4).tolist()}")
    print(f"mean d_loss={d_loss.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"mean d_top1={d_top1.mean():.4f}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
