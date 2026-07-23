"""Retrain one v0.8 transformer seed with per-epoch embedding snapshots.

Purpose: a *pedagogical* artifact — "watch the role clusters form" — showing
how the champion embedding table evolves from random init to structure over
training. This mirrors the production seed-16 fit as closely as the local
machine allows:

  - multi-year 2024-2026 data when draft_sequences_multi.parquet is present
    (falls back to the 2026-only file), same EWC cutoff and 14-day val split
    as experiment_v08.split_dates
  - production config (d192x4L6H, lr=3e-4, do=0.1, patience 8), seed 16
  - single seed, not the 5-seed ensemble; runs on MPS when available (the
    production run was CPU, so tiny numeric drift vs the shipped weights)
  - the frozen EWC July 2026 test games are excluded and never touched

Every snapshot stores the raw candidate-champion embedding rows; a fixed 2D
projection for display is computed downstream (the artifact uses per-frame
t-SNE with neighbor warm starts). Per-frame 5-NN role purity is computed in
the raw 192-D space, same as chart_embeddings_v08.py.

Output: data/processed/embedding_evolution_v08_demo.json
        data/processed/embedding_evolution_v08_snapshots.npz
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from chart_embeddings_v08 import role_shares
from common import DATA_PROCESSED
from draft_transformer import (
    Config, DraftTransformer, Vocab, build_games, masked_loss, to_tensors,
)
# train_draft_model.VAL_DAYS, inlined: importing that module drags in
# lightgbm, whose libomp isn't installed on this machine.
VAL_DAYS = 14

SUBEPOCH_EVERY = 8   # extra snapshots every N batches during the first epochs
SUBEPOCH_EPOCHS = 2  # how many early epochs get sub-epoch snapshots
DEVICE_KEYS = ("prev", "target", "avail", "league", "fearless", "gis")


def main() -> None:
    multi = DATA_PROCESSED / "draft_sequences_multi.parquet"
    if multi.exists():
        seq = pd.read_parquet(multi)
        cand = pd.read_parquet(DATA_PROCESSED / "draft_decisions_multi.parquet",
                               columns=["candidate"])
        data_note = "multi-year 2024-2026"
    else:
        seq = pd.read_parquet(DATA_PROCESSED / "draft_sequences.parquet")
        cand = pd.read_parquet(DATA_PROCESSED / "draft_decisions.parquet",
                               columns=["candidate"])
        data_note = "2026-only"
    seq["date"] = pd.to_datetime(seq["date"])

    # Same split rule as experiment_v08.split_dates; test games dropped
    # entirely — the frozen EWC set is never touched here.
    is_test = ((seq.league == "EWC") & (seq.date.dt.month == 7)
               & (seq.date.dt.year == 2026))
    cutoff = seq.loc[is_test, "date"].min()
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS)
    seq = seq[~is_test & (seq.date < cutoff)]

    vocab = Vocab(list(cand.candidate.unique()), list(seq.champion.unique()))
    games = build_games(seq, vocab)
    g_train = games[games.date < val_start].reset_index(drop=True)
    g_val = games[games.date >= val_start].reset_index(drop=True)
    n_leagues = len(games.attrs["leagues"])

    # CPU on purpose: masked_fill(-inf) + cross_entropy yields inf losses on
    # MPS (known backend issue), and the production run was CPU anyway.
    dev = torch.device("cpu")
    def to_dev(t: dict) -> dict:
        return {k: (v.to(dev) if k in DEVICE_KEYS else v) for k, v in t.items()}
    t_train, t_val = to_dev(to_tensors(g_train)), to_dev(to_tensors(g_val))
    print(f"{data_note}: train {len(g_train)} games / val {len(g_val)} games, "
          f"cutoff {cutoff.date()}, vocab {vocab.size}, device {dev}")

    cfg = Config(d_model=192, n_layers=4, n_heads=6, seed=16)
    torch.manual_seed(cfg.seed)
    model = DraftTransformer(cfg, vocab.size, n_leagues).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    rng = np.random.RandomState(cfg.seed)
    n = t_train["prev"].shape[0]

    cand_ids = torch.from_numpy(vocab.candidate_ids)
    frames: list[dict] = []
    embs: list[np.ndarray] = []

    def snapshot(label: str, train_loss: float | None, val_loss: float) -> None:
        with torch.no_grad():
            embs.append(model.champ_emb.weight.cpu()[cand_ids].numpy().copy())
        frames.append({"label": label, "train_loss": train_loss,
                       "val_loss": val_loss})

    def val_loss_now() -> float:
        model.eval()
        with torch.no_grad():
            vl = float(masked_loss(model(t_val), t_val))
        model.train()
        return vl

    snapshot("init (random)", None, val_loss_now())
    best_val, bad = float("inf"), 0
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(n)
        tot = cnt = 0.0
        batches = list(range(0, n, cfg.batch_size))
        for bi, i in enumerate(batches):
            ix = torch.from_numpy(order[i:i + cfg.batch_size])
            batch = {k: (v[ix.to(v.device)] if k in DEVICE_KEYS else v[ix])
                     for k, v in t_train.items()}
            loss = masked_loss(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(ix)
            cnt += len(ix)
            if (epoch < SUBEPOCH_EPOCHS and bi < len(batches) - 1
                    and (bi + 1) % SUBEPOCH_EVERY == 0):
                snapshot(f"epoch {epoch} · batch {bi + 1}/{len(batches)}",
                         tot / cnt, val_loss_now())
        vl = val_loss_now()
        snapshot(f"epoch {epoch}", tot / cnt, vl)
        print(f"  epoch {epoch:3d} train_loss={tot / cnt:.4f} val_loss={vl:.4f} "
              f"({time.time() - t0:.0f}s)")
        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    # --- roles, purity ---
    shares = role_shares()
    champs = np.array(vocab.champs)
    known = np.array([c in shares.index for c in champs])
    primary = np.array([shares.loc[c].idxmax() if k else "?"
                        for c, k in zip(champs, known)])
    flex = np.array([bool(shares.loc[c].max() < 0.7) if k else False
                     for c, k in zip(champs, known)])

    def purity(e: np.ndarray) -> float:
        pts, lab = e[known], primary[known]
        d = ((pts[:, None] - pts[None]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        nn5 = np.argsort(d, axis=1)[:, :5]
        return float((lab[nn5] == lab[:, None]).mean())

    for fr, e in zip(frames, embs):
        fr["purity"] = round(purity(e), 4)

    out = {
        "note": (f"pedagogical demo retrain: single seed 16, {data_note} data, "
                 "production config d192x4L6H; frozen EWC test set excluded"),
        "config": cfg.tag(), "device": str(dev),
        "n_train_games": len(g_train), "n_val_games": len(g_val),
        "champions": champs.tolist(),
        "role": primary.tolist(), "flex": flex.tolist(),
        "frames": frames,
    }
    (DATA_PROCESSED / "embedding_evolution_v08_demo.json").write_text(
        json.dumps(out))
    np.savez_compressed(
        DATA_PROCESSED / "embedding_evolution_v08_snapshots.npz",
        embs=np.stack(embs).astype(np.float32), champs=champs,
        role=primary, flex=flex,
        labels=np.array([f["label"] for f in frames]),
        val_loss=np.array([f["val_loss"] for f in frames], dtype=np.float32),
        purity=np.array([f["purity"] for f in frames], dtype=np.float32),
    )
    print(f"done: {len(frames)} frames, final purity {frames[-1]['purity']:.3f}, "
          f"best val {best_val:.4f}")


if __name__ == "__main__":
    main()
