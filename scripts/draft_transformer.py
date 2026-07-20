"""Small causal transformer over the draft sequence with learned champion embeddings.

Each game is a 20-slot sequence (standard tournament draft order). At slot t the
model sees the champions chosen at slots < t plus the metadata of slot t (draft
position, ban/pick, side, league, fearless, game-in-series) and predicts the
champion chosen at slot t over the champions still available. Champion identity
enters only through a learned embedding table (weight-tied with the output
head), so attention over the draft-so-far has to discover synergy/counter/role
structure that the GBM feature sets hand-build.

Slots with no decision (missed bans) contribute a MISSED input token and no
loss. Slots whose champion was never picked in-sample have no candidate rows in
the pointwise dataset, so they are likewise excluded from loss and eval to keep
metrics comparable. Availability masks (champions not yet taken this game nor
used earlier in a fearless series) restrict both the training loss and eval
scores to the same candidate sets the GBM models rank.

This module only defines the machinery; selection lives in experiment_v08.py
(val only) and the final fit + blind test in train_draft_model_v08.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from draft_dataset import DRAFT_SEQUENCE

PAD, START, MISSED = 0, 1, 2
N_SPECIALS = 3
SLOT_TYPE = torch.tensor([int(d == "ban") for _, d, _, _ in DRAFT_SEQUENCE])
SLOT_SIDE = torch.tensor([int(s == "Blue") for _, _, s, _ in DRAFT_SEQUENCE])
SEQ_TO_SLOT = {seq: i for i, (seq, _, _, _) in enumerate(DRAFT_SEQUENCE)}


@dataclass
class Config:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 8
    time_decay_tau_days: float | None = None  # weight games by exp(-age/tau)
    seed: int = 16

    def tag(self) -> str:
        tau = f",tau={self.time_decay_tau_days:g}" if self.time_decay_tau_days else ""
        return (f"d{self.d_model}x{self.n_layers}L{self.n_heads}H,"
                f"lr={self.lr:g},do={self.dropout:g}{tau}")


class Vocab:
    """Champion <-> token id. Candidate champions are the ones the pointwise
    dataset ranks (picked at least once in-sample); extra champions that only
    ever appear as bans still get input tokens but are never predicted."""

    def __init__(self, candidate_champs: list[str], extra_champs: list[str]):
        self.champs = sorted(candidate_champs)
        extras = sorted(set(extra_champs) - set(self.champs))
        self.id_of = {c: N_SPECIALS + i for i, c in enumerate(self.champs + extras)}
        self.size = N_SPECIALS + len(self.champs) + len(extras)
        self.candidate_ids = np.array(
            [self.id_of[c] for c in self.champs], dtype=np.int64
        )


def build_games(seq_df: pd.DataFrame, vocab: Vocab) -> pd.DataFrame:
    """One row per game with fixed-length slot arrays.

    Columns: champ (20, id or 0 where no decision), target (20, id or -100),
    avail (20 x vocab.size bool), plus league/fearless/game_in_series/date.
    """
    leagues = sorted(seq_df.league.unique())
    lidx = {lg: i for i, lg in enumerate(leagues)}
    rows = []
    for gameid, g in seq_df.groupby("gameid", sort=False):
        champ = np.zeros(20, dtype=np.int64)
        target = np.full(20, -100, dtype=np.int64)
        avail = np.zeros((20, vocab.size), dtype=bool)
        prior = g.series_prior.iloc[0]
        banned_prior = {vocab.id_of[c] for c in prior.split("|") if c} if prior else set()
        open_ids = np.array(
            [i for i in vocab.candidate_ids if i not in banned_prior], dtype=np.int64
        )
        taken: list[int] = []
        for _, r in g.iterrows():
            slot = SEQ_TO_SLOT[r.seq]
            cid = vocab.id_of[r.champion]
            champ[slot] = cid
            if r.in_candidates:
                target[slot] = cid
                mask = np.ones(vocab.size, dtype=bool)
                mask[open_ids] = False
                mask[taken] = True  # no-op for non-candidates, cheap
                avail[slot] = ~mask
            taken.append(cid)
        rows.append({
            "gameid": gameid, "date": g.date.iloc[0],
            "league_id": lidx[g.league.iloc[0]],
            "fearless": int(g.fearless.iloc[0]),
            "game_in_series": min(int(g.game_in_series.iloc[0]), 7),
            "champ": champ, "target": target, "avail": avail,
        })
    out = pd.DataFrame(rows)
    out.attrs["leagues"] = leagues
    return out


def to_tensors(games: pd.DataFrame) -> dict[str, torch.Tensor]:
    inp = np.stack(games.champ.to_numpy())  # champion chosen at each slot
    prev = np.full_like(inp, START)
    prev[:, 1:] = np.where(inp[:, :-1] > 0, inp[:, :-1], MISSED)
    return {
        "prev": torch.from_numpy(prev),
        "target": torch.from_numpy(np.stack(games.target.to_numpy())),
        "avail": torch.from_numpy(np.stack(games.avail.to_numpy())),
        "league": torch.from_numpy(games.league_id.to_numpy(np.int64)),
        "fearless": torch.from_numpy(games.fearless.to_numpy(np.int64)),
        "gis": torch.from_numpy(games.game_in_series.to_numpy(np.int64) - 1),
        "date": torch.from_numpy(
            games.date.to_numpy(dtype="datetime64[D]").astype(np.int64)
        ),
    }


class DraftTransformer(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, n_leagues: int):
        super().__init__()
        d = cfg.d_model
        self.champ_emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos_emb = nn.Embedding(20, d)
        self.type_emb = nn.Embedding(2, d)
        self.side_emb = nn.Embedding(2, d)
        self.league_emb = nn.Embedding(n_leagues, d)
        self.fearless_emb = nn.Embedding(2, d)
        self.gis_emb = nn.Embedding(7, d)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=2 * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(d)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        causal = torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal)
        # Default embedding init is N(0,1); with a weight-tied head that puts
        # initial logits at +/-50 and the first epochs are spent recovering.
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        with torch.no_grad():
            self.champ_emb.weight[PAD].zero_()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prev = batch["prev"]
        n = prev.shape[0]
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
        h = self.norm(self.encoder(self.drop(x), mask=self.causal))
        # weight-tied output head: score every champion at every slot
        return h @ self.champ_emb.weight.T + self.out_bias


def masked_loss(
    logits: torch.Tensor, batch: dict[str, torch.Tensor],
    game_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = logits.masked_fill(~batch["avail"], float("-inf"))
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), batch["target"].reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape(batch["target"].shape)
    m = batch["target"] != -100
    if game_weight is None:
        return losses[m].mean()
    w = game_weight[:, None].expand_as(losses)[m]
    return (losses[m] * w).sum() / w.sum()


def train_model(
    cfg: Config,
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    vocab_size: int,
    n_leagues: int,
    verbose: bool = True,
) -> DraftTransformer:
    torch.manual_seed(cfg.seed)
    model = DraftTransformer(cfg, vocab_size, n_leagues)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_t["prev"].shape[0]
    weight = None
    if cfg.time_decay_tau_days:
        age = (train_t["date"].max() - train_t["date"]).float()
        weight = torch.exp(-age / cfg.time_decay_tau_days)
    rng = np.random.RandomState(cfg.seed)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(n)
        tot = cnt = 0.0
        for i in range(0, n, cfg.batch_size):
            ix = torch.from_numpy(order[i : i + cfg.batch_size])
            batch = {k: v[ix] for k, v in train_t.items()}
            loss = masked_loss(
                model(batch), batch, None if weight is None else weight[ix]
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(ix)
            cnt += len(ix)
        model.eval()
        with torch.no_grad():
            vl = float(masked_loss(model(val_t), val_t))
        if verbose:
            print(f"  epoch {epoch:3d} train_loss={tot / cnt:.4f} val_loss={vl:.4f}")
        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"  [{cfg.tag()} seed={cfg.seed}] stopped epoch {epoch}, best val_loss={best_val:.4f}")
    return model


@dataclass
class ScoreIndex:
    """Maps candidate-level dataset rows to (game, slot, champion) logit cells."""

    game_pos: dict[str, int] = field(default_factory=dict)


def probs_for(
    model: DraftTransformer, tensors: dict[str, torch.Tensor], batch_size: int = 256
) -> torch.Tensor:
    """(n_games, 20, vocab) softmax probabilities over available champions."""
    outs = []
    with torch.no_grad():
        n = tensors["prev"].shape[0]
        for i in range(0, n, batch_size):
            batch = {k: v[i : i + batch_size] for k, v in tensors.items()}
            logits = model(batch).masked_fill(~batch["avail"], float("-inf"))
            outs.append(torch.softmax(logits, dim=-1).nan_to_num(0.0))
    return torch.cat(outs)


def attach_scores(
    part: pd.DataFrame,
    prob_stack: np.ndarray,
    game_pos: dict[str, int],
    vocab: Vocab,
) -> np.ndarray:
    """Align mean ensemble probabilities to candidate-level rows."""
    gi = part.gameid.map(game_pos).to_numpy()
    si = part.seq.map(SEQ_TO_SLOT).to_numpy()
    ci = part.candidate.map(vocab.id_of).to_numpy()
    return prob_stack[gi, si, ci]
