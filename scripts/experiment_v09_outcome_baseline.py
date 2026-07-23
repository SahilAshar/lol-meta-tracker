"""v0.9 rung 0: is there any draft signal in win prediction at all?

The falsifiable gate for the outcome-head ("coach tool") idea. Today's draft
model predicts what pros WILL pick, not what wins; before building a
win-probability head onto the transformer, this experiment asks the cheap
question: does knowing the completed draft improve held-out win-prediction
log-loss over a side + team-strength baseline? If the full 10-pick draft
carries no measurable signal, partial-draft coaching value is off the table
and the outcome head dies here.

Discipline: chronological splits. Hyperparameters (Elo K, logistic C, which
draft representation) are chosen on the val slice only; the holdout slice is
scored once per promoted model at the end. The frozen EWC July 2026 main
event is excluded from every slice — it stays spent. Online features (Elo,
trailing champion winrate) are causal: each game's features use only games
strictly before it, so computing them across the full timeline is not
leakage.

Writes data/processed/outcome_baseline_v09.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from common import DATA_PROCESSED
from draft_dataset import load_games

TRAIN_END = pd.Timestamp("2026-03-15")
VAL_END = pd.Timestamp("2026-05-15")
ELO_KS = [10.0, 20.0, 30.0, 40.0]
CS = [0.003, 0.01, 0.03, 0.1, 0.3]
META_HALF_LIFE_DAYS = 30.0
META_PRIOR_GAMES = 20.0
N_BOOT = 10_000


def build_game_table() -> pd.DataFrame:
    players, teams = load_games([2024, 2025, 2026])
    blue = teams[teams.side == "Blue"]
    games = blue.groupby("gameid").agg(
        date=("date", "first"), league=("league", "first"),
        blue_team=("teamname", "first"), blue_win=("result", "first"),
    )
    red = teams[teams.side == "Red"].groupby("gameid").teamname.first()
    games["red_team"] = red
    for side in ("Blue", "Red"):
        picks = (players[players.side == side]
                 .groupby("gameid").champion.apply(list))
        games[f"{side.lower()}_picks"] = picks
    games = games.reset_index().sort_values("date").reset_index(drop=True)
    assert games.blue_picks.str.len().eq(5).all()
    assert games.red_picks.str.len().eq(5).all()
    return games


def elo_diffs(games: pd.DataFrame, k: float) -> np.ndarray:
    """Blue-minus-red Elo before each game, updated online after it."""
    ratings: dict[str, float] = {}
    out = np.zeros(len(games))
    for i, g in enumerate(games.itertuples()):
        rb = ratings.get(g.blue_team, 1500.0)
        rr = ratings.get(g.red_team, 1500.0)
        out[i] = rb - rr
        expect_b = 1.0 / (1.0 + 10 ** ((rr - rb) / 400.0))
        delta = k * (g.blue_win - expect_b)
        ratings[g.blue_team] = rb + delta
        ratings[g.red_team] = rr - delta
    return out


def meta_diffs(games: pd.DataFrame) -> np.ndarray:
    """Trailing champion-winrate edge: mean shrunk WR of blue picks minus red.

    Per-champion exponentially decayed (wins, games) with a 30-day half-life,
    shrunk toward 0.5 by a pseudo-count prior; updated after each game.
    """
    state: dict[str, tuple[pd.Timestamp, float, float]] = {}

    def rate(champ: str, now: pd.Timestamp) -> float:
        last, w, n = state.get(champ, (now, 0.0, 0.0))
        decay = 0.5 ** ((now - last).days / META_HALF_LIFE_DAYS)
        return (w * decay + 0.5 * META_PRIOR_GAMES) / (n * decay + META_PRIOR_GAMES)

    def bump(champ: str, now: pd.Timestamp, won: float) -> None:
        last, w, n = state.get(champ, (now, 0.0, 0.0))
        decay = 0.5 ** ((now - last).days / META_HALF_LIFE_DAYS)
        state[champ] = (now, w * decay + won, n * decay + 1.0)

    out = np.zeros(len(games))
    for i, g in enumerate(games.itertuples()):
        now = g.date
        out[i] = (np.mean([rate(c, now) for c in g.blue_picks])
                  - np.mean([rate(c, now) for c in g.red_picks]))
        for c in g.blue_picks:
            bump(c, now, float(g.blue_win))
        for c in g.red_picks:
            bump(c, now, 1.0 - float(g.blue_win))
    return out


def champ_matrix(games: pd.DataFrame, vocab: dict[str, int]) -> np.ndarray:
    """Signed champion indicators: +1 if blue picked, -1 if red picked."""
    x = np.zeros((len(games), len(vocab)), dtype=np.float32)
    for i, g in enumerate(games.itertuples()):
        for c in g.blue_picks:
            if c in vocab:
                x[i, vocab[c]] = 1.0
        for c in g.red_picks:
            if c in vocab:
                x[i, vocab[c]] = -1.0
    return x


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": round(float(log_loss(y, p)), 5),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "acc": round(float(((p > 0.5) == y).mean()), 4),
    }


def fit_logistic(x_tr, y_tr, c: float = 1e6) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=2000).fit(x_tr, y_tr)


def main() -> None:
    games = build_game_table()
    print(f"{len(games)} games {games.date.min().date()} -> {games.date.max().date()}")

    is_frozen = (games.league == "EWC") & (games.date >= "2026-07-01")
    tr = (games.date < TRAIN_END).to_numpy()
    va = ((games.date >= TRAIN_END) & (games.date < VAL_END)).to_numpy()
    ho = ((games.date >= VAL_END) & ~is_frozen).to_numpy()
    trva = tr | va
    y = games.blue_win.to_numpy(dtype=float)
    print(f"train {tr.sum()} / val {va.sum()} / holdout {ho.sum()} "
          f"(frozen EWC main event excluded: {is_frozen.sum()})")

    # --- tune Elo K on val (M1 log-loss), features recomputed per K ---
    val_by_k = {}
    for k in ELO_KS:
        d = elo_diffs(games, k)
        m = fit_logistic(d[tr, None], y[tr])
        val_by_k[k] = metrics(y[va], m.predict_proba(d[va, None])[:, 1])
        print(f"  elo K={k:4.0f}: val {val_by_k[k]}")
    best_k = min(val_by_k, key=lambda k: val_by_k[k]["log_loss"])
    elo = elo_diffs(games, best_k)
    print(f"chose K={best_k}")

    meta = meta_diffs(games)
    vocab = {c: i for i, c in enumerate(sorted(
        {c for picks in games.loc[trva, ["blue_picks", "red_picks"]].to_numpy().ravel()
         for c in picks}))}
    champs = champ_matrix(games, vocab)
    print(f"{len(vocab)} champions in train+val vocab")

    # --- candidate models, val-scored (fit on train only) ---
    designs = {
        "M1_elo": elo[:, None],
        "M2a_elo_champs": np.hstack([elo[:, None], champs]),
        "M2b_elo_meta": np.column_stack([elo, meta]),
        "M2c_elo_meta_champs": np.hstack([elo[:, None], meta[:, None], champs]),
    }
    val_results, best_c = {}, {}
    for name, x in designs.items():
        grid = CS if "champs" in name else [1e6]
        scored = {}
        for c in grid:
            m = fit_logistic(x[tr], y[tr], c)
            scored[c] = metrics(y[va], m.predict_proba(x[va])[:, 1])
        best_c[name] = min(scored, key=lambda c: scored[c]["log_loss"])
        val_results[name] = scored[best_c[name]]
        print(f"  {name} (C={best_c[name]}): val {val_results[name]}")
    val_results["M0_baserate"] = metrics(
        y[va], np.full(va.sum(), y[tr].mean()))
    print(f"  M0_baserate: val {val_results['M0_baserate']}")

    challengers = [n for n in designs if n != "M1_elo"]
    gate = min(challengers, key=lambda n: val_results[n]["log_loss"])
    print(f"val-chosen challenger for the gate: {gate}")

    # --- one holdout look: refit on train+val with chosen hypers ---
    holdout, probs = {}, {}
    holdout["M0_baserate"] = metrics(y[ho], np.full(ho.sum(), y[trva].mean()))
    for name, x in designs.items():
        m = fit_logistic(x[trva], y[trva], best_c[name])
        probs[name] = m.predict_proba(x[ho])[:, 1]
        holdout[name] = metrics(y[ho], probs[name])
        print(f"  {name}: holdout {holdout[name]}")

    # paired bootstrap on per-game log-loss, baseline minus challenger
    # (positive diff = draft model better)
    def pergame_ll(p):
        return -(y[ho] * np.log(p) + (1 - y[ho]) * np.log(1 - p))
    d = pergame_ll(probs["M1_elo"]) - pergame_ll(probs[gate])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    ci = [round(float(v), 5) for v in np.percentile(boots, [2.5, 97.5])]
    gate_result = {
        "challenger": gate,
        "holdout_ll_diff_mean": round(float(d.mean()), 5),
        "holdout_ll_diff_ci95": ci,
        "p_challenger_better": round(float((boots > 0).mean()), 4),
    }
    print(f"gate: {gate_result}")

    out = {
        "experiment": "v0.9 rung 0 — outcome-head falsifiable baseline",
        "question": ("does the completed draft improve win-prediction "
                     "log-loss over side + team-strength (online Elo)?"),
        "splits": {
            "train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date()),
            "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "n_holdout": int(ho.sum()),
            "frozen_ewc_main_event_excluded": int(is_frozen.sum()),
        },
        "elo_k": best_k, "logistic_c": best_c,
        "val": val_results, "holdout": holdout, "gate": gate_result,
        "caveats": [
            "holdout is ~a few hundred games; only coarse effects are "
            "detectable — treat the CI, not the point estimate, as the result",
            "team strength = online Elo over teamname; rebrands reset ratings",
            "draft features see the COMPLETED draft — this is the easiest "
            "possible setting for draft signal; a per-slot head can only "
            "have less",
            "champion indicators can proxy residual team identity "
            "(signature picks of strong teams) beyond Elo — part of the "
            "edge may be who-picks-it, not what-wins; M2b (trailing champ "
            "winrate, team-agnostic) also beating M1 suggests the signal "
            "is not only that",
            "holdout scored once per design listed here; no further "
            "iteration against it",
        ],
    }
    path = DATA_PROCESSED / "outcome_baseline_v09.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
