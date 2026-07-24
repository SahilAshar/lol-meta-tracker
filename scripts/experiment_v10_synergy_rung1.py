"""Synergy rung 1: do champion-pair interactions predict soloq wins?

The falsifiable gate for the whole "synergy lift" direction (research doc
docs/2026-07-23-soloq-synergy-research.md §5, execution handoff
docs/2026-07-23-synergy-rung1-handoff.md). One row per clean soloq game,
predict blue win. M0: intercept only (side edge). M1: + signed champion main
effects. M2: M1 + signed same-team pair indicators. M3: M2 + signed
cross-team counter pairs. GO if M2 (or M3) beats M1 on holdout log-loss with
a 10k-resample paired-bootstrap 95% CI excluding 0; NO-GO if within noise —
a null is a real answer and redirects the soloq asset to meta-rate features.

Discipline: chronological 70/15/15 splits by game_creation; C swept per
model on val only; holdout scored exactly once per final model. Soloq only —
no pro data, and the frozen EWC test set is not in this dataset at all.

Writes data/processed/synergy_rung1.json.
"""

from __future__ import annotations

import json
import sqlite3
import time
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from common import DATA_PROCESSED, DATA_RAW

DB = DATA_RAW / "soloq" / "soloq.db"
MIN_CLEAN_GAMES = 76_995  # verified anchor 2026-07-23; scraper may add more
EXPECTED_CHAMPS = 173
BLUE_WR_ANCHOR = 0.4789
CS = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_BOOT = 10_000
SHRINK_GAMES = 200.0  # EB prior strength for the descriptive pair-lift tables
RNG = np.random.default_rng(20260723)


def load_games() -> list[dict]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT payload, game_creation, game_version FROM matches "
        "WHERE state='done' AND payload IS NOT NULL "
        "AND queue_id=420 AND duration>=300"
    ).fetchall()
    con.close()
    games = []
    for payload, creation, version in rows:
        parts = json.loads(payload).get("participants", [])
        if len(parts) != 10:
            continue
        blue = sorted(p["champ"] for p in parts if p["team"] == 100)
        red = sorted(p["champ"] for p in parts if p["team"] == 200)
        if len(blue) != 5 or len(red) != 5:
            continue
        blue_win = next(p["win"] for p in parts if p["team"] == 100)
        games.append({
            "creation": creation, "version": version,
            "blue": blue, "red": red, "blue_win": bool(blue_win),
        })
    return games


def build_design(games: list[dict], champs: list[str]):
    """Signed sparse blocks: main effects, same-team pairs, counter pairs.

    mains[c] = +1 on blue / -1 on red. pair{A,B} (same team, A<B) = +1 on
    blue / -1 on red. ctr{A,B} (A<B alphabetically, opposite teams) = +1
    when A is on blue, -1 when A is on red — antisymmetry lives in the one
    signed column.
    """
    c_idx = {c: i for i, c in enumerate(champs)}
    pairs = list(combinations(champs, 2))
    p_idx = {p: i for i, p in enumerate(pairs)}

    def block(n_cols, entries_per_game):
        data, indices, indptr = [], [], [0]
        for g in games:
            cols = entries_per_game(g)
            for col, val in sorted(cols.items()):
                indices.append(col)
                data.append(val)
            indptr.append(len(indices))
        return csr_matrix(
            (np.array(data, dtype=np.float32), indices, indptr),
            shape=(len(games), n_cols),
        )

    def main_cols(g):
        cols = {c_idx[c]: 1.0 for c in g["blue"]}
        cols.update({c_idx[c]: -1.0 for c in g["red"]})
        return cols

    def team_pair_cols(g):
        cols = {}
        for team, sign in ((g["blue"], 1.0), (g["red"], -1.0)):
            for a, b in combinations(team, 2):
                cols[p_idx[(a, b)]] = sign
        return cols

    def counter_cols(g):
        cols = {}
        for a in g["blue"]:
            for b in g["red"]:
                lo, hi = (a, b) if a < b else (b, a)
                cols[p_idx[(lo, hi)]] = 1.0 if lo == a else -1.0
        return cols

    return (block(len(champs), main_cols),
            block(len(pairs), team_pair_cols),
            block(len(pairs), counter_cols),
            pairs)


def fit_eval(X_tr, y_tr, X_va, y_va, name):
    """Sweep C on val; return best model + val metrics."""
    best = None
    for c in CS:
        t0 = time.time()
        clf = LogisticRegression(penalty="l2", C=c, solver="lbfgs",
                                 max_iter=2000, tol=1e-6)
        clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        p = clf.predict_proba(X_va)[:, 1]
        ll = log_loss(y_va, p)
        print(f"  {name} C={c}: val ll={ll:.5f} ({elapsed:.0f}s)")
        assert elapsed < 300, f"{name} C={c} took {elapsed:.0f}s — rethink grid"
        if best is None or ll < best["val_ll"]:
            best = {"C": c, "val_ll": ll, "clf": clf}
    assert best["C"] not in (CS[0], CS[-1]) or name == "M0", \
        f"{name} best C={best['C']} at sweep edge — extend the grid"
    return best


def paired_bootstrap(ll_a: np.ndarray, ll_b: np.ndarray):
    """95% CI of mean per-game log-loss difference a - b (positive = b better)."""
    d = ll_a - ll_b
    n = len(d)
    means = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return d.mean(), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def per_game_ll(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def pair_lift_tables(games, champs):
    """Descriptive EB-shrunk pair lifts from the train split only."""
    champ_w = {c: [0, 0] for c in champs}  # wins, games
    syn = {}  # (a,b) -> [wins, games] same-team
    ctr = {}  # (a,b) -> [a_wins, games] a vs b opposite teams
    for g in games:
        for team, won in ((g["blue"], g["blue_win"]), (g["red"], not g["blue_win"])):
            for c in team:
                champ_w[c][0] += won
                champ_w[c][1] += 1
            for a, b in combinations(team, 2):
                s = syn.setdefault((a, b), [0, 0])
                s[0] += won
                s[1] += 1
        for a in g["blue"]:
            for b in g["red"]:
                lo, hi = (a, b) if a < b else (b, a)
                lo_won = g["blue_win"] if lo == a else not g["blue_win"]
                s = ctr.setdefault((lo, hi), [0, 0])
                s[0] += lo_won
                s[1] += 1
    wr = {c: w / n for c, (w, n) in champ_w.items()}

    def shrunk(table, expected):
        out = []
        for (a, b), (w, n) in table.items():
            lift = (w / n - expected(a, b)) * n / (n + SHRINK_GAMES)
            out.append({"pair": f"{a}+{b}", "games": n,
                        "lift": round(float(lift), 4)})
        return out

    # Expected pair WR = simple average of the two champs' train win rates.
    syn_rows = shrunk(syn, lambda a, b: (wr[a] + wr[b]) / 2)
    # Expected for counters: lo's WR vs field, adjusted for hi's strength.
    ctr_rows = shrunk(ctr, lambda a, b: (wr[a] + (1 - wr[b])) / 2)
    top_syn = sorted(syn_rows, key=lambda r: -r["lift"])[:20]
    top_ctr = sorted(ctr_rows, key=lambda r: -abs(r["lift"]))[:20]
    return top_syn, top_ctr


def main() -> None:
    games = load_games()
    games = [g for g in games if g["creation"] > 0]  # drop epoch-1970 rows
    games.sort(key=lambda g: g["creation"])
    n = len(games)
    assert n >= MIN_CLEAN_GAMES, f"only {n} clean games (expected >= {MIN_CLEAN_GAMES})"

    y = np.array([g["blue_win"] for g in games], dtype=np.float64)
    blue_wr = y.mean()
    print(f"{n} clean games, blue WR {blue_wr:.4f}")
    assert abs(blue_wr - BLUE_WR_ANCHOR) < 0.01, \
        f"blue WR {blue_wr:.4f} far from anchor {BLUE_WR_ANCHOR} — team-id bug?"

    champs = sorted({c for g in games for c in g["blue"] + g["red"]})
    assert len(champs) == EXPECTED_CHAMPS, f"{len(champs)} champs != {EXPECTED_CHAMPS}"

    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    bounds = {
        "train_end": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(games[i_tr - 1]["creation"] / 1000)),
        "val_end": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(games[i_va - 1]["creation"] / 1000)),
        "holdout_end": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(games[-1]["creation"] / 1000)),
    }
    print(f"splits: train {i_tr}, val {i_va - i_tr}, holdout {n - i_va} — {bounds}")

    print("building design matrices...")
    X_main, X_pair, X_ctr, pairs = build_design(games, champs)
    designs = {
        "M1": X_main,
        "M2": hstack([X_main, X_pair], format="csr"),
        "M3": hstack([X_main, X_pair, X_ctr], format="csr"),
    }

    results = {}
    # M0: intercept only — closed form on train base rate.
    p0 = y[:i_tr].mean()
    results["M0"] = {
        "C": None,
        "val": {"log_loss": float(log_loss(y[i_tr:i_va], np.full(i_va - i_tr, p0)))},
        "holdout": {
            "log_loss": float(log_loss(y[i_va:], np.full(n - i_va, p0))),
            "auc": 0.5,
            "acc": float(max(1 - y[i_va:].mean(), y[i_va:].mean())),
        },
    }

    holdout_p = {}
    for name, X in designs.items():
        print(f"sweeping {name} ({X.shape[1]} cols)...")
        best = fit_eval(X[:i_tr], y[:i_tr], X[i_tr:i_va], y[i_tr:i_va], name)
        p_ho = best["clf"].predict_proba(X[i_va:])[:, 1]  # scored once
        holdout_p[name] = p_ho
        results[name] = {
            "C": best["C"],
            "val": {"log_loss": float(best["val_ll"])},
            "holdout": {
                "log_loss": float(log_loss(y[i_va:], p_ho)),
                "auc": float(roc_auc_score(y[i_va:], p_ho)),
                "acc": float(((p_ho > 0.5) == y[i_va:]).mean()),
            },
        }
        print(f"  {name} holdout: {results[name]['holdout']}")

    y_ho = y[i_va:]
    ll1 = per_game_ll(y_ho, holdout_p["M1"])
    gates = {}
    for name in ("M2", "M3"):
        mean_d, lo, hi = paired_bootstrap(ll1, per_game_ll(y_ho, holdout_p[name]))
        gates[f"M1_vs_{name}"] = {
            "mean_delta": round(float(mean_d), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "go": lo > 0,
        }
    verdict = "GO" if any(g["go"] for g in gates.values()) else "NO-GO"
    print(f"gates: {gates}\nVERDICT: {verdict}")

    print("computing train-split pair-lift tables...")
    top_syn, top_ctr = pair_lift_tables(games[:i_tr], champs)

    versions = {}
    for g in games:
        v = ".".join(str(g["version"]).split(".")[:2])
        versions[v] = versions.get(v, 0) + 1

    out = {
        "experiment": "synergy_rung1",
        "question": "do champion-pair interactions predict soloq wins beyond main effects?",
        "verdict": verdict,
        "gates": gates,
        "models": results,
        "provenance": {
            "clean_games": n,
            "champions": len(champs),
            "blue_win_rate": round(float(blue_wr), 4),
            "patch_mix": versions,
            "split_boundaries": bounds,
            "split_sizes": {"train": i_tr, "val": i_va - i_tr, "holdout": n - i_va},
            "c_grid": CS,
            "n_bootstrap": N_BOOT,
        },
        "top_synergy_pairs": top_syn,
        "top_counter_pairs": top_ctr,
    }
    path = DATA_PROCESSED / "synergy_rung1.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
