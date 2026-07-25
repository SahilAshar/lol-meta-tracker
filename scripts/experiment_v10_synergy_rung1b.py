"""Synergy rung 1b: role-aware pairs and slot permutations.

Executes docs/2026-07-23-synergy-rung1b-spec.md after rung 1's role-blind
NO-GO (docs/2026-07-23-synergy-rung1-results.md). One row per clean soloq
game, predict blue win, signed indicators (+1 blue / -1 red).

Ladder (each step falsifiable on its own):
  M1: champ main effects (rung 1's M1, refit at current scale — baseline).
  A:  champ-at-role main effects; combos under MIN_COMBO train games pool
      into a champ-only fallback column.
  B:  A + same-team role-pair interactions restricted to the four priority
      vectors (BOT+UTIL, MID+JG, TOP+JG, JG+UTIL). Per-vector ablations
      reported so "bot duo matters, top-jungle doesn't" is available.
  C:  B + lane counters: cross-team same-role pairs (top/jg/mid, bot 2v2
      collapsed to ADC-vs-ADC + SUP-vs-SUP).
  D:  descriptive slot-permutation table (not gated): champ pairs observed
      in >=2 distinct role-slottings, EB-shrunk lifts side by side.
      Camille+Galio reported explicitly regardless of thresholds.

GO if B beats A, or C beats B, on holdout log-loss with the 10k paired
bootstrap 95% CI excluding 0. A-vs-M1 is reported but gates interpretation
only. Discipline: chronological 70/15/15 by game_creation; C swept per
gated model on val only; holdout scored exactly once per final model;
feature retention thresholds computed on the train split only.

Writes data/processed/synergy_rung1b.json.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from common import DATA_PROCESSED, DATA_RAW

DB = DATA_RAW / "soloq" / "soloq.db"
MIN_CLEAN_GAMES = 300_000  # db pulled 2026-07-25 with 329,703 done games
BLUE_WR_ANCHOR = 0.4789    # rung 1 anchor; assert within 1 point
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
SYN_VECTORS = (            # same-team priority vectors, spec §ladder B
    ("BOTTOM", "UTILITY"),
    ("MIDDLE", "JUNGLE"),
    ("TOP", "JUNGLE"),
    ("JUNGLE", "UTILITY"),
)
MIN_COMBO = 50       # train games for a champ@role column (spec: ~50)
MIN_PAIR = 50        # train games for a synergy/counter pair column
MIN_PERM = 150       # train games per slotting for the D table
CS = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_BOOT = 10_000
SHRINK_GAMES = 200.0  # EB prior strength for descriptive tables
RNG = np.random.default_rng(20260725)


def load_games() -> tuple[list[dict], dict]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT payload, game_creation, game_version FROM matches "
        "WHERE state='done' AND payload IS NOT NULL "
        "AND queue_id=420 AND duration>=300"
    ).fetchall()
    con.close()
    games, drops = [], Counter()
    for payload, creation, version in rows:
        if not creation or creation <= 0:
            drops["bad_creation"] += 1
            continue
        parts = json.loads(payload).get("participants", [])
        if len(parts) != 10:
            drops["not_10_participants"] += 1
            continue
        teams = {100: {}, 200: {}}
        ok = True
        for p in parts:
            team = teams.get(p["team"])
            pos = p.get("pos")
            if team is None or pos not in ROLES or pos in team:
                ok = False  # unknown team, blank role, or dup role per team
                break
            team[pos] = p["champ"]
        if not ok or len(teams[100]) != 5 or len(teams[200]) != 5:
            drops["bad_roles"] += 1
            continue
        blue_win = next(p["win"] for p in parts if p["team"] == 100)
        games.append({
            "creation": creation, "version": version,
            "blue": teams[100], "red": teams[200],  # role -> champ
            "blue_win": bool(blue_win),
        })
    return games, dict(drops)


def build_designs(games: list[dict], i_tr: int):
    """Signed sparse blocks with train-split-only column selection.

    Returns (X_main, X_cr, X_syn_by_vector, X_ctr, meta). All blocks span
    every game; only which columns exist is decided on games[:i_tr].
    """
    train = games[:i_tr]
    champs = sorted({c for g in train for c in
                     list(g["blue"].values()) + list(g["red"].values())})
    c_idx = {c: i for i, c in enumerate(champs)}

    # --- A: champ@role columns (>= MIN_COMBO train games) + champ fallback.
    combo_n = Counter()
    for g in train:
        for side in (g["blue"], g["red"]):
            for role, champ in side.items():
                combo_n[(champ, role)] += 1
    combos = sorted(k for k, n in combo_n.items() if n >= MIN_COMBO)
    cr_idx = {k: i for i, k in enumerate(combos)}

    # --- B: same-team pairs per priority vector (>= MIN_PAIR train games).
    syn_n = {v: Counter() for v in SYN_VECTORS}
    for g in train:
        for side in (g["blue"], g["red"]):
            for ra, rb in SYN_VECTORS:
                syn_n[(ra, rb)][(side[ra], side[rb])] += 1
    syn_idx = {}
    for v in SYN_VECTORS:
        keys = sorted(k for k, n in syn_n[v].items() if n >= MIN_PAIR)
        syn_idx[v] = {k: i for i, k in enumerate(keys)}

    # --- C: same-role cross-team counters (>= MIN_PAIR train games).
    ctr_n = {r: Counter() for r in ROLES}
    for g in train:
        for role in ROLES:
            a, b = g["blue"][role], g["red"][role]
            lo, hi = (a, b) if a < b else (b, a)
            ctr_n[role][(lo, hi)] += 1
    ctr_idx, ctr_off = {}, {}
    off = 0
    for role in ROLES:
        keys = sorted(k for k, n in ctr_n[role].items() if n >= MIN_PAIR)
        ctr_idx[role] = {k: i for i, k in enumerate(keys)}
        ctr_off[role] = off
        off += len(keys)
    n_ctr_cols = off

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
        cols = defaultdict(float)
        for side, sign in ((g["blue"], 1.0), (g["red"], -1.0)):
            for champ in side.values():
                if champ in c_idx:  # champs unseen in train contribute 0
                    cols[c_idx[champ]] += sign
        return cols

    def cr_cols(g):
        # champ@role column when retained, else champ fallback column.
        cols = defaultdict(float)
        n = len(combos)
        for side, sign in ((g["blue"], 1.0), (g["red"], -1.0)):
            for role, champ in side.items():
                if (champ, role) in cr_idx:
                    cols[cr_idx[(champ, role)]] += sign
                elif champ in c_idx:
                    cols[n + c_idx[champ]] += sign
        return cols

    def syn_cols_for(v):
        idx, (ra, rb) = syn_idx[v], v

        def cols_fn(g):
            cols = {}
            for side, sign in ((g["blue"], 1.0), (g["red"], -1.0)):
                key = (side[ra], side[rb])
                if key in idx:
                    cols[idx[key]] = sign
            return cols
        return cols_fn

    def ctr_cols(g):
        cols = {}
        for role in ROLES:
            a, b = g["blue"][role], g["red"][role]
            lo, hi = (a, b) if a < b else (b, a)
            if (lo, hi) in ctr_idx[role]:
                cols[ctr_off[role] + ctr_idx[role][(lo, hi)]] = \
                    1.0 if lo == a else -1.0
        return cols

    X_main = block(len(champs), main_cols)
    X_cr = block(len(combos) + len(champs), cr_cols)
    X_syn = {v: block(len(syn_idx[v]), syn_cols_for(v)) for v in SYN_VECTORS}
    X_ctr = block(n_ctr_cols, ctr_cols)

    meta = {
        "champions": len(champs),
        "champ_role_combos_retained": len(combos),
        "syn_cols_per_vector": {f"{a}+{b}": len(syn_idx[(a, b)])
                                for a, b in SYN_VECTORS},
        "counter_cols_per_role": {r: len(ctr_idx[r]) for r in ROLES},
    }
    return X_main, X_cr, X_syn, X_ctr, champs, meta


def fit_eval(X_tr, y_tr, X_va, y_va, name, cs=CS):
    """Sweep C on val; return best model + val metrics."""
    best = None
    for c in cs:
        t0 = time.time()
        clf = LogisticRegression(penalty="l2", C=c, solver="lbfgs",
                                 max_iter=2000, tol=1e-6)
        clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        p = clf.predict_proba(X_va)[:, 1]
        ll = log_loss(y_va, p)
        print(f"  {name} C={c}: val ll={ll:.5f} ({elapsed:.0f}s)", flush=True)
        assert elapsed < 300, f"{name} C={c} took {elapsed:.0f}s — rethink grid"
        if best is None or ll < best["val_ll"]:
            best = {"C": c, "val_ll": ll, "clf": clf}
    assert best["C"] not in (cs[0], cs[-1]) or len(cs) < len(CS), \
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


def permutation_table(train: list[dict]):
    """D: per-slotting EB-shrunk lifts for champ pairs with >=2 slottings.

    Expected pair WR = mean of the two champ@role WRs, each EB-shrunk
    toward the champion's overall train WR (prior SHRINK_GAMES). The pair
    lift is then shrunk by n/(n+SHRINK_GAMES), as rung 1.
    """
    champ_w = defaultdict(lambda: [0, 0])
    combo_w = defaultdict(lambda: [0, 0])
    perm_w = defaultdict(lambda: [0, 0])  # (a, b, ra, rb) a<b alphabetical
    for g in train:
        for side, won in ((g["blue"], g["blue_win"]),
                          (g["red"], not g["blue_win"])):
            for role, champ in side.items():
                champ_w[champ][0] += won
                champ_w[champ][1] += 1
                combo_w[(champ, role)][0] += won
                combo_w[(champ, role)][1] += 1
            for (ra, ca), (rb, cb) in combinations(sorted(side.items()), 2):
                if ca > cb:
                    (ra, ca), (rb, cb) = (rb, cb), (ra, ca)
                s = perm_w[(ca, cb, ra, rb)]
                s[0] += won
                s[1] += 1

    champ_wr = {c: w / n for c, (w, n) in champ_w.items()}

    def role_wr(champ, role):
        w, n = combo_w.get((champ, role), (0, 0))
        prior = champ_wr[champ]
        return (w + prior * SHRINK_GAMES) / (n + SHRINK_GAMES)

    rows_by_pair = defaultdict(list)
    for (a, b, ra, rb), (w, n) in perm_w.items():
        expected = (role_wr(a, ra) + role_wr(b, rb)) / 2
        lift = (w / n - expected) * n / (n + SHRINK_GAMES)
        rows_by_pair[(a, b)].append({
            "slotting": f"{a}@{ra} + {b}@{rb}", "games": n,
            "raw_wr": round(w / n, 4), "expected_wr": round(expected, 4),
            "lift": round(float(lift), 4),
        })

    table = []
    for (a, b), rows in rows_by_pair.items():
        qual = [r for r in rows if r["games"] >= MIN_PERM]
        if len(qual) < 2:
            continue
        qual.sort(key=lambda r: -r["lift"])
        table.append({
            "pair": f"{a}+{b}",
            "n_slottings": len(qual),
            "lift_spread": round(qual[0]["lift"] - qual[-1]["lift"], 4),
            "slottings": qual,
        })
    table.sort(key=lambda r: -r["lift_spread"])

    camille_galio = sorted(
        rows_by_pair.get(("Camille", "Galio"), []),
        key=lambda r: -r["games"])
    return table, camille_galio


def main() -> None:
    t_start = time.time()
    games, drops = load_games()
    games.sort(key=lambda g: g["creation"])
    n = len(games)
    print(f"{n} clean games (drops: {drops})")
    assert n >= MIN_CLEAN_GAMES, f"only {n} clean games (expected >= {MIN_CLEAN_GAMES})"

    y = np.array([g["blue_win"] for g in games], dtype=np.float64)
    blue_wr = y.mean()
    print(f"blue WR {blue_wr:.4f}")
    assert abs(blue_wr - BLUE_WR_ANCHOR) < 0.01, \
        f"blue WR {blue_wr:.4f} far from anchor {BLUE_WR_ANCHOR} — team-id bug?"

    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    bounds = {
        "train_end": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(games[i_tr - 1]["creation"] / 1000)),
        "val_end": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(games[i_va - 1]["creation"] / 1000)),
        "holdout_end": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(games[-1]["creation"] / 1000)),
    }
    print(f"splits: train {i_tr}, val {i_va - i_tr}, holdout {n - i_va} — {bounds}")

    print("building design matrices...", flush=True)
    X_main, X_cr, X_syn, X_ctr, champs, feat_meta = build_designs(games, i_tr)
    print(f"feature counts: {feat_meta}", flush=True)

    X_syn_all = hstack(list(X_syn.values()), format="csr")
    designs = {
        "M1": X_main,
        "A": X_cr,
        "B": hstack([X_cr, X_syn_all], format="csr"),
        "C": hstack([X_cr, X_syn_all, X_ctr], format="csr"),
    }

    results, holdout_p = {}, {}
    for name, X in designs.items():
        print(f"sweeping {name} ({X.shape[1]} cols)...", flush=True)
        best = fit_eval(X[:i_tr], y[:i_tr], X[i_tr:i_va], y[i_tr:i_va], name)
        p_ho = best["clf"].predict_proba(X[i_va:])[:, 1]  # scored once
        holdout_p[name] = p_ho
        results[name] = {
            "C": best["C"],
            "n_cols": int(X.shape[1]),
            "val": {"log_loss": float(best["val_ll"])},
            "holdout": {
                "log_loss": float(log_loss(y[i_va:], p_ho)),
                "auc": float(roc_auc_score(y[i_va:], p_ho)),
                "acc": float(((p_ho > 0.5) == y[i_va:]).mean()),
            },
        }
        print(f"  {name} holdout: {results[name]['holdout']}", flush=True)

    y_ho = y[i_va:]
    gates = {}
    for base, challenger in (("M1", "A"), ("A", "B"), ("B", "C")):
        mean_d, lo, hi = paired_bootstrap(
            per_game_ll(y_ho, holdout_p[base]),
            per_game_ll(y_ho, holdout_p[challenger]))
        gates[f"{base}_vs_{challenger}"] = {
            "mean_delta": round(float(mean_d), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "go": lo > 0,
        }
    # Rung gate is B-over-A or C-over-B; A-over-M1 gates interpretation only.
    verdict = "GO" if (gates["A_vs_B"]["go"] or gates["B_vs_C"]["go"]) else "NO-GO"
    print(f"gates: {gates}\nVERDICT: {verdict}", flush=True)

    # Per-vector ablations of B (descriptive, not gated): refit B without
    # each vector at a 3-point C neighborhood of B's best (full house-sweep
    # reserved for gated models to keep runtime sane).
    b_c = results["B"]["C"]
    ci = CS.index(b_c)
    cs_abl = CS[max(0, ci - 1):ci + 2]
    ll_b = per_game_ll(y_ho, holdout_p["B"])
    ablations = {}
    for v in SYN_VECTORS:
        vname = f"{v[0]}+{v[1]}"
        X_abl = hstack([X_cr] + [X_syn[u] for u in SYN_VECTORS if u != v],
                       format="csr")
        print(f"ablating {vname} ({X_abl.shape[1]} cols)...", flush=True)
        best = fit_eval(X_abl[:i_tr], y[:i_tr], X_abl[i_tr:i_va],
                        y[i_tr:i_va], f"B-{vname}", cs=cs_abl)
        p_ho = best["clf"].predict_proba(X_abl[i_va:])[:, 1]
        mean_d, lo, hi = paired_bootstrap(per_game_ll(y_ho, p_ho), ll_b)
        ablations[vname] = {
            "C": best["C"],
            "holdout_log_loss": float(log_loss(y_ho, p_ho)),
            # positive = full B better than B-without-this-vector,
            # i.e. this vector carries signal.
            "delta_vs_full_B": {"mean": round(float(mean_d), 6),
                                "ci95": [round(lo, 6), round(hi, 6)]},
        }
        print(f"  {vname}: {ablations[vname]['delta_vs_full_B']}", flush=True)

    print("computing slot-permutation tables (train split only)...", flush=True)
    perm_table, camille_galio = permutation_table(games[:i_tr])

    versions = {}
    for g in games:
        v = ".".join(str(g["version"]).split(".")[:2])
        versions[v] = versions.get(v, 0) + 1

    out = {
        "experiment": "synergy_rung1b",
        "question": "does role-aware slotting/synergy predict soloq wins beyond champ effects?",
        "verdict": verdict,
        "gates": gates,
        "models": results,
        "vector_ablations": ablations,
        "provenance": {
            "clean_games": n,
            "dropped": drops,
            "champions": len(champs),
            "blue_win_rate": round(float(blue_wr), 4),
            "patch_mix": versions,
            "split_boundaries": bounds,
            "split_sizes": {"train": i_tr, "val": i_va - i_tr, "holdout": n - i_va},
            "c_grid": CS,
            "n_bootstrap": N_BOOT,
            "min_train_games": {"champ_role": MIN_COMBO, "pair": MIN_PAIR,
                                "permutation_row": MIN_PERM},
            "feature_counts": feat_meta,
            "runtime_seconds": round(time.time() - t_start),
        },
        "permutation_table_top50": perm_table[:50],
        "permutation_pairs_qualifying": len(perm_table),
        "camille_galio_all_slottings": camille_galio,
    }
    path = DATA_PROCESSED / "synergy_rung1b.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({round(time.time() - t_start)}s total)")


if __name__ == "__main__":
    main()
