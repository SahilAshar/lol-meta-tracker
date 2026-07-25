"""Synergy rung 1c: the single-scalar duo-lift test.

Executes docs/2026-07-25-synergy-rung1c-spec.md after rung 1b's split
verdict (slotting GO, counters GO, same-team pair synergy NULL at 323k
games). Maximum-power 1-df version of the synergy test: collapse each
team's claimed duo synergy into one scalar (sum of EB-shrunk train-split
duo lifts over the four priority vectors) and ask whether it improves
held-out prediction over champ@role main effects. A same-construction
counter scalar is the positive control — counters are proven real
(1b's B-vs-C CI [+0.000274, +0.000713]), so if the scalar mechanism
can't detect them the SYN result is uninterpretable, not a falsification.

Models: A (champ@role mains, 1b's design refit), A+SYN (the test),
A+CTR (positive control), A+SYN+CTR (interaction check).

Primary gate: A+SYN beats A on holdout log-loss, 10k paired bootstrap
95% CI excluding 0. Control gate: A+CTR must beat A.

Loader and discipline imported verbatim from experiment_v10_synergy_rung1b:
clean-game filter, chronological 70/15/15 by game_creation, C swept per
model on val only, holdout scored once. Lift tables are built on the
train split ONLY; no val/holdout games touch any table.

Leakage repair (first run tripped the sweep-edge assert): the scalars are
outcome-derived, so a train game's own result leaks into its own pair's
lift and the in-sample-inflated feature wrecks val (A+SYN val LL rose
monotonically in C, 0.694->0.708 vs A's 0.687). Train-game scalars are
therefore computed leave-one-out on the pair counts (drop the game's own
outcome: n-1 games, w-won wins, shrink (n-1)/(n-1+200)); val/holdout
scalars use the full train tables — those games are disjoint from the
tables, so no leakage exists there. Residual self-influence through the
champ@role means is O(1/n_combo), negligible, and shared symmetrically
by SYN and CTR — the control gate validates the repaired mechanism.

Writes data/processed/synergy_rung1c.json.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import log_loss, roc_auc_score

from common import DATA_PROCESSED
from experiment_v10_synergy_rung1b import (
    BLUE_WR_ANCHOR,
    CS,
    MIN_CLEAN_GAMES,
    N_BOOT,
    ROLES,
    SHRINK_GAMES,
    SYN_VECTORS,
    build_designs,
    fit_eval,
    load_games,
    paired_bootstrap,
    per_game_ll,
)


def build_lift_tables(train: list[dict]):
    """Train-split-only EB-shrunk lift tables (spec: identical to 1b's D).

    wr(c) = champ train WR; wr(c@r) = champ@role WR EB-shrunk toward wr(c)
    with prior SHRINK_GAMES. Duo lift per priority vector:
      syn_lift(a@ra, b@rb) = shrink_200(WR_obs(pair) - mean(wr(a@ra), wr(b@rb)))
    Counter lift per same-role matchup, antisymmetric key (lo, hi) alphabetical:
      ctr_lift(lo, hi | r) = shrink_200(WR_obs(lo beats hi in r)
                                        - (wr(lo@r) + 1 - wr(hi@r)) / 2)
    Unobserved pairs contribute 0 downstream.
    """
    champ_w = defaultdict(lambda: [0, 0])
    combo_w = defaultdict(lambda: [0, 0])
    duo_w = {v: defaultdict(lambda: [0, 0]) for v in SYN_VECTORS}
    ctr_w = {r: defaultdict(lambda: [0, 0]) for r in ROLES}

    for g in train:
        for side, won in ((g["blue"], g["blue_win"]),
                          (g["red"], not g["blue_win"])):
            for role, champ in side.items():
                champ_w[champ][0] += won
                champ_w[champ][1] += 1
                combo_w[(champ, role)][0] += won
                combo_w[(champ, role)][1] += 1
            for v in SYN_VECTORS:
                ra, rb = v
                s = duo_w[v][(side[ra], side[rb])]
                s[0] += won
                s[1] += 1
        for role in ROLES:
            a, b = g["blue"][role], g["red"][role]
            lo, hi = (a, b) if a < b else (b, a)
            lo_won = g["blue_win"] if lo == a else not g["blue_win"]
            s = ctr_w[role][(lo, hi)]
            s[0] += lo_won
            s[1] += 1

    champ_wr = {c: w / n for c, (w, n) in champ_w.items()}

    def role_wr(champ, role):
        w, n = combo_w.get((champ, role), (0, 0))
        prior = champ_wr[champ]
        return (w + prior * SHRINK_GAMES) / (n + SHRINK_GAMES)

    # Each table maps key -> (wins, games, expected_wr); lift is computed at
    # lookup time so train games can drop their own outcome (LOO).
    syn_tables = {}
    for v in SYN_VECTORS:
        ra, rb = v
        syn_tables[v] = {
            (a, b): (w, n, (role_wr(a, ra) + role_wr(b, rb)) / 2)
            for (a, b), (w, n) in duo_w[v].items()
        }
    ctr_tables = {}
    for role in ROLES:
        ctr_tables[role] = {
            (lo, hi): (w, n, (role_wr(lo, role) + 1 - role_wr(hi, role)) / 2)
            for (lo, hi), (w, n) in ctr_w[role].items()
        }
    return syn_tables, ctr_tables


def _lift(entry, drop_won=None):
    """EB-shrunk lift from a (wins, games, expected) entry.

    drop_won (train games only) removes the querying game's own outcome
    before computing the lift; None (val/holdout) uses the full counts.
    Unobserved after the drop -> 0.
    """
    if entry is None:
        return 0.0
    w, n, expected = entry
    if drop_won is not None:
        w, n = w - drop_won, n - 1
    if n <= 0:
        return 0.0
    return (w / n - expected) * n / (n + SHRINK_GAMES)


def game_scalars(games: list[dict], syn_tables, ctr_tables, i_tr: int):
    """Signed per-game scalars, blue minus red; LOO inside the train split."""
    syn = np.zeros(len(games))
    ctr = np.zeros(len(games))
    for i, g in enumerate(games):
        in_train = i < i_tr
        s = 0.0
        for v in SYN_VECTORS:
            ra, rb = v
            tbl = syn_tables[v]
            for side, sign, won in ((g["blue"], 1.0, g["blue_win"]),
                                    (g["red"], -1.0, not g["blue_win"])):
                entry = tbl.get((side[ra], side[rb]))
                s += sign * _lift(entry, drop_won=won if in_train else None)
        syn[i] = s
        c = 0.0
        for role in ROLES:
            a, b = g["blue"][role], g["red"][role]
            lo, hi = (a, b) if a < b else (b, a)
            lo_won = g["blue_win"] if lo == a else not g["blue_win"]
            entry = ctr_tables[role].get((lo, hi))
            lift = _lift(entry, drop_won=lo_won if in_train else None)
            c += lift if lo == a else -lift
        ctr[i] = c
    return syn, ctr


def decile_table(scores, y_actual, p_baseline):
    """Equal-count decile buckets of a holdout score vs actual and A-predicted WR."""
    order = np.argsort(scores, kind="stable")
    buckets = np.array_split(order, 10)
    rows = []
    for d, idx in enumerate(buckets, start=1):
        rows.append({
            "decile": d,
            "n": int(len(idx)),
            "score_mean": round(float(scores[idx].mean()), 5),
            "score_range": [round(float(scores[idx].min()), 5),
                            round(float(scores[idx].max()), 5)],
            "actual_blue_wr": round(float(y_actual[idx].mean()), 4),
            "model_A_predicted_wr": round(float(p_baseline[idx].mean()), 4),
        })
    return rows


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

    print("building A design (1b's champ@role block)...", flush=True)
    _, X_cr, _, _, champs, feat_meta = build_designs(games, i_tr)

    print("building train-split lift tables...", flush=True)
    syn_tables, ctr_tables = build_lift_tables(games[:i_tr])
    syn_raw, ctr_raw = game_scalars(games, syn_tables, ctr_tables, i_tr)

    scalar_stats = {}
    scaled = {}
    for name, raw in (("SYN", syn_raw), ("CTR", ctr_raw)):
        mu, sd = raw[:i_tr].mean(), raw[:i_tr].std()
        assert sd > 0, f"{name} train sd is 0 — table construction broken"
        scaled[name] = (raw - mu) / sd
        scalar_stats[name] = {
            "train_mean": round(float(mu), 6),
            "train_sd": round(float(sd), 6),
            "train_nonzero_frac": round(float((raw[:i_tr] != 0).mean()), 4),
            "holdout_nonzero_frac": round(float((raw[i_va:] != 0).mean()), 4),
        }
    print(f"scalar stats: {scalar_stats}", flush=True)

    def col(v):
        return csr_matrix(v.reshape(-1, 1).astype(np.float32))

    designs = {
        "A": X_cr,
        "A+SYN": hstack([X_cr, col(scaled["SYN"])], format="csr"),
        "A+CTR": hstack([X_cr, col(scaled["CTR"])], format="csr"),
        "A+SYN+CTR": hstack([X_cr, col(scaled["SYN"]), col(scaled["CTR"])],
                            format="csr"),
    }

    results, holdout_p, coefs = {}, {}, {}
    for name, X in designs.items():
        print(f"sweeping {name} ({X.shape[1]} cols)...", flush=True)
        best = fit_eval(X[:i_tr], y[:i_tr], X[i_tr:i_va], y[i_tr:i_va], name)
        p_ho = best["clf"].predict_proba(X[i_va:])[:, 1]  # scored once
        holdout_p[name] = p_ho
        n_scalars = X.shape[1] - X_cr.shape[1]
        if n_scalars:
            coefs[name] = [round(float(c), 6)
                           for c in best["clf"].coef_[0][-n_scalars:]]
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
    for base, challenger in (("A", "A+SYN"), ("A", "A+CTR"),
                             ("A+CTR", "A+SYN+CTR")):
        mean_d, lo, hi = paired_bootstrap(
            per_game_ll(y_ho, holdout_p[base]),
            per_game_ll(y_ho, holdout_p[challenger]))
        gates[f"{base}_vs_{challenger}"] = {
            "mean_delta": round(float(mean_d), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "go": lo > 0,
        }

    control_ok = gates["A_vs_A+CTR"]["go"]
    syn_ok = gates["A_vs_A+SYN"]["go"]
    if not control_ok:
        verdict = "UNINTERPRETABLE"  # mechanism broken — debug, don't conclude
    elif syn_ok:
        verdict = "GO"
    else:
        verdict = "NO-GO"
    print(f"gates: {gates}\nVERDICT: {verdict}", flush=True)

    # Implied WR effect per point (0.01 WR) of summed claimed lift, at p=0.5:
    # beta_std / train_sd is the logit slope per raw SYN unit; x0.25 x0.01.
    implied = {}
    for name, scalar in (("A+SYN", "SYN"), ("A+CTR", "CTR")):
        beta = coefs[name][0]
        sd = scalar_stats[scalar]["train_sd"]
        implied[scalar] = {
            "coef_standardized": beta,
            "implied_wr_per_point_of_lift": round(beta / sd * 0.25 * 0.01, 6),
        }

    print("building holdout decile tables...", flush=True)
    deciles = {
        "SYN": decile_table(syn_raw[i_va:], y_ho, holdout_p["A"]),
        "CTR": decile_table(ctr_raw[i_va:], y_ho, holdout_p["A"]),
    }

    versions = {}
    for g in games:
        v = ".".join(str(g["version"]).split(".")[:2])
        versions[v] = versions.get(v, 0) + 1

    out = {
        "experiment": "synergy_rung1c",
        "question": ("does a single U.GG-style duo-synergy scalar improve "
                     "held-out prediction beyond champ@role main effects?"),
        "verdict": verdict,
        "control_gate_passed": control_ok,
        "gates": gates,
        "models": results,
        "scalar_coefficients": {"per_model": coefs, "implied_effects": implied},
        "scalar_stats": scalar_stats,
        "holdout_decile_tables": deciles,
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
            "shrink_games": SHRINK_GAMES,
            "champ_role_combos_retained": feat_meta["champ_role_combos_retained"],
            "syn_pairs_in_tables": {f"{a}+{b}": len(syn_tables[(a, b)])
                                    for a, b in SYN_VECTORS},
            "ctr_pairs_in_tables": {r: len(ctr_tables[r]) for r in ROLES},
            "runtime_seconds": round(time.time() - t_start),
        },
    }
    path = DATA_PROCESSED / "synergy_rung1c.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({round(time.time() - t_start)}s total)")


if __name__ == "__main__":
    main()
