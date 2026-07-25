"""Rung 2 arm 2 (coach): soloq comp features in completed-draft win prediction.

Executes docs/2026-07-25-rung2-transfer-spec.md arm 2. Adds two soloq-derived
scalars to the v0.9 outcome baseline's best model M2a (online Elo + signed
champ indicators, experiment_v09_outcome_baseline.py):

  CTR     = sum over the 5 lanes of soloq_ctr(blue champ, red champ | role),
            blue perspective (already signed).
  ROLE_WR = sum of blue soloq_wr(c@r) minus sum of red soloq_wr(c@r).
  SYN     = sum over the four priority vectors of soloq duo lift, blue minus
            red (rung 1c PASS branch — optional feature, added as a third
            model M2a_soloq_syn). No leave-one-out: pro games are disjoint
            from the soloq sample the tables were built on.

Both from the ONE static lift table (soloq games < 2026-07-01 00:00 UTC,
soloq_lift_tables.py). The features are patch-blind by construction — the
table pools patches 16.11-16.14 and is applied to every game regardless of
date, per the spec's pre-registered timing caveat: soloq coverage starts
2026-06-08, the v0.9 holdout starts mid-May, so early-holdout games carry
real patch misalignment (meta shift is the noise term). Primary eval keeps
the standard v0.9 splits for comparability; a secondary slice restricts
holdout to games on/after 2026-06-08 (reported, not gated).

Discipline mirrors v0.9: Elo K reused from the stored v0.9 selection (val-
chosen there), logistic C swept on val only (fit on train), one refit on
train+val per promoted model, holdout scored exactly once, frozen EWC July
main event excluded everywhere. New scalars standardized on train statistics.

Gate: M2a+soloq beats M2a on holdout log-loss with the 10k paired bootstrap
95% CI excluding 0.

Writes data/processed/rung2_outcome.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from common import DATA_PROCESSED
from experiment_v09_outcome_baseline import (
    CS, TRAIN_END, VAL_END, build_game_table, champ_matrix, elo_diffs,
    fit_logistic, metrics,
)
from draft_dataset import load_games
from soloq_lift_tables import (
    PRO_ROLE_TO_SOLOQ, PRO_TO_SOLOQ, SYN_VECTORS, load_tables,
)

N_BOOT = 10_000
SOLOQ_START = pd.Timestamp("2026-06-08")
RNG = np.random.default_rng(20260725)


def side_role_champs() -> pd.DataFrame:
    """(gameid, side) -> {pro role: champion}; all sides verified clean."""
    players, _ = load_games([2024, 2025, 2026])
    rc = players.groupby(["gameid", "side"]).apply(
        lambda g: dict(zip(g.position, g.champion)), include_groups=False,
    ).rename("roles").reset_index()
    assert rc.roles.map(len).eq(5).all(), "side without 5 distinct roles"
    return rc.pivot(index="gameid", columns="side", values="roles")


def soloq_features(
    games: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    t = load_tables()
    roles_by_game = side_role_champs()
    champ_wr, role_wr, ctr = t["champ_wr"], t["role_wr"], t["ctr"]
    syn = t["syn"]
    soloq_to_pro = {v: k for k, v in PRO_ROLE_TO_SOLOQ.items()}

    def wr(pro_name: str, pro_role: str) -> float:
        sq = PRO_TO_SOLOQ.get(pro_name, pro_name)
        if sq not in champ_wr:
            return 0.5  # champ unseen in soloq — neutral
        return role_wr.get(f"{sq}|{PRO_ROLE_TO_SOLOQ[pro_role]}", champ_wr[sq])

    def ctr_lift(blue_name: str, red_name: str, pro_role: str) -> float:
        a = PRO_TO_SOLOQ.get(blue_name, blue_name)
        b = PRO_TO_SOLOQ.get(red_name, red_name)
        r = PRO_ROLE_TO_SOLOQ[pro_role]
        lo, hi = (a, b) if a < b else (b, a)
        lift = ctr.get(f"{r}|{lo}|{hi}", 0.0)
        return lift if lo == a else -lift

    seen = {c for _, row in roles_by_game.iterrows()
            for side in ("Blue", "Red") for c in row[side].values()}
    misses = sorted(c for c in seen
                    if PRO_TO_SOLOQ.get(c, c) not in champ_wr)

    def syn_lift(side: dict) -> float:
        total = 0.0
        for ra, rb in SYN_VECTORS:
            a = side[soloq_to_pro[ra]]
            b = side[soloq_to_pro[rb]]
            key = (f"{ra}+{rb}|{PRO_TO_SOLOQ.get(a, a)}"
                   f"|{PRO_TO_SOLOQ.get(b, b)}")
            total += syn.get(key, 0.0)
        return total

    ctr_x = np.zeros(len(games))
    rwr_x = np.zeros(len(games))
    syn_x = np.zeros(len(games))
    for i, g in enumerate(games.itertuples()):
        blue = roles_by_game.loc[g.gameid, "Blue"]
        red = roles_by_game.loc[g.gameid, "Red"]
        ctr_x[i] = sum(ctr_lift(blue[r], red[r], r) for r in blue)
        rwr_x[i] = (sum(wr(c, r) for r, c in blue.items())
                    - sum(wr(c, r) for r, c in red.items()))
        syn_x[i] = syn_lift(blue) - syn_lift(red)
    coverage = {"pro_champs_in_games": len(seen),
                "matched": len(seen) - len(misses), "misses": misses}
    return ctr_x, rwr_x, syn_x, coverage


def sweep_refit_score(x, y, tr, va, trva, ho, name):
    """v0.9 protocol: C on val (fit train), refit train+val, holdout probs."""
    scored = {}
    for c in CS:
        m = fit_logistic(x[tr], y[tr], c)
        scored[c] = metrics(y[va], m.predict_proba(x[va])[:, 1])
    best_c = min(scored, key=lambda c: scored[c]["log_loss"])
    print(f"  {name}: val by C "
          f"{ {c: v['log_loss'] for c, v in scored.items()} } -> C={best_c}")
    m = fit_logistic(x[trva], y[trva], best_c)
    p_ho = m.predict_proba(x[ho])[:, 1]
    return best_c, scored[best_c], p_ho, m


def pergame_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def paired_bootstrap(d: np.ndarray) -> dict:
    """d = ll_base - ll_challenger per game; positive mean = challenger better."""
    n = len(d)
    means = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return {
        "n_games": n,
        "mean_delta": round(float(d.mean()), 6),
        "ci95": [round(float(np.quantile(means, 0.025)), 6),
                 round(float(np.quantile(means, 0.975)), 6)],
        "p_challenger_better": round(float((means > 0).mean()), 4),
    }


def main() -> None:
    t0 = time.time()
    games = build_game_table()
    y = games.blue_win.to_numpy(dtype=float)
    is_frozen = (games.league == "EWC") & (games.date >= "2026-07-01")
    tr = (games.date < TRAIN_END).to_numpy()
    va = ((games.date >= TRAIN_END) & (games.date < VAL_END)).to_numpy()
    ho = ((games.date >= VAL_END) & ~is_frozen).to_numpy()
    trva = tr | va
    print(f"{len(games)} games; train {tr.sum()} / val {va.sum()} / "
          f"holdout {ho.sum()} (frozen EWC excluded: {is_frozen.sum()})")

    stored = json.loads(
        (DATA_PROCESSED / "outcome_baseline_v09.json").read_text())
    elo_k = stored["elo_k"]
    elo = elo_diffs(games, elo_k)
    vocab = {c: i for i, c in enumerate(sorted(
        {c for picks in games.loc[trva, ["blue_picks", "red_picks"]]
         .to_numpy().ravel() for c in picks}))}
    champs = champ_matrix(games, vocab)
    print(f"elo K={elo_k} (stored v0.9 choice), {len(vocab)} champs in vocab")

    ctr_x, rwr_x, syn_x, coverage = soloq_features(games)
    print(f"bridge coverage: {coverage['matched']}/"
          f"{coverage['pro_champs_in_games']} (misses: {coverage['misses']})")
    # Standardize the new scalars on train statistics.
    scalars, z = {}, {}
    for nm, x in (("CTR", ctr_x), ("ROLE_WR", rwr_x), ("SYN", syn_x)):
        mu, sd = x[tr].mean(), x[tr].std()
        scalars[nm] = {"train_mean": round(float(mu), 6),
                       "train_sd": round(float(sd), 6)}
        assert sd > 0, f"{nm} degenerate on train"
        z[nm] = (x - mu) / sd

    designs = {
        "M2a": np.hstack([elo[:, None], champs]),
        "M2a_soloq": np.hstack([elo[:, None], champs,
                                z["CTR"][:, None], z["ROLE_WR"][:, None]]),
        "M2a_soloq_syn": np.hstack(
            [elo[:, None], champs, z["CTR"][:, None],
             z["ROLE_WR"][:, None], z["SYN"][:, None]]),
    }
    scalar_names = {"M2a_soloq": ["CTR", "ROLE_WR"],
                    "M2a_soloq_syn": ["CTR", "ROLE_WR", "SYN"]}
    results = {}
    probs = {}
    for name, x in designs.items():
        best_c, val_m, p_ho, model = sweep_refit_score(
            x, y, tr, va, trva, ho, name)
        probs[name] = p_ho
        results[name] = {"C": best_c, "val": val_m,
                         "holdout": metrics(y[ho], p_ho)}
        if name in scalar_names:
            k = len(scalar_names[name])
            results[name]["soloq_coefs"] = {
                nm: round(float(c), 5) for nm, c in
                zip(scalar_names[name], model.coef_[0][-k:])}
        print(f"  {name}: holdout {results[name]['holdout']}")

    y_ho = y[ho]
    deltas = {
        "M2a_vs_soloq": ("M2a", "M2a_soloq"),
        "M2a_vs_soloq_syn": ("M2a", "M2a_soloq_syn"),
        "syn_increment": ("M2a_soloq", "M2a_soloq_syn"),
    }
    gates, d_primary = {}, {}
    for tag, (base, chal) in deltas.items():
        d = pergame_ll(y_ho, probs[base]) - pergame_ll(y_ho, probs[chal])
        d_primary[tag] = d
        gates[tag] = paired_bootstrap(d)
        gates[tag]["pass"] = gates[tag]["ci95"][0] > 0
        print(f"gate {tag}: {gates[tag]}")
    go = gates["M2a_vs_soloq"]["pass"] or gates["M2a_vs_soloq_syn"]["pass"]
    gate = {"per_comparison": gates,
            "criterion": "either soloq variant beats M2a on holdout LL, "
                         "95% CI excluding 0"}

    # Secondary slice (reported, not gated): holdout games on/after the start
    # of soloq coverage — where transfer is temporally aligned.
    late = (games.loc[ho, "date"] >= SOLOQ_START).to_numpy()
    slice_gate = {tag: paired_bootstrap(d[late])
                  for tag, d in d_primary.items()}
    slice_metrics = {name: metrics(y_ho[late], p[late])
                     for name, p in probs.items()}
    print(f"secondary slice (holdout >= {SOLOQ_START.date()}, "
          f"{late.sum()} games): {slice_gate['M2a_vs_soloq']}")

    m2a_repro = results["M2a"]["holdout"]["log_loss"]
    m2a_stored = stored["holdout"]["M2a_elo_champs"]["log_loss"]
    out = {
        "experiment": "rung2_arm2_outcome",
        "spec": "docs/2026-07-25-rung2-transfer-spec.md",
        "question": ("do soloq lane-counter and role-strength scalars improve "
                     "completed-draft win prediction over M2a (Elo + champ "
                     "indicators)?"),
        "gate": {**gate, "pass": bool(go)},
        "rung1c": ("PASS (docs/2026-07-25-synergy-rung1c-results.md) — SYN "
                   "scalar included as the optional third model per the "
                   "rung 2 spec's PASS branch; soloq-side effect is ~1/10 "
                   "of the counter effect, expectations set accordingly"),
        "models": results,
        "secondary_slice_post_soloq_start": {
            "note": "holdout restricted to games on/after 2026-06-08; "
                    "reported, not gated",
            "bootstrap": slice_gate,
            "metrics": slice_metrics,
        },
        "m2a_reproduction": {
            "stored_holdout_ll": m2a_stored,
            "reproduced_holdout_ll": m2a_repro,
            "note": "raw CSVs may have grown since v0.9 ran; gate is "
                    "internally paired on identical data either way",
        },
        "timing_caveat": (
            "MANDATORY READING (pre-registered): soloq lifts are measured on "
            "patches 16.11-16.14 (2026-06-08 onward, table cut 2026-07-01); "
            "the v0.9 holdout runs from 2026-05-15. Features are patch-blind "
            "(the table pools patches), so early-holdout games carry real "
            "patch misalignment. A null on the primary gate with a positive "
            "post-06-08 slice reads 'transfer works where the data overlaps' "
            "and the follow-up is a re-split, not a NO-GO. A null on both is "
            "a misaligned-window null for the pre-06-08 portion but a real "
            "falsification for the aligned slice."),
        "provenance": {
            "splits": {"train_end": str(TRAIN_END.date()),
                       "val_end": str(VAL_END.date()),
                       "n_train": int(tr.sum()), "n_val": int(va.sum()),
                       "n_holdout": int(ho.sum()),
                       "n_holdout_post_soloq_start": int(late.sum()),
                       "frozen_ewc_excluded": int(is_frozen.sum())},
            "elo_k": elo_k,
            "c_grid": CS,
            "n_bootstrap": N_BOOT,
            "soloq_table": {k: json.loads(
                (DATA_PROCESSED / "soloq_lift_tables.json").read_text())[k]
                for k in ("cutoff_iso", "clean_games", "blue_win_rate",
                          "patch_mix", "n_champs", "n_counter_pairs",
                          "n_syn_pairs")},
            "loo_note": ("no leave-one-out correction anywhere: every pro "
                         "game scored is disjoint from the soloq games the "
                         "tables were built from (1c's LOO repair applies "
                         "only inside the table sample)"),
            "bridge_coverage": coverage,
            "scalar_standardization": scalars,
            "runtime_seconds": round(time.time() - t0),
        },
    }
    path = DATA_PROCESSED / "rung2_outcome.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({round(time.time() - t0)}s)")


if __name__ == "__main__":
    main()
