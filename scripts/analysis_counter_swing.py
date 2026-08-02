"""Counter-swing quantification: soloq lift tables -> readable win-probability swings.

Descriptive companion to the rung 1b/1c/2 ladder. Those experiments measured the
draft-composition effects in per-game log-loss (rung 1c: counters +0.000624,
duo synergy +0.000061) — numbers no reader can feel. This script translates the
same tables into the units a reader already has: "a draft in the top decile of
counter advantage wins X% of the time, one in the bottom decile wins Y%."

No model training. The only fits are low-dimensional logistic regressions used to
read implied swings off the same scalars; everything else is bin counting.
Deterministic: no sampling anywhere, analytic binomial SEs, one pass over a
date-pinned window.

THE EPISTEMIC TRAP (rung 1c's first run died on it): the lift tables are
outcome-derived, so scoring a game that is INSIDE the table's own sample leaks
that game's result into its own lift and inflates every gradient. Rung 1c repaired
this with leave-one-out counts. This script takes the cleaner route the spec
prefers: it scores ONLY games created at/after the tables' cutoff
(2026-07-01 00:00 UTC), which are disjoint from the table sample by construction,
so full-count lifts are correct and no LOO is needed. The disjointness is asserted
against the table file's own recorded cutoff, not assumed.

Window: [2026-07-01 00:00 UTC, 2026-07-25 00:00 UTC). The upper bound is pinned
(rather than "whatever is in the db") because the soloq scrape has resumed and is
appending live; the pin keeps this analysis reproducible and drops the thin
partially-scraped tail after 07-24. The db is opened strictly read-only.

Anchors, so every swing has scale:
  - side selection: the blue-side handicap in this very sample;
  - Elo-equivalent: the team-strength gap that produces the same expected-score
    spread under the standard 400-point Elo curve (the curve v0.9's online Elo
    uses); v0.9's stored holdout metrics are quoted from its JSON for reference.

Writes data/processed/counter_swing.json.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA_PROCESSED, DATA_RAW
from soloq_lift_tables import ROLES, SYN_VECTORS, load_tables

DB = DATA_RAW / "soloq" / "soloq.db"
OUT_PATH = DATA_PROCESSED / "counter_swing.json"

WINDOW_START_MS = 1_782_864_000_000  # 2026-07-01 00:00 UTC — the tables' cutoff
WINDOW_END_MS = 1_784_937_600_000    # 2026-07-25 00:00 UTC — pinned, see docstring
WINDOW_START_ISO = "2026-07-01T00:00:00Z"
WINDOW_END_ISO = "2026-07-25T00:00:00Z"
MIN_GAMES = 150_000                  # tripwire: the window should carry ~190k
FIT_FRAC = 0.70                      # chronological fit/eval split for fitted numbers
SCALARS = ("CTR", "SYN", "ROLE")
# (name, low percentile, high percentile). p75/p25 is the "typical game" band —
# the tails make the loudest headline, the quartiles make the honest one.
BANDS = (("p90_vs_p10", 10, 90), ("p99_vs_p1", 1, 99), ("p75_vs_p25", 25, 75))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_window_games() -> tuple[list[dict], dict]:
    """Rung 1b's cleaning verbatim (queue 420, duration >= 300, creation > 0,
    10 participants, no blank/dup roles per team), restricted to the pinned
    window. Read-only connection: a parallel scrape owns this db."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    rows = con.execute(
        "SELECT payload, game_creation, game_version, region FROM matches "
        "WHERE state='done' AND payload IS NOT NULL "
        "AND queue_id=420 AND duration>=300 "
        "AND game_creation >= ? AND game_creation < ?",
        (WINDOW_START_MS, WINDOW_END_MS),
    ).fetchall()
    con.close()
    games, drops = [], Counter()
    for payload, creation, version, region in rows:
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
                ok = False
                break
            team[pos] = p["champ"]
        if not ok or len(teams[100]) != 5 or len(teams[200]) != 5:
            drops["bad_roles"] += 1
            continue
        blue_win = next(p["win"] for p in parts if p["team"] == 100)
        games.append({
            "creation": creation, "version": version, "region": region,
            "blue": teams[100], "red": teams[200], "blue_win": bool(blue_win),
        })
    games.sort(key=lambda g: g["creation"])
    return games, dict(drops)


def score_games(games: list[dict], t: dict) -> tuple[dict, dict]:
    """Per-game blue-minus-red scalars from the static (pre-cutoff) lift tables.

    CTR  = sum over the 5 roles of the antisymmetric same-role counter lift.
    SYN  = sum over the 4 priority vectors of the slot-ordered duo lift,
           blue side minus red side.
    ROLE = sum over the 5 roles of champ@role WR, blue minus red (champ@role
           unseen -> the champ's overall soloq WR; champ unseen -> 0.5).

    Construction is identical to experiment_v10_synergy_rung1c.game_scalars,
    minus the leave-one-out branch — see the module docstring for why LOO is
    neither needed nor correct here.
    """
    ctr_tbl, syn_tbl, role_wr, champ_wr = t["ctr"], t["syn"], t["role_wr"], t["champ_wr"]
    n = len(games)
    out = {k: np.zeros(n) for k in SCALARS}
    cov = Counter()
    unseen_champs = set()

    def wr(champ: str, role: str) -> float:
        v = role_wr.get(f"{champ}|{role}")
        if v is not None:
            cov["role_wr_hit"] += 1
            return v
        v = champ_wr.get(champ)
        if v is not None:
            cov["role_wr_champ_fallback"] += 1
            return v
        cov["role_wr_unseen_champ"] += 1
        unseen_champs.add(champ)
        return 0.5

    for i, g in enumerate(games):
        blue, red = g["blue"], g["red"]
        c = 0.0
        r = 0.0
        for role in ROLES:
            a, b = blue[role], red[role]
            r += wr(a, role) - wr(b, role)
            if a == b:
                cov["mirror_matchup"] += 1
                continue  # mirror carries no counter information (table skips it)
            lo, hi = (a, b) if a < b else (b, a)
            lift = ctr_tbl.get(f"{role}|{lo}|{hi}")
            if lift is None:
                cov["ctr_miss"] += 1
                continue
            cov["ctr_hit"] += 1
            c += lift if lo == a else -lift
        s = 0.0
        for ra, rb in SYN_VECTORS:
            for side, sign in ((blue, 1.0), (red, -1.0)):
                lift = syn_tbl.get(f"{ra}+{rb}|{side[ra]}|{side[rb]}")
                if lift is None:
                    cov["syn_miss"] += 1
                    continue
                cov["syn_hit"] += 1
                s += sign * lift
        out["CTR"][i], out["SYN"][i], out["ROLE"][i] = c, s, r

    coverage = {k: int(v) for k, v in cov.items()}
    coverage["unseen_champs"] = sorted(unseen_champs)
    for k in SCALARS:
        coverage[f"{k}_nonzero_frac"] = round(float((out[k] != 0).mean()), 4)
    return out, coverage


# --------------------------------------------------------------------------
# descriptive machinery
# --------------------------------------------------------------------------

def wr_band(score: np.ndarray, y: np.ndarray, lo_pct: float, hi_pct: float) -> dict:
    """Empirical blue WR in the <=lo_pct and >=hi_pct tails, and the raw gap.

    Pure counting — this is the number that owes nothing to any model.
    """
    lo_cut = float(np.percentile(score, lo_pct))
    hi_cut = float(np.percentile(score, hi_pct))
    lo_m, hi_m = score <= lo_cut, score >= hi_cut
    p_lo, p_hi = float(y[lo_m].mean()), float(y[hi_m].mean())
    n_lo, n_hi = int(lo_m.sum()), int(hi_m.sum())
    se = math.sqrt(p_lo * (1 - p_lo) / n_lo + p_hi * (1 - p_hi) / n_hi)
    gap = p_hi - p_lo
    return {
        "low_pct": lo_pct, "high_pct": hi_pct,
        "low_cut": round(lo_cut, 6), "high_cut": round(hi_cut, 6),
        # the band MEAN sits beyond its cut (it averages the whole tail); a fit
        # evaluated at the cut is therefore not comparable to this gap without
        # being re-evaluated at these means — see implied_at().
        "low_band_mean": round(float(score[lo_m].mean()), 6),
        "high_band_mean": round(float(score[hi_m].mean()), 6),
        "n_low": n_lo, "n_high": n_hi,
        "wr_low": round(p_lo, 5), "wr_high": round(p_hi, 5),
        "gap_wr_points": round(gap * 100, 3),
        "gap_se_points": round(se * 100, 3),
        "gap_ci95_points": [round((gap - 1.96 * se) * 100, 3),
                            round((gap + 1.96 * se) * 100, 3)],
    }


def decile_table(score: np.ndarray, y: np.ndarray, p_ref: np.ndarray | None) -> list[dict]:
    """Equal-count deciles of a score: actual WR, and (optionally) the WR a
    reference model predicts for the same games — the residual column is what
    separates 'this score is informative' from 'this score proxies strength'."""
    order = np.argsort(score, kind="stable")
    rows = []
    for d, idx in enumerate(np.array_split(order, 10), start=1):
        row = {
            "decile": d, "n": int(len(idx)),
            "score_mean": round(float(score[idx].mean()), 6),
            "actual_blue_wr": round(float(y[idx].mean()), 4),
        }
        if p_ref is not None:
            row["ref_predicted_wr"] = round(float(p_ref[idx].mean()), 4)
            row["residual_wr_points"] = round(
                float(y[idx].mean() - p_ref[idx].mean()) * 100, 2)
        rows.append(row)
    return rows


def fit_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Effectively unpenalised: 1-3 features over ~135k games."""
    return LogisticRegression(C=1e6, max_iter=2000).fit(X, y)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def implied_swing(model: LogisticRegression, score: np.ndarray, mu: float,
                  sd: float, lo_pct: float, hi_pct: float, col: int = 0,
                  n_features: int = 1) -> dict:
    """Logistic-implied WR at two percentiles of one scalar, others held at their
    means (z = 0). With one feature this is the marginal swing; inside the joint
    model it is the swing that survives conditioning on the other scalars."""
    b = float(model.coef_[0][col])
    a = float(model.intercept_[0])
    lo_v = float(np.percentile(score, lo_pct))
    hi_v = float(np.percentile(score, hi_pct))
    p_lo = sigmoid(a + b * (lo_v - mu) / sd)
    p_hi = sigmoid(a + b * (hi_v - mu) / sd)
    return {
        "low_pct": lo_pct, "high_pct": hi_pct,
        "coef_per_sd": round(b, 5),
        "wr_low": round(p_lo, 5), "wr_high": round(p_hi, 5),
        "gap_wr_points": round((p_hi - p_lo) * 100, 3),
        "conditional_on": ("nothing (single-scalar fit)" if n_features == 1
                           else "the other scalars held at their sample means"),
    }


def implied_at(model: LogisticRegression, lo_v: float, hi_v: float, mu: float,
               sd: float, col: int = 0) -> dict:
    """Logistic-implied WR at two explicit scalar values, others at their means.

    Used to re-evaluate a fit at the raw bands' own mean scores, which is the
    only apples-to-apples comparison against wr_band()'s counted gap.
    """
    b, a = float(model.coef_[0][col]), float(model.intercept_[0])
    p_lo, p_hi = sigmoid(a + b * (lo_v - mu) / sd), sigmoid(a + b * (hi_v - mu) / sd)
    return {
        "at_low_score": round(lo_v, 6), "at_high_score": round(hi_v, 6),
        "wr_low": round(p_lo, 5), "wr_high": round(p_hi, 5),
        "gap_wr_points": round((p_hi - p_lo) * 100, 3),
    }


def elo_equivalent(gap_points: float) -> dict:
    """The team-strength gap that reproduces the same expected-score spread.

    A gap of G WR points between two groups is split symmetrically about even:
    the favourite scores 0.5 + G/200. Under the standard Elo curve
    (expected = 1/(1+10^(-D/400))) that expected score comes from a rating edge
    of D = 400 * log10(p/(1-p)) against an even opponent. Same curve family the
    v0.9 outcome baseline fits its online Elo on (K=10).
    """
    p = 0.5 + gap_points / 200.0
    p = min(max(p, 1e-6), 1 - 1e-6)
    return {
        "gap_wr_points": round(gap_points, 3),
        "favourite_expected_score": round(p, 5),
        "elo_points_vs_even_opponent": round(400 * math.log10(p / (1 - p)), 1),
    }


# --------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    tables = load_tables()
    assert tables["cutoff_ms"] == WINDOW_START_MS, (
        f"lift tables were built to {tables['cutoff_iso']} but this analysis scores "
        f"from {WINDOW_START_ISO} — the disjointness that removes the need for "
        f"leave-one-out no longer holds. Rebuild the tables or move the window.")
    print(f"tables: {tables['clean_games']} games before {tables['cutoff_iso']}, "
          f"{tables['n_counter_pairs']} counter pairs, {tables['n_syn_pairs']} duo pairs")

    games, drops = load_window_games()
    n = len(games)
    print(f"{n} clean games in [{WINDOW_START_ISO}, {WINDOW_END_ISO}) (drops: {drops})")
    assert n >= MIN_GAMES, f"only {n} clean games in window (expected >= {MIN_GAMES})"
    assert games[0]["creation"] >= tables["cutoff_ms"], "window leaks into the table sample"

    y = np.array([g["blue_win"] for g in games], dtype=np.float64)
    blue_wr = float(y.mean())
    print(f"blue WR {blue_wr:.4f} (tables' pre-cutoff sample: {tables['blue_win_rate']})")
    assert abs(blue_wr - 0.4754) < 0.02, f"blue WR {blue_wr:.4f} — team-id bug?"

    raw, coverage = score_games(games, tables)
    print(f"coverage: { {k: v for k, v in coverage.items() if isinstance(v, int)} }")

    # --- chronological fit/eval split: every FITTED number is reported out of
    # sample, so nothing in this doc is a 3-parameter model reading its own fit.
    i_fit = int(n * FIT_FRAC)
    ev = slice(i_fit, n)
    y_ev = y[ev]
    stats = {}
    z = {}
    for k in SCALARS:
        mu, sd = float(raw[k][:i_fit].mean()), float(raw[k][:i_fit].std())
        assert sd > 0, f"{k} has zero variance"
        z[k] = (raw[k] - mu) / sd
        stats[k] = {
            "fit_mean": round(mu, 6), "fit_sd": round(sd, 6),
            "p1": round(float(np.percentile(raw[k], 1)), 6),
            "p10": round(float(np.percentile(raw[k], 10)), 6),
            "p90": round(float(np.percentile(raw[k], 90)), 6),
            "p99": round(float(np.percentile(raw[k], 99)), 6),
            "auc_full_window": round(float(roc_auc_score(y, raw[k])), 4),
        }

    # --- headline: raw binned win rates (no model anywhere) ---
    bands = {k: {name: wr_band(raw[k], y, lo, hi) for name, lo, hi in BANDS}
             for k in SCALARS}
    for k in SCALARS:
        b = bands[k]["p90_vs_p10"]
        print(f"  {k}: p90 {b['wr_high']:.4f} vs p10 {b['wr_low']:.4f} "
              f"-> {b['gap_wr_points']:+.2f} pts")

    # --- logistic-implied swings: single-scalar and joint (fit on the fit slice) ---
    singles, single_models, matched = {}, {}, {}
    for k in SCALARS:
        m = fit_logit(z[k][:i_fit, None], y[:i_fit])
        single_models[k] = m
        mu, sd = stats[k]["fit_mean"], stats[k]["fit_sd"]
        singles[k] = {name: implied_swing(m, raw[k], mu, sd, lo, hi)
                      for name, lo, hi in BANDS}
        # like-for-like: the same fit re-read at the raw bands' own mean scores
        matched[k] = {
            name: {
                "raw_counted_gap_points": bands[k][name]["gap_wr_points"],
                "fit_at_same_scores": implied_at(
                    m, bands[k][name]["low_band_mean"],
                    bands[k][name]["high_band_mean"], mu, sd),
            } for name, _, _ in BANDS
        }

    X = np.column_stack([z[k] for k in SCALARS])
    joint = fit_logit(X[:i_fit], y[:i_fit])
    joint_swings = {}
    for col, k in enumerate(SCALARS):
        mu, sd = stats[k]["fit_mean"], stats[k]["fit_sd"]
        joint_swings[k] = {
            name: implied_swing(joint, raw[k], mu, sd, lo, hi, col, len(SCALARS))
            for name, lo, hi in BANDS
        }
        joint_swings[k]["at_raw_band_means"] = {
            name: implied_at(joint, bands[k][name]["low_band_mean"],
                             bands[k][name]["high_band_mean"], mu, sd, col)
            for name, _, _ in BANDS
        }

    # --- the combined draft score: the whole-composition number for the piece.
    # Fitted on the fit slice, every reported bin is from the held-out eval slice.
    p_joint_ev = joint.predict_proba(X[ev])[:, 1]
    combined_ev = joint.decision_function(X[ev])
    combined = {
        "definition": ("logit of the 3-scalar logistic (CTR + SYN + ROLE), fitted on "
                       "the first 70% of the window and evaluated on the last 30%"),
        "n_eval": int(n - i_fit),
        "coef_per_sd": {k: round(float(joint.coef_[0][i]), 5)
                        for i, k in enumerate(SCALARS)},
        "intercept": round(float(joint.intercept_[0]), 5),
        "auc_eval": round(float(roc_auc_score(y_ev, combined_ev)), 4),
        **{name: wr_band(combined_ev, y_ev, lo, hi) for name, lo, hi in BANDS},
        "deciles": decile_table(combined_ev, y_ev, p_joint_ev),
    }
    print(f"  COMBINED (eval slice): p90 vs p10 "
          f"{combined['p90_vs_p10']['gap_wr_points']:+.2f} pts, "
          f"AUC {combined['auc_eval']}")

    # --- does the counter swing survive knowing champ@role strength? ---
    role_only = fit_logit(z["ROLE"][:i_fit, None], y[:i_fit])
    p_role_ev = role_only.predict_proba(z["ROLE"][ev, None])[:, 1]
    deciles = {
        k: {
            "full_window_vs_nothing": decile_table(raw[k], y, None),
            "eval_slice_vs_role_model": decile_table(raw[k][ev], y_ev, p_role_ev),
        } for k in SCALARS
    }

    corr = np.corrcoef(np.column_stack([raw[k] for k in SCALARS]).T)
    correlations = {f"{a}~{b}": round(float(corr[i][j]), 4)
                    for i, a in enumerate(SCALARS) for j, b in enumerate(SCALARS) if i < j}

    # --- anchors ---
    side_gap_points = (1 - blue_wr) * 100 - blue_wr * 100
    v09 = json.loads((DATA_PROCESSED / "outcome_baseline_v09.json").read_text())
    anchors = {
        "side_selection": {
            "blue_win_rate_this_sample": round(blue_wr, 5),
            "blue_handicap_wr_points_below_even": round((0.5 - blue_wr) * 100, 3),
            "red_minus_blue_gap_wr_points": round(side_gap_points, 3),
            "note": ("the largest uncontested edge in the game: nobody drafts around "
                     "it because nobody can — side is assigned, not chosen"),
            "elo_equivalent": elo_equivalent(side_gap_points),
        },
        "team_strength_v09": {
            "source": "data/processed/outcome_baseline_v09.json (pro, 2024-26)",
            "elo_k": v09["elo_k"],
            "elo_only_holdout": v09["holdout"]["M1_elo"],
            "elo_plus_completed_draft_holdout": v09["holdout"]["M2a_elo_champs"],
            "base_rate_holdout": v09["holdout"]["M0_baserate"],
            "note": ("scale reference only. The Elo-equivalent conversions in this "
                     "file are analytic (standard 400-point curve); the pro Elo-gap "
                     "distribution was deliberately NOT recomputed here because the "
                     "frozen EWC July test set must not be loaded by this analysis."),
        },
        "elo_equivalents": {
            k: {name: elo_equivalent(bands[k][name]["gap_wr_points"])
                for name, _, _ in BANDS} for k in SCALARS
        },
        "elo_equivalents_role_conditioned": {
            k: {name: elo_equivalent(
                joint_swings[k]["at_raw_band_means"][name]["gap_wr_points"])
                for name, _, _ in BANDS} for k in SCALARS
        },
        "combined_elo_equivalent": {
            name: elo_equivalent(combined[name]["gap_wr_points"])
            for name, _, _ in BANDS
        },
        "counter_swing_vs_side_selection": {
            "raw_p90_vs_p10_multiple_of_side_gap": round(
                bands["CTR"]["p90_vs_p10"]["gap_wr_points"] / side_gap_points, 2),
            "role_conditioned_p90_vs_p10_multiple_of_side_gap": round(
                joint_swings["CTR"]["at_raw_band_means"]["p90_vs_p10"]["gap_wr_points"]
                / side_gap_points, 2),
            "combined_draft_p90_vs_p10_multiple_of_side_gap": round(
                combined["p90_vs_p10"]["gap_wr_points"] / side_gap_points, 2),
        },
    }

    patch_mix = Counter(".".join(str(g["version"]).split(".")[:2]) for g in games)
    region_mix = Counter(g["region"] for g in games)
    day_span = {
        "first_game": time.strftime("%Y-%m-%d %H:%M UTC",
                                    time.gmtime(games[0]["creation"] / 1000)),
        "last_game": time.strftime("%Y-%m-%d %H:%M UTC",
                                   time.gmtime(games[-1]["creation"] / 1000)),
        "fit_eval_boundary": time.strftime("%Y-%m-%d %H:%M UTC",
                                           time.gmtime(games[i_fit]["creation"] / 1000)),
    }

    out = {
        "analysis": "counter_swing",
        "question": ("how large is the draft-composition win-probability swing in "
                     "soloq, in units a reader can feel?"),
        "headline": {
            "counter_p90_vs_p10_raw_wr_points": bands["CTR"]["p90_vs_p10"]["gap_wr_points"],
            "counter_p99_vs_p1_raw_wr_points": bands["CTR"]["p99_vs_p1"]["gap_wr_points"],
            "counter_p90_vs_p10_role_conditioned_wr_points":
                joint_swings["CTR"]["at_raw_band_means"]["p90_vs_p10"]["gap_wr_points"],
            "combined_draft_p90_vs_p10_raw_wr_points": combined["p90_vs_p10"]["gap_wr_points"],
            "role_strength_p90_vs_p10_raw_wr_points": bands["ROLE"]["p90_vs_p10"]["gap_wr_points"],
            "synergy_p90_vs_p10_raw_wr_points": bands["SYN"]["p90_vs_p10"]["gap_wr_points"],
            "side_selection_anchor_wr_points": round(side_gap_points, 3),
        },
        "raw_win_rate_bands": bands,
        "raw_vs_fit_matched": matched,
        "logistic_implied_single_scalar": singles,
        "logistic_implied_joint": joint_swings,
        "combined_draft_score": combined,
        "decile_tables": deciles,
        "scalar_stats": stats,
        "scalar_correlations": correlations,
        "anchors": anchors,
        "provenance": {
            "lift_tables": {
                "path": "data/processed/soloq_lift_tables.json",
                "built_from_games_before": tables["cutoff_iso"],
                "clean_games": tables["clean_games"],
                "blue_win_rate": tables["blue_win_rate"],
                "patch_mix": tables["patch_mix"],
                "shrink_prior_games": tables["shrink_games"],
                "n_champs": tables["n_champs"],
                "n_champ_role_combos": tables["n_champ_role_combos"],
                "n_counter_pairs": tables["n_counter_pairs"],
                "n_syn_pairs": tables["n_syn_pairs"],
            },
            "scoring_window": {
                "start": WINDOW_START_ISO, "end": WINDOW_END_ISO,
                "clean_games": n, "dropped": drops,
                "blue_win_rate": round(blue_wr, 5),
                "patch_mix": dict(patch_mix), "region_mix": dict(region_mix),
                **day_span,
                "fit_games": i_fit, "eval_games": int(n - i_fit),
                "upper_bound_pinned_because": (
                    "the soloq scrape resumed 2026-07-25 and is appending live; "
                    "days after 07-24 are thinly covered. Pinning keeps re-runs "
                    "reproducible and the sample density flat."),
            },
            "leave_one_out": (
                "NOT APPLIED, AND NOT NEEDED: every scored game postdates the lift "
                "tables' 2026-07-01 cutoff, so no game contributes to its own lift. "
                "Asserted against the table file's recorded cutoff at run time. "
                "Rung 1c's LOO repair applies only to in-sample scoring."),
            "table_coverage": coverage,
            "population": ("soloq ranked solo/duo (queue 420), Master+ scrape, "
                           "regions na/kr/euw — NOT professional play"),
            "determinism": ("no random sampling; analytic binomial SEs; logistic "
                            "fits are deterministic (lbfgs, C=1e6)"),
            "frozen_ewc_test_set": "never loaded or scored by this script",
            "runtime_seconds": round(time.time() - t0),
        },
        "caveats": [
            "correlational, not causal: these are win rates conditional on the "
            "champions that got picked, not the effect of intervening on a draft",
            "soloq Master+ on patches 16.11-16.13; pro play has different champion "
            "pools, coordination, and a ban phase that removes the extremes",
            "the lift tables are a static pre-2026-07-01 snapshot applied to a later "
            "window — champion meta drift inside the window is unmodelled noise",
            "the counter/synergy tails are partly selection: an extreme differential "
            "usually means one side got a strong champ@role, which is why the "
            "role-conditioned columns (joint fit, residual deciles) matter more "
            "than the raw gaps for any causal reading",
            "no player-skill term exists in this data, and soloq outcomes are "
            "player-dominated (rung 1b's ceiling: AUC .551 with every comp feature)",
        ],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH} ({round(time.time() - t0)}s)")


if __name__ == "__main__":
    main()
