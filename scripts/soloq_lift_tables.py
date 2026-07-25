"""Shared soloq lift tables for rung 2 (docs/2026-07-25-rung2-transfer-spec.md).

Builds ONE static pair of tables from soloq games with game_creation strictly
before the pro val window start (2026-07-01 00:00 UTC — the v0.7 GBM's val
window opens 2026-07-01, cutoff 2026-07-15 minus 14 days), used by both rung-2
arms per the spec's look-ahead rule:

  - soloq_wr(c@r): champ@role WR, EB-shrunk toward the champ's overall WR with
    a 200-game prior (identical construction to rung 1b's D table).
  - soloq_ctr(a, b | r): same-role counter lift, antisymmetric —
    shrink_200( WR_obs(a beats b in r) − (wr(a@r) + 1 − wr(b@r))/2 ),
    unobserved pairs → 0 (identical construction to the rung 1c spec).
  - soloq_syn(a@ra, b@rb): same-team duo lift over the four priority vectors
    (rung 1c GO, docs/2026-07-25-synergy-rung1c-results.md) —
    shrink_200( WR_obs(pair) − mean(wr(a@ra), wr(b@rb)) ), unobserved → 0.
    NO leave-one-out here: 1c's LOO repair applies only when scoring games
    that are inside the table sample; every pro game rung 2 scores is
    disjoint from the soloq games the tables are built from.

Cleaning rules are rung 1b's verbatim (queue 420, duration ≥300, creation >0,
10 participants, no blank/dup roles per team). Output is cached to
data/processed/soloq_lift_tables.json; downstream consumers call
load_tables() / pro_arrays().

The PRO_TO_SOLOQ name bridge is copied from experiment_v09_soloq_transfer.py:46
(importing that module drags in torch); coverage is re-verified by pro_arrays.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from common import DATA_PROCESSED, DATA_RAW

DB = DATA_RAW / "soloq" / "soloq.db"
TABLE_PATH = DATA_PROCESSED / "soloq_lift_tables.json"
CUTOFF_MS = 1_782_864_000_000  # 2026-07-01 00:00 UTC, pro val window start
CUTOFF_ISO = "2026-07-01T00:00:00Z"
SOLOQ_COVERAGE_START = "2026-06-08"  # first soloq game in the db
SHRINK_GAMES = 200.0
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
SYN_VECTORS = (            # same-team priority vectors, rung 1b/1c
    ("BOTTOM", "UTILITY"),
    ("MIDDLE", "JUNGLE"),
    ("TOP", "JUNGLE"),
    ("JUNGLE", "UTILITY"),
)

# Pro (Oracle's Elixir) position labels -> soloq pos labels.
PRO_ROLE_TO_SOLOQ = {"top": "TOP", "jng": "JUNGLE", "mid": "MIDDLE",
                     "bot": "BOTTOM", "sup": "UTILITY"}

# 21 pro names differ from soloq internal names (copied from
# experiment_v09_soloq_transfer.py; 168/168 pro vocab covered 2026-07-23).
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


def load_clean_games_before(cutoff_ms: int = CUTOFF_MS) -> tuple[list[dict], dict]:
    """Rung 1b's loader/cleaning verbatim, restricted to creation < cutoff."""
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT payload, game_creation, game_version FROM matches "
        "WHERE state='done' AND payload IS NOT NULL "
        "AND queue_id=420 AND duration>=300 AND game_creation < ?",
        (cutoff_ms,),
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
                ok = False
                break
            team[pos] = p["champ"]
        if not ok or len(teams[100]) != 5 or len(teams[200]) != 5:
            drops["bad_roles"] += 1
            continue
        blue_win = next(p["win"] for p in parts if p["team"] == 100)
        games.append({
            "creation": creation, "version": version,
            "blue": teams[100], "red": teams[200],
            "blue_win": bool(blue_win),
        })
    return games, dict(drops)


def build_tables() -> dict:
    games, drops = load_clean_games_before()
    blue_wr = float(np.mean([g["blue_win"] for g in games]))
    assert abs(blue_wr - 0.4755) < 0.01, \
        f"blue WR {blue_wr:.4f} far from 1b anchor 0.4755 — team-id bug?"

    champ_w = defaultdict(lambda: [0, 0])
    combo_w = defaultdict(lambda: [0, 0])
    ctr_w = defaultdict(lambda: [0, 0])  # (role, a, b) a<b: [wins_of_a, games]
    duo_w = defaultdict(lambda: [0, 0])  # (ra, rb, a@ra, b@rb): [wins, games]
    for g in games:
        for side, won in ((g["blue"], g["blue_win"]),
                          (g["red"], not g["blue_win"])):
            for role, champ in side.items():
                champ_w[champ][0] += won
                champ_w[champ][1] += 1
                combo_w[(champ, role)][0] += won
                combo_w[(champ, role)][1] += 1
            for ra, rb in SYN_VECTORS:
                s = duo_w[(ra, rb, side[ra], side[rb])]
                s[0] += won
                s[1] += 1
        for role in ROLES:
            a, b = g["blue"][role], g["red"][role]
            if a == b:
                continue  # mirror matchup carries no counter information
            lo, hi = (a, b) if a < b else (b, a)
            s = ctr_w[(role, lo, hi)]
            s[0] += g["blue_win"] if lo == a else (not g["blue_win"])
            s[1] += 1

    champ_wr = {c: w / n for c, (w, n) in champ_w.items()}

    def role_wr(champ, role):
        w, n = combo_w.get((champ, role), (0, 0))
        return (w + champ_wr[champ] * SHRINK_GAMES) / (n + SHRINK_GAMES)

    role_wr_table = {f"{c}|{r}": round(role_wr(c, r), 6)
                     for (c, r) in combo_w}
    ctr_table = {}
    for (role, lo, hi), (w, n) in ctr_w.items():
        expected = (role_wr(lo, role) + 1.0 - role_wr(hi, role)) / 2.0
        lift = (w / n - expected) * n / (n + SHRINK_GAMES)
        ctr_table[f"{role}|{lo}|{hi}"] = round(float(lift), 6)

    # Same-team duo lifts (rung 1c construction, full counts — no LOO: rung 2
    # only scores pro games, which are disjoint from this table's sample).
    syn_table = {}
    for (ra, rb, a, b), (w, n) in duo_w.items():
        expected = (role_wr(a, ra) + role_wr(b, rb)) / 2.0
        lift = (w / n - expected) * n / (n + SHRINK_GAMES)
        syn_table[f"{ra}+{rb}|{a}|{b}"] = round(float(lift), 6)

    versions = Counter(".".join(str(g["version"]).split(".")[:2]) for g in games)
    out = {
        "cutoff_ms": CUTOFF_MS,
        "cutoff_iso": CUTOFF_ISO,
        "soloq_coverage_start": SOLOQ_COVERAGE_START,
        "clean_games": len(games),
        "dropped": drops,
        "blue_win_rate": round(blue_wr, 4),
        "patch_mix": dict(versions),
        "shrink_games": SHRINK_GAMES,
        "n_champs": len(champ_wr),
        "n_champ_role_combos": len(role_wr_table),
        "n_counter_pairs": len(ctr_table),
        "n_syn_pairs": len(syn_table),
        "champ_wr": {c: round(v, 6) for c, v in champ_wr.items()},
        "role_wr": role_wr_table,
        "ctr": ctr_table,
        "syn": syn_table,
    }
    TABLE_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {TABLE_PATH}: {len(games)} games pre-{CUTOFF_ISO}, "
          f"{len(champ_wr)} champs, {len(role_wr_table)} champ@role combos, "
          f"{len(ctr_table)} counter pairs, {len(syn_table)} duo pairs "
          f"(drops: {drops})")
    return out


def load_tables() -> dict:
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"{TABLE_PATH} missing — run scripts/soloq_lift_tables.py first")
    return json.loads(TABLE_PATH.read_text())


def pro_arrays(pro_champs: list[str], tables: dict | None = None,
               pro_roles: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup"),
               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dense lookup arrays over a pro champion index space.

    Returns (W, C, S, coverage):
      W[i, r]    = soloq_wr(champ_i @ role_r), champ@role unobserved -> the
                   champ's overall soloq WR (the shrunk value at n=0), champ
                   entirely unseen in soloq -> 0.5.
      C[r, i, j] = soloq_ctr(champ_i, champ_j | role_r), antisymmetric,
                   unobserved -> 0.
      S[v, i, j] = soloq_syn(champ_i @ ra, champ_j @ rb) for priority vector
                   v = (ra, rb) in SYN_VECTORS order (slot order matters),
                   unobserved -> 0.
    """
    t = tables or load_tables()
    soloq_roles = [PRO_ROLE_TO_SOLOQ[r] for r in pro_roles]
    n = len(pro_champs)
    W = np.full((n, len(soloq_roles)), 0.5, dtype=np.float64)
    C = np.zeros((len(soloq_roles), n, n), dtype=np.float64)
    S = np.zeros((len(SYN_VECTORS), n, n), dtype=np.float64)
    v_idx = {f"{ra}+{rb}": k for k, (ra, rb) in enumerate(SYN_VECTORS)}
    bridged, misses = {}, []
    for i, name in enumerate(pro_champs):
        sq = PRO_TO_SOLOQ.get(name, name)
        if sq not in t["champ_wr"]:
            misses.append(name)
            continue
        bridged[sq] = i
        for r_i, r in enumerate(soloq_roles):
            W[i, r_i] = t["role_wr"].get(f"{sq}|{r}", t["champ_wr"][sq])
    for key, lift in t["ctr"].items():
        role, lo, hi = key.split("|")
        i, j = bridged.get(lo), bridged.get(hi)
        if i is None or j is None or role not in soloq_roles:
            continue
        r_i = soloq_roles.index(role)
        C[r_i, i, j] = lift
        C[r_i, j, i] = -lift
    for key, lift in t.get("syn", {}).items():
        vec, a, b = key.split("|")
        i, j = bridged.get(a), bridged.get(b)
        if i is None or j is None:
            continue
        S[v_idx[vec], i, j] = lift
    coverage = {"pro_champs": n, "matched": n - len(misses), "misses": misses}
    return W, C, S, coverage


if __name__ == "__main__":
    build_tables()
