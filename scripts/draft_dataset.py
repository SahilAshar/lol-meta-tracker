"""Build the draft-decision dataset for the next-pick model.

Explodes each game into 20 ordered draft decisions (10 bans, 10 picks) and, for
each decision, emits one row per *available* candidate champion with point-in-time
features. Availability accounts for champions already picked/banned this game and,
in fearless leagues, champions used by either team earlier in the same series.

v0.5 adds role-constraint features: each champion gets a trailing-window role
distribution (share of games at top/jng/mid/bot/sup), a team's picks-so-far are
probabilistically assigned to roles by enumerating role permutations weighted by
those distributions (so flex picks like Poppy stay ambiguous until another pick
resolves them), and each candidate is scored on how well it fills the roles the
relevant team still has open. For bans the relevant team is the opponent, since
bans target what the opponent could pick next.

Draft order assumes the standard tournament sequence with blue banning/picking
first. NOTE: as of 2026 blue is no longer guaranteed to pick first; Oracle's
Elixir data does not record which team opened the draft, so sequence indices are
approximate for a minority of games. Team-level ban/pick ordinals are exact.

Output: data/processed/draft_decisions.parquet
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import DATA_PROCESSED, raw_csv_path

LEAGUES = ["LCK", "LPL", "LEC", "LCS", "MSI", "EWC", "FST"]

RATE_WINDOW_DAYS = 28  # global champion pick/ban rates
USAGE_WINDOW_DAYS = 56  # team-specific champion usage
ROLE_WINDOW_DAYS = 120  # per-champion role distributions
MIN_ROLE_GAMES = 5  # below this, back off to uniform over ever-played roles

ROLES = ["top", "jng", "mid", "bot", "sup"]

# (sequence index, decision type, side, per-team ordinal), standard tournament draft.
DRAFT_SEQUENCE = [
    (1, "ban", "Blue", 1), (2, "ban", "Red", 1), (3, "ban", "Blue", 2),
    (4, "ban", "Red", 2), (5, "ban", "Blue", 3), (6, "ban", "Red", 3),
    (7, "pick", "Blue", 1), (8, "pick", "Red", 1), (9, "pick", "Red", 2),
    (10, "pick", "Blue", 2), (11, "pick", "Blue", 3), (12, "pick", "Red", 3),
    (13, "ban", "Red", 4), (14, "ban", "Blue", 4), (15, "ban", "Red", 5),
    (16, "ban", "Blue", 5),
    (17, "pick", "Red", 4), (18, "pick", "Blue", 4), (19, "pick", "Blue", 5),
    (20, "pick", "Red", 5),
]


def load_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(raw_csv_path(), low_memory=False)
    df = df[df.league.isin(LEAGUES)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.normalize()

    # OE occasionally carries the same game under two gameids (e.g. HLE-JDG at
    # EWC 2026-07-16). Keep the first gameid per (timestamp, teams, champions).
    sig = df[df.position != "team"].groupby("gameid").agg(
        date=("date", "first"),
        champ_sig=("champion", frozenset),
        team_sig=("teamname", frozenset),
    )
    keep = sig.sort_index().drop_duplicates(
        subset=["date", "champ_sig", "team_sig"], keep="first"
    ).index
    dropped = sig.index.difference(keep)
    if len(dropped):
        print(f"dropping {len(dropped)} duplicate game record(s): {list(dropped)}")
        df = df[df.gameid.isin(keep)]

    players = df[df.position != "team"].copy()
    teams = df[df.position == "team"].copy()
    return players, teams


def detect_fearless(players: pd.DataFrame) -> set[str]:
    """A league is fearless if champions never repeat across games of a series."""
    fearless = set()
    for lg, lgdf in players.groupby("league"):
        game_info = lgdf.groupby("gameid").agg(
            day=("day", "first"),
            teams=("teamname", lambda s: tuple(sorted(s.unique()))),
        )
        champs = lgdf.groupby("gameid")["champion"].apply(set)
        overlaps = 0
        for _, gids in game_info.groupby(["day", "teams"]).groups.items():
            gids = list(gids)
            for i in range(len(gids)):
                for j in range(i + 1, len(gids)):
                    overlaps += len(champs[gids[i]] & champs[gids[j]])
        if overlaps == 0:
            fearless.add(lg)
    return fearless


class RollingRates:
    """Point-in-time champion rates and team usage, as of the day before a game."""

    def __init__(self, players: pd.DataFrame, teams: pd.DataFrame, champs: list[str]):
        self.champs = champs
        self.cidx = {c: i for i, c in enumerate(champs)}
        start = players.day.min()
        self.days = pd.date_range(start, players.day.max())
        self.didx = {d: i for i, d in enumerate(self.days)}
        n_days, n_champs = len(self.days), len(champs)

        # Daily counts -> cumulative matrices (days x champs).
        def cum_matrix(counts: pd.Series) -> np.ndarray:
            mat = np.zeros((n_days, n_champs), dtype=np.float32)
            for (day, champ), n in counts.items():
                if champ in self.cidx:
                    mat[self.didx[day], self.cidx[champ]] = n
            return mat.cumsum(axis=0)

        self.cum_picked = cum_matrix(
            players.groupby(["day", "champion"]).gameid.nunique()
        )
        bans = teams[["day", "gameid", "ban1", "ban2", "ban3", "ban4", "ban5"]].melt(
            id_vars=["day", "gameid"],
            value_name="champion",
        ).dropna(subset=["champion"])
        self.cum_banned = cum_matrix(bans.groupby(["day", "champion"]).gameid.nunique())

        # Per-role cumulative pick counts (one days x champs matrix per role).
        self.cum_role = [
            cum_matrix(
                players[players.position == r]
                .groupby(["day", "champion"])
                .gameid.nunique()
            )
            for r in ROLES
        ]

        games_per_day = players.groupby("day").gameid.nunique()
        daily_games = np.zeros(n_days, dtype=np.float32)
        for day, n in games_per_day.items():
            daily_games[self.didx[day]] = n
        self.cum_games = daily_games.cumsum()

        # Per-team cumulative usage (days x champs) and games played.
        self.team_cum: dict[str, np.ndarray] = {}
        self.team_games: dict[str, np.ndarray] = {}
        for team, tdf in players.groupby("teamname"):
            mat = np.zeros((n_days, n_champs), dtype=np.float32)
            for (day, champ), n in tdf.groupby(["day", "champion"]).gameid.nunique().items():
                if champ in self.cidx:
                    mat[self.didx[day], self.cidx[champ]] = n
            self.team_cum[team] = mat.cumsum(axis=0)
            g = np.zeros(n_days, dtype=np.float32)
            for day, n in tdf.groupby("day").gameid.nunique().items():
                g[self.didx[day]] = n
            self.team_games[team] = g.cumsum()

    def _window(self, cum: np.ndarray, day: pd.Timestamp, days: int) -> np.ndarray:
        """Sum over [day - days, day - 1], i.e. strictly before the game day."""
        hi = self.didx[day] - 1
        lo = hi - days
        zero = np.zeros(cum.shape[1], dtype=np.float32) if cum.ndim == 2 else 0.0
        top = cum[hi] if hi >= 0 else zero
        bot = cum[lo] if lo >= 0 else zero
        return top - bot

    def global_rates(self, day: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        games = self._window(self.cum_games, day, RATE_WINDOW_DAYS)
        games = max(games, 1.0)
        pick = self._window(self.cum_picked, day, RATE_WINDOW_DAYS) / games
        ban = self._window(self.cum_banned, day, RATE_WINDOW_DAYS) / games
        return pick, ban

    def role_shares(self, day: pd.Timestamp) -> np.ndarray:
        """(n_champs x 5) role distribution over the trailing ROLE_WINDOW_DAYS.

        Champions with fewer than MIN_ROLE_GAMES windowed games back off to a
        uniform distribution over roles they had ever played before `day`, then
        to fully uniform if they had never been seen.
        """
        win = np.stack(
            [self._window(m, day, ROLE_WINDOW_DAYS) for m in self.cum_role]
        )  # (5, n_champs)
        hi = self.didx[day] - 1
        ever = (
            np.stack([m[hi] for m in self.cum_role]) > 0
            if hi >= 0
            else np.zeros_like(win, dtype=bool)
        )
        shares = np.full_like(win, 1 / len(ROLES))
        tot = win.sum(axis=0)
        ok = tot >= MIN_ROLE_GAMES
        shares[:, ok] = win[:, ok] / tot[ok]
        backoff = ~ok & (ever.sum(axis=0) > 0)
        shares[:, backoff] = ever[:, backoff] / ever[:, backoff].sum(axis=0)
        return shares.T

    def team_usage(self, team: str, day: pd.Timestamp) -> np.ndarray:
        if team not in self.team_cum:
            return np.zeros(len(self.champs), dtype=np.float32)
        games = max(self._window(self.team_games[team], day, USAGE_WINDOW_DAYS), 1.0)
        return self._window(self.team_cum[team], day, USAGE_WINDOW_DAYS) / games


def role_open_probs(
    picks: list[str], shares: np.ndarray, cidx: dict[str, int]
) -> np.ndarray:
    """P(each role is still open) given a team's picks so far.

    Enumerates all assignments of the picked champions to distinct roles,
    weighting each assignment by the product of the champions' role-shares,
    then marginalizes. A flex pick (Poppy: jng/sup) keeps both roles partially
    open until another pick's assignment resolves it.
    """
    open_p = np.ones(len(ROLES), dtype=np.float32)
    idx = [cidx[c] for c in picks if c in cidx]
    if not idx:
        return open_p
    # Smooth so a set of picks whose shares contradict every permutation
    # (e.g. two pure-mid champions) still yields a distribution.
    s = shares[idx] + 1e-3
    filled = np.zeros(len(ROLES))
    total = 0.0
    for perm in itertools.permutations(range(len(ROLES)), len(idx)):
        w = 1.0
        for i, r in enumerate(perm):
            w *= s[i, r]
        total += w
        for r in perm:
            filled[r] += w
    return (open_p - filled / total).astype(np.float32)


def build() -> pd.DataFrame:
    players, teams = load_games()
    champs = sorted(players.champion.dropna().unique())
    cidx = {c: i for i, c in enumerate(champs)}
    fearless_leagues = detect_fearless(players)
    print(f"champions: {len(champs)}, fearless leagues: {sorted(fearless_leagues)}")

    rates = RollingRates(players, teams, champs)

    # One record per game with both teams' ordered picks/bans.
    game_rows = {}
    for _, row in teams.iterrows():
        rec = game_rows.setdefault(
            row.gameid,
            {"date": row.date, "day": row.day, "league": row.league},
        )
        rec[row.side] = {
            "team": row.teamname,
            "bans": [row[f"ban{i}"] for i in range(1, 6)],
            "picks": [row[f"pick{i}"] for i in range(1, 6)],
        }

    # Series key for fearless state: same league/day/team-pair, ordered by date.
    series_prior: dict[tuple, list] = {}
    role_share_cache: dict[pd.Timestamp, np.ndarray] = {}
    out = []
    skipped = 0
    for gameid, rec in sorted(game_rows.items(), key=lambda kv: kv[1]["date"]):
        if "Blue" not in rec or "Red" not in rec:
            skipped += 1
            continue
        picks_ok = all(
            isinstance(c, str)
            for s in ("Blue", "Red")
            for c in rec[s]["picks"]
        )
        if not picks_ok:
            skipped += 1
            continue

        league, day = rec["league"], rec["day"]
        pair = tuple(sorted([rec["Blue"]["team"], rec["Red"]["team"]]))
        skey = (league, day, pair)
        fearless = league in fearless_leagues
        unavailable_series = (
            set().union(*series_prior.get(skey, [set()])) if fearless else set()
        )

        pick_rate, ban_rate = rates.global_rates(day)
        presence = pick_rate + ban_rate
        usage = {
            s: rates.team_usage(rec[s]["team"], day) for s in ("Blue", "Red")
        }
        if day not in role_share_cache:
            role_share_cache[day] = rates.role_shares(day)
        shares = role_share_cache[day]

        taken = set(unavailable_series)  # champs unavailable at draft start
        picks_so_far: dict[str, list[str]] = {"Blue": [], "Red": []}
        game_num = len(series_prior.get(skey, []))
        for seq, dtype, side, ordinal in DRAFT_SEQUENCE:
            actual = rec[side]["bans" if dtype == "ban" else "picks"][ordinal - 1]
            if not isinstance(actual, str):
                continue  # missed ban: no decision was made, state unchanged
            if actual not in cidx:
                taken.add(actual)
                if dtype == "pick":
                    picks_so_far[side].append(actual)
                continue
            opp = "Red" if side == "Blue" else "Blue"
            avail = [c for c in champs if c not in taken]
            ci = np.array([cidx[c] for c in avail])
            n = len(avail)
            # Role need is scored against the team whose next pick is at stake:
            # own open roles for picks, the opponent's for bans.
            ref = side if dtype == "pick" else opp
            p_open = role_open_probs(picks_so_far[ref], shares, cidx)
            cand_shares = shares[ci]  # (n_avail x 5)
            out.append(pd.DataFrame({
                "gameid": gameid,
                "date": rec["date"],
                "league": league,
                "seq": seq,
                "is_ban": int(dtype == "ban"),
                "phase2": int(seq > 12),
                "is_blue": int(side == "Blue"),
                "ordinal": ordinal,
                "fearless": int(fearless),
                "game_in_series": game_num + 1,
                "team": rec[side]["team"],
                "candidate": avail,
                "pick_rate": pick_rate[ci],
                "ban_rate": ban_rate[ci],
                "presence": presence[ci],
                "team_usage": usage[side][ci],
                "opp_usage": usage[opp][ci],
                "role_need": cand_shares @ p_open,
                "role_overlap_max": (cand_shares * p_open).max(axis=1),
                "label": [int(c == actual) for c in avail],
            }))
            taken.add(actual)
            if dtype == "pick":
                picks_so_far[side].append(actual)

        used_this_game = {
            c for s in ("Blue", "Red") for c in rec[s]["picks"] if isinstance(c, str)
        }
        series_prior.setdefault(skey, []).append(used_this_game)

    ds = pd.concat(out, ignore_index=True)
    n_games = ds.gameid.nunique()
    n_decisions = ds.groupby(["gameid", "seq"]).ngroups
    print(f"games: {n_games}, decisions: {n_decisions}, rows: {len(ds)}, skipped games: {skipped}")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(DATA_PROCESSED / "draft_decisions.parquet", index=False)
    return ds


if __name__ == "__main__":
    build()
