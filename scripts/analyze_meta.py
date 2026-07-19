"""Compute cross-league champion pick/ban rates from Oracle's Elixir data.

Filters to the four major leagues (LCK, LPL, LEC, LCS) within a date window
(default: summer 2026 start). If the window has no games yet — the leagues
start July 22-29 — falls back to the most recent 8 weeks of available data so
the pipeline always produces output.

Outputs to data/processed/:
  pick_rates.csv            league, champion, games_picked, total_games, pick_rate
  ban_rates.csv             league, champion, games_banned, total_games, ban_rate
  pick_rate_comparison.csv  champion x league pivot of pick rates (+ mean)
  weekly_pick_rates.csv     iso_week, league, champion, pick_rate (for WoW deltas)
  meta.json                 run metadata (window, games per league, fallback flag)
"""

import argparse
import json
import sys
from datetime import timedelta

import pandas as pd

from common import (
    CURRENT_YEAR,
    DATA_PROCESSED,
    DEFAULT_SINCE,
    MAJOR_LEAGUES,
    raw_csv_path,
)

USECOLS = [
    "gameid", "league", "split", "date", "position", "champion",
    "ban1", "ban2", "ban3", "ban4", "ban5",
]


def load_major_leagues(year: int) -> pd.DataFrame:
    path = raw_csv_path(year)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/download_data.py first")
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    df = df[df["league"].isin(MAJOR_LEAGUES)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def apply_window(df: pd.DataFrame, since: str) -> tuple[pd.DataFrame, dict]:
    """Filter to games on/after `since`; fall back to recent data if empty."""
    since_ts = pd.Timestamp(since)
    windowed = df[df["date"] >= since_ts]
    meta = {"since": since, "fallback": False}
    if windowed["gameid"].nunique() == 0:
        latest = df["date"].max()
        fallback_start = latest - timedelta(weeks=8)
        windowed = df[df["date"] >= fallback_start]
        meta = {
            "since": str(fallback_start.date()),
            "fallback": True,
            "fallback_reason": f"No games on/after {since} yet; using most recent 8 weeks",
        }
    return windowed.copy(), meta


def compute_pick_rates(df: pd.DataFrame) -> pd.DataFrame:
    players = df[df["position"] != "team"].dropna(subset=["champion"])
    games = df.groupby("league")["gameid"].nunique().rename("total_games")
    picked = (
        players.groupby(["league", "champion"])["gameid"]
        .nunique()
        .rename("games_picked")
        .reset_index()
    )
    picked = picked.merge(games, on="league")
    picked["pick_rate"] = picked["games_picked"] / picked["total_games"]
    return picked.sort_values(["league", "pick_rate"], ascending=[True, False])


def compute_ban_rates(df: pd.DataFrame) -> pd.DataFrame:
    teams = df[df["position"] == "team"].drop(columns=["champion"])
    games = df.groupby("league")["gameid"].nunique().rename("total_games")
    bans = teams.melt(
        id_vars=["gameid", "league"],
        value_vars=["ban1", "ban2", "ban3", "ban4", "ban5"],
        value_name="champion",
    ).dropna(subset=["champion"])
    banned = (
        bans.groupby(["league", "champion"])["gameid"]
        .nunique()
        .rename("games_banned")
        .reset_index()
    )
    banned = banned.merge(games, on="league")
    banned["ban_rate"] = banned["games_banned"] / banned["total_games"]
    return banned.sort_values(["league", "ban_rate"], ascending=[True, False])


def compute_comparison(pick_rates: pd.DataFrame) -> pd.DataFrame:
    pivot = pick_rates.pivot_table(
        index="champion", columns="league", values="pick_rate", fill_value=0.0
    )
    for lg in MAJOR_LEAGUES:  # leagues with no games yet still get a column
        if lg not in pivot.columns:
            pivot[lg] = float("nan")
    pivot = pivot[MAJOR_LEAGUES]
    pivot["mean_pick_rate"] = pivot.mean(axis=1, skipna=True)
    return pivot.sort_values("mean_pick_rate", ascending=False)


def compute_weekly(df: pd.DataFrame) -> pd.DataFrame:
    players = df[df["position"] != "team"].dropna(subset=["champion"]).copy()
    iso = players["date"].dt.isocalendar()
    players["iso_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    games = players.groupby(["iso_week", "league"])["gameid"].nunique().rename("total_games")
    picked = (
        players.groupby(["iso_week", "league", "champion"])["gameid"]
        .nunique()
        .rename("games_picked")
        .reset_index()
    )
    picked = picked.merge(games, on=["iso_week", "league"])
    picked["pick_rate"] = picked["games_picked"] / picked["total_games"]
    return picked.sort_values(["iso_week", "league", "pick_rate"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=CURRENT_YEAR)
    parser.add_argument("--since", default=DEFAULT_SINCE, help="Start of analysis window (YYYY-MM-DD)")
    args = parser.parse_args()

    df = load_major_leagues(args.year)
    windowed, meta = apply_window(df, args.since)

    games_per_league = (
        windowed.groupby("league")["gameid"].nunique().reindex(MAJOR_LEAGUES, fill_value=0)
    )
    meta["year"] = args.year
    meta["games_per_league"] = games_per_league.to_dict()
    meta["window_end"] = str(windowed["date"].max().date()) if len(windowed) else None

    pick_rates = compute_pick_rates(windowed)
    ban_rates = compute_ban_rates(windowed)
    comparison = compute_comparison(pick_rates)
    weekly = compute_weekly(windowed)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pick_rates.to_csv(DATA_PROCESSED / "pick_rates.csv", index=False)
    ban_rates.to_csv(DATA_PROCESSED / "ban_rates.csv", index=False)
    comparison.to_csv(DATA_PROCESSED / "pick_rate_comparison.csv")
    weekly.to_csv(DATA_PROCESSED / "weekly_pick_rates.csv", index=False)
    (DATA_PROCESSED / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Window: since {meta['since']}" + (" (FALLBACK)" if meta["fallback"] else ""))
    print(f"Games per league: {meta['games_per_league']}")
    print(f"Champions picked: {comparison.shape[0]}")
    print(f"Wrote outputs to {DATA_PROCESSED}/")


if __name__ == "__main__":
    sys.exit(main())
