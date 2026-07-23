"""Long-running high-elo soloq scraper via the official Riot API (personal key).

Collects ranked-solo (queue 420) match details for Master+ players in KR/EUW/NA
into a SQLite store. Per docs/2026-07-22-soloq-data-research.md, historical
matches carry the unordered 10-pick set, ordered bans, roles, and outcome —
enough for co-occurrence embeddings (soloq rung 0), trailing meta rates (the
ban time-signal rung), and the Challenger leading-indicator. Pick ORDER is not
in the API, so no sequence pretraining — that limitation is upstream, not ours.

Design:
  - One thread per region; each paces to ~96 requests / 2 min (the key limit
    is 100/2min per regional host) and honors 429 Retry-After.
  - Cycle: snapshot the Chall/GM/Master ladder -> pull each player's recent
    ranked match IDs -> fetch match details as they queue -> repeat until the
    deadline. Later cycles mostly find new games played during the day.
  - SQLite state machine: match IDs enter as 'pending'; fetches move them to
    'done' (with payload) or back to 'pending' with attempts+1, then 'failed'
    at 3 attempts. Restarts resume exactly where they left off — a crashed
    fetch is retried, never silently dropped.
  - Progress: data/raw/soloq/scrape_status.json, refreshed every batch.

Key: RIOT_API_KEY env var (Codespaces secret), or a .riot-api-key file at the
repo root (gitignored). Everything under data/raw/soloq/ is gitignored.

Run:    python3 scripts/riot_soloq_scrape.py --hours 10
Export: python3 scripts/riot_soloq_scrape.py --export matches.jsonl.gz
Peek:   sqlite3 data/raw/soloq/soloq.db \
          "SELECT region, state, COUNT(*) FROM matches GROUP BY 1, 2"

Stdlib only — runs on a bare Codespaces image, no venv needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "raw" / "soloq"
DB_PATH = OUT_DIR / "soloq.db"

REGIONS = {
    "kr": {"platform": "kr", "routing": "asia"},
    "euw": {"platform": "euw1", "routing": "europe"},
    "na": {"platform": "na1", "routing": "americas"},
}
TIER_ENDPOINTS = ["challengerleagues", "grandmasterleagues", "masterleagues"]
QUEUE_SOLO = 420
REQUEST_GAP = 1.25  # seconds between calls per regional host => 96 per 2 min
MAX_ATTEMPTS = 3
BATCH = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches(
    match_id      TEXT PRIMARY KEY,
    region        TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    game_creation INTEGER,
    game_version  TEXT,
    queue_id      INTEGER,
    duration      INTEGER,
    payload       TEXT,
    fetched_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_region_state ON matches(region, state);
CREATE TABLE IF NOT EXISTS ladder(
    day    TEXT NOT NULL,
    region TEXT NOT NULL,
    tier   TEXT NOT NULL,
    puuid  TEXT NOT NULL,
    lp     INTEGER,
    wins   INTEGER,
    losses INTEGER,
    PRIMARY KEY(day, region, tier, puuid)
);
"""

status_lock = threading.Lock()


def connect() -> sqlite3.Connection:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def load_key() -> str:
    key = os.environ.get("RIOT_API_KEY", "").strip()
    if not key:
        key_file = REPO / ".riot-api-key"
        if key_file.exists():
            key = key_file.read_text().strip()
    if not key.startswith("RGAPI-"):
        sys.exit(
            "No Riot API key. Set RIOT_API_KEY or write the key to "
            f"{REPO / '.riot-api-key'} (gitignored)."
        )
    return key


def write_status(con: sqlite3.Connection) -> None:
    with status_lock:
        rows = con.execute(
            "SELECT region, state, COUNT(*) FROM matches GROUP BY 1, 2"
        ).fetchall()
        summary: dict = {}
        for region, state, n in rows:
            summary.setdefault(region, {})[state] = n
        summary["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (OUT_DIR / "scrape_status.json").write_text(json.dumps(summary, indent=1))


class RegionScraper(threading.Thread):
    def __init__(self, name: str, cfg: dict, key: str, deadline: float, start_days: int):
        super().__init__(name=name, daemon=True)
        self.region = name
        self.platform = cfg["platform"]
        self.routing = cfg["routing"]
        self.key = key
        self.deadline = deadline
        self.start_time = int(time.time() - start_days * 86400)
        self.last_call = 0.0
        self.calls = 0
        self.done = 0
        self.fatal: str | None = None
        self.con: sqlite3.Connection | None = None  # created on the thread

    def expired(self) -> bool:
        return time.time() > self.deadline or self.fatal is not None

    # -- HTTP ----------------------------------------------------------------
    def get(self, host: str, path: str):
        """Rate-limited GET returning parsed JSON, or None on skippable errors."""
        for attempt in range(4):
            wait = self.last_call + REQUEST_GAP - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.monotonic()
            self.calls += 1
            # Cloudflare on Riot's edge 403s the default Python-urllib UA.
            req = urllib.request.Request(
                f"https://{host}{path}",
                headers={
                    "X-Riot-Token": self.key,
                    "User-Agent": "lol-meta-tracker/0.1 (personal research scraper)",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(float(e.headers.get("Retry-After", 10)))
                elif e.code in (500, 502, 503, 504):
                    time.sleep(5 * (attempt + 1))
                elif e.code in (401, 403):
                    self.fatal = f"HTTP {e.code} — key invalid or expired"
                    return None
                else:  # 404 etc: skip this resource
                    return None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(5 * (attempt + 1))
        return None

    def platform_get(self, path: str):
        return self.get(f"{self.platform}.api.riotgames.com", path)

    def routing_get(self, path: str):
        return self.get(f"{self.routing}.api.riotgames.com", path)

    # -- pipeline ------------------------------------------------------------
    def ladder_puuids(self) -> list[str]:
        """Snapshot Chall/GM/Master; return puuids best-first so the top of
        the ladder is covered even if the deadline cuts the cycle short."""
        puuids: list[str] = []
        day = time.strftime("%Y%m%d")
        for tier in TIER_ENDPOINTS:
            if self.expired():
                break
            data = self.platform_get(
                f"/lol/league/v4/{tier}/by-queue/RANKED_SOLO_5x5"
            )
            if not data:
                continue
            entries = sorted(
                data.get("entries", []),
                key=lambda e: e.get("leaguePoints", 0), reverse=True,
            )
            self.con.executemany(
                "INSERT OR REPLACE INTO ladder VALUES(?,?,?,?,?,?,?)",
                [
                    (day, self.region, tier, e["puuid"], e.get("leaguePoints", 0),
                     e.get("wins", 0), e.get("losses", 0))
                    for e in entries if e.get("puuid")
                ],
            )
            self.con.commit()
            puuids.extend(e["puuid"] for e in entries if e.get("puuid"))
        return puuids

    def discover(self, puuid: str) -> None:
        ids = self.routing_get(
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
            f"?queue={QUEUE_SOLO}&startTime={self.start_time}&count=100"
        )
        if ids:
            self.con.executemany(
                "INSERT OR IGNORE INTO matches(match_id, region) VALUES(?, ?)",
                [(mid, self.region) for mid in ids],
            )
            self.con.commit()

    def fetch_match(self, match_id: str) -> dict | None:
        data = self.routing_get(f"/lol/match/v5/matches/{match_id}")
        if not data or "info" not in data:
            return None
        info = data["info"]
        return {
            "gameCreation": info.get("gameCreation"),
            "gameVersion": info.get("gameVersion"),
            "queueId": info.get("queueId"),
            "gameDuration": info.get("gameDuration"),
            "participants": [
                {
                    "champ": p.get("championName"),
                    "team": p.get("teamId"),
                    "pos": p.get("teamPosition", ""),
                    "win": bool(p.get("win")),
                }
                for p in info.get("participants", [])
            ],
            "bans": [
                {"champId": b.get("championId"), "turn": b.get("pickTurn"),
                 "team": t.get("teamId")}
                for t in info.get("teams", [])
                for b in t.get("bans", [])
            ],
        }

    def drain(self, limit: int = BATCH) -> int:
        """Fetch up to `limit` pending matches; return how many were tried."""
        rows = self.con.execute(
            "SELECT match_id FROM matches "
            "WHERE region = ? AND state = 'pending' AND attempts < ? LIMIT ?",
            (self.region, MAX_ATTEMPTS, limit),
        ).fetchall()
        for (mid,) in rows:
            if self.expired():
                break
            rec = self.fetch_match(mid)
            if self.fatal:
                break  # don't burn an attempt on a dead key
            if rec:
                self.con.execute(
                    "UPDATE matches SET state='done', game_creation=?, "
                    "game_version=?, queue_id=?, duration=?, payload=?, "
                    "fetched_at=datetime('now') WHERE match_id=?",
                    (rec["gameCreation"], rec["gameVersion"], rec["queueId"],
                     rec["gameDuration"],
                     json.dumps({"participants": rec["participants"],
                                 "bans": rec["bans"]}),
                     mid),
                )
                self.done += 1
            else:
                self.con.execute(
                    "UPDATE matches SET attempts = attempts + 1, "
                    "state = CASE WHEN attempts + 1 >= ? THEN 'failed' "
                    "ELSE 'pending' END WHERE match_id = ?",
                    (MAX_ATTEMPTS, mid),
                )
        self.con.commit()
        write_status(self.con)
        return len(rows)

    def run(self) -> None:
        self.con = connect()
        cycle = 0
        while not self.expired():
            cycle += 1
            puuids = self.ladder_puuids()
            print(f"[{self.region}] cycle {cycle}: {len(puuids)} ladder players",
                  flush=True)
            for i, puuid in enumerate(puuids):
                if self.expired():
                    break
                self.discover(puuid)
                self.drain()
                if i % 50 == 0:
                    print(f"[{self.region}] cycle {cycle}: player {i}/{len(puuids)}, "
                          f"{self.done} done, {self.calls} calls", flush=True)
            while not self.expired() and self.drain():
                pass  # discovery finished; clear the remaining backlog
            if not self.expired() and time.time() + 600 < self.deadline:
                time.sleep(600)  # new games accumulate slowly between cycles
        print(f"[{self.region}] finished: {self.done} matches, {self.calls} calls"
              + (f", FATAL: {self.fatal}" if self.fatal else ""), flush=True)


def export(path: str) -> None:
    con = connect()
    n = 0
    with gzip.open(path, "wt") as f:
        for mid, region, gc, gv, qid, dur, payload in con.execute(
            "SELECT match_id, region, game_creation, game_version, queue_id, "
            "duration, payload FROM matches WHERE state = 'done'"
        ):
            rec = {"matchId": mid, "region": region, "gameCreation": gc,
                   "gameVersion": gv, "queueId": qid, "gameDuration": dur,
                   **json.loads(payload)}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"exported {n} matches to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--regions", default="kr,euw,na")
    ap.add_argument("--start-days", type=int, default=45,
                    help="only pull matches newer than this many days")
    ap.add_argument("--export", metavar="PATH",
                    help="export done matches to jsonl.gz and exit")
    ap.add_argument("--snapshot", action="store_true",
                    help="write a consistent backup.db next to the db and exit "
                         "(safe against a live WAL writer; for off-box backups)")
    args = ap.parse_args()

    if args.snapshot:
        src = connect()
        dst_path = OUT_DIR / "backup.db"
        dst_path.unlink(missing_ok=True)
        dst = sqlite3.connect(dst_path)
        with dst:
            src.backup(dst)
        dst.close()
        print(f"snapshot written to {dst_path}")
        return

    if args.export:
        export(args.export)
        return

    key = load_key()
    deadline = time.time() + args.hours * 3600

    threads = []
    for name in args.regions.split(","):
        name = name.strip()
        if name not in REGIONS:
            sys.exit(f"unknown region {name!r}; choices: {list(REGIONS)}")
        t = RegionScraper(name, REGIONS[name], key, deadline, args.start_days)
        t.start()
        threads.append(t)

    for t in threads:
        while t.is_alive():
            t.join(timeout=60)
    print("All regions finished.", flush=True)
    if any(t.fatal for t in threads):
        sys.exit(1)


if __name__ == "__main__":
    main()
