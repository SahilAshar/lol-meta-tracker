# Soloq scrape — status & close-out (2026-07-25)

**Written 2026-07-25 ~12:53 ET / 16:53Z.** Session closed out cleanly: the
scrape is stopped, the codespace deleted, and the authoritative DB is local and
verified. This supersedes `2026-07-23-soloq-scrape-resume-handoff.md`.

## Current state (verified)

- **Authoritative DB is LOCAL**: `data/raw/soloq/soloq.db` (gitignored).
  - **329,727 done · 0 pending · 0 failed** · `PRAGMA integrity_check = ok`.
  - Grew from the 78,171 seed → 329,727 this session (**+251,556, ~4.2×**).
- **No scraper running. No codespace running for this** (see codespace note).
- **No background monitors / caffeinate running.** All stopped.
- Off-box backups: 6 verified gzip snapshots in `data/raw/soloq/backups/`
  (newest `soloq_20260725T165240Z.db.gz`, done=329,727).
- Pre-promotion safety copies of the DB at each promote step:
  `soloq.db.pre-promo-20260724` (78k), `…-20260725` (149k),
  `…pre-final-promo` (329,417). Safe to delete once you trust the current DB.

## What happened this session

1. **Run 1** (`--hours 9`): 78,171 → 149,910. Clean finish.
2. **Run 2** (`--hours 23`): 149,910 → 329,417. A **codespace restart** ~04:35Z
   killed the detached process ~11h early (not a crash/OOM/key issue — the box
   rebooted). Auto-recovered by a supervisor loop; finished clean at the 23h
   target.
3. **Failed-row reset**: the 59 `failed` NA matches were transient (they return
   200 now), so reset to `pending`; they drained to `done` — hence 0 failed.
4. **Cursor fix shipped to `main`** (commit `2950d36`, see below).
5. **Close-out**: killed the scrape, pulled the final DB off-box, promoted to
   local, deleted the `special-train` codespace.

## Code change shipped: per-cycle resume cursor (on `main`)

`scripts/riot_soloq_scrape.py` — new `walked(day, region, puuid)` table. On each
cycle, the scraper skips players already walked today and resumes where it left
off instead of re-walking from player 0. The walk is cleared once a cycle covers
the full ladder (so the next cycle still re-discovers everyone for new games).

**Why:** a restart previously re-spent ~20–40 min of match-list calls
re-covering already-walked players — a real tax against the rate ceiling, and
restarts recur. With this, restarts are ~free. The change is idempotent and
auto-migrates (`CREATE TABLE IF NOT EXISTS`), so existing DBs just work.

## Throughput reality (why it's "slow")

Not the code — it runs at ~90%+ of the hard **dev-key rate ceiling**:
- Limits: **20 req/s (burst) + 100 req / 2 min (sustained)** per host. The
  100/2min governs → ~0.83 req/s/host. Bursting doesn't help (drains the 2-min
  bucket then stalls).
- Match data (list + detail — the bulk of calls) goes to **3 routing hosts**
  (`americas`/`europe`/`asia`), all already used → **~2.4 req/s aggregate max**.
  Adding platforms (br/la/…) shares an existing routing bucket → no gain.
- **The only real lever is a production key** (much higher per-host limits).
  Everything else is marginal yield-tuning.
- Practical rate at the ceiling: ~7,800 games/hr early, degrading as Master+
  dedup overlap grows.

## Codespaces

- **`special-train-4pj99jq7r6fqp6v` — DELETED** during close-out (its DB was
  pulled to local first; nothing unique lost).
- **`effective-space-couscous-q57gg77r5rh4pp6` — LEFT UNTOUCHED.** Created
  07-24, currently *Shutdown*, has uncommitted changes, and was **not created by
  this workflow.** Not deleted without owner sign-off. Clean it up if unneeded.

## API keys

Every dev key used this session is **expired or will be** (dev keys last ~24h).
Any future run needs a **fresh key** from developer.riotgames.com. Update BOTH
stores at run start:
```
gh secret set RIOT_API_KEY --user --app codespaces \
  --repos SahilAshar/lol-meta-tracker --body "RGAPI-..."
# and overwrite .riot-api-key at the repo root (gitignored)
```
The scraper reads `RIOT_API_KEY` env first, else the `.riot-api-key` file. In a
codespace, only **login shells** (`bash -lc`) get the secret env, so the file is
the reliable path for a detached `nohup` launch.

## To resume scraping later (runbook)

1. Fresh Riot key → set secret + `.riot-api-key` (above).
2. `gh codespace create -R SahilAshar/lol-meta-tracker -m basicLinux32gb
   --idle-timeout 240m`.
3. Seed it from the local DB so it resumes at 329,727 instead of re-fetching
   (gzip + ssh stdin; `gh codespace cp` is broken):
   ```
   python3 scripts/riot_soloq_scrape.py --snapshot   # consistent backup.db
   gzip -kf data/raw/soloq/backup.db
   gh codespace ssh -c <NAME> -- 'mkdir -p /workspaces/lol-meta-tracker/data/raw/soloq &&
     gunzip -c > /workspaces/lol-meta-tracker/data/raw/soloq/soloq.db' \
     < data/raw/soloq/backup.db.gz
   ```
4. Launch detached (the `bash -lc` + `</dev/null` matters):
   ```
   gh codespace ssh -c <NAME> -- "bash -lc 'cd /workspaces/lol-meta-tracker &&
     nohup python3 scripts/riot_soloq_scrape.py --hours N </dev/null >
     data/raw/soloq/scrape.log 2>&1 & echo launched'"
   ```
   The launch ssh will appear to hang (gh holds the channel); the process still
   starts — verify from a fresh connection, don't trust the hang.
5. **Keep-alive is required**: a running process does NOT reset the Codespaces
   idle timer — only ssh activity does. Ping every ~25 min (also your health
   check). The scraper self-resumes from the DB on any restart now, so a
   codespace reboot is cheap (no re-walk tax).
6. **Verify exactly one scraper** after launch (`ps ... | grep -c python3`).
   Double-launches double the request rate → 429s.
7. End of run: pull a verified snapshot off-box, promote to local `soloq.db`
   only after counts verify, then `gh codespace delete`.

## Open items / next levers

- **Production API key** — the only way to raise the throughput ceiling.
- **More regions** would only help ladder *discovery*, not match throughput
  (routing hosts are the bottleneck and all 3 are used).
- The ladder is far from exhausted — each run got ~2,900–3,900 of ~11k players
  per region into cycle 1, so more runs still yield meaningfully new games.
- Rung 1 (synergy) was NO-GO; marginal value of more data is scale for future
  rungs (role-aware 1b) and the meta-rate / ban time-signal directions.

## Repo etiquette (unchanged)

Never stage `docs/ROADMAP.md`, `artifact/*`, or anything you didn't create; run
`date` before writing dates; everything under `data/raw/soloq/` stays
gitignored; commit/push only when Sahil approves.
