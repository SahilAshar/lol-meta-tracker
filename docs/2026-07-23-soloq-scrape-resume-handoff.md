# Handoff: resume the soloq scrape in a fresh GitHub Codespace

**Written 2026-07-23 ~22:20 ET. Audience: a fresh agent tasked with
restarting the scrape.** This updates the 07-23 morning deploy handoff with
tonight's verified state. The architecture decisions there still stand — ONE
codespace, ONE scraper process, ONE SQLite file; a fleet adds zero
throughput against the per-key rate limit. Read
`scripts/riot_soloq_scrape.py`'s docstring for schema/states/commands.

## State tonight (verified 2026-07-23 ~22:00 ET — trust this, not the repo's

older docs)

- **The authoritative db is LOCAL**: `data/raw/soloq/soloq.db` on the Mac
  (121MB, gitignored) — 78,171 done + 2,533 pending, latest game
  2026-07-23 21:50 UTC. The morning codespace was deleted after its day run;
  no codespace currently holds newer data.
- The two surviving codespaces are stale/deletable:
  - `reimagined-engine-q57gg77w6xh97x7` — created 07-20, **predates the
    Codespaces secret, so `RIOT_API_KEY` is not injected into it and never
    will be** (secrets inject at creation). It has a copy of tonight's db at
    `/workspaces/lol-meta-tracker/data/raw/soloq/soloq.db` from a launch
    attempt that failed on the missing key. Do not reuse it; create fresh.
  - `congenial-succotash-6x97799qvgh56g6` — 07-19 leftover, no soloq data.
- The scraper code is on origin/main (latest relevant: `6f2e17f` lock fix,
  `a381ecc` --snapshot). Rung 1 (synergy) is done — NO-GO, see
  `docs/2026-07-23-synergy-rung1-results.md` — so the marginal value of new
  data is scale for future rungs (role-aware rung 1b wants 2-3x more games)
  and the meta-rate/ban time-signal direction.

## Step 0 — the key, before anything else

The `RIOT_API_KEY` is a DEVELOPMENT key refreshed the morning of
**2026-07-23** ET; dev keys die ~24h after issue, so assume it is expired or
about to be. **Ask Sahil for a fresh key first** (regenerated at
developer.riotgames.com), then update BOTH stores before creating the
codespace:

```
gh secret set RIOT_API_KEY --user --app codespaces \
  --repos SahilAshar/lol-meta-tracker --body "RGAPI-..."
# and overwrite .riot-api-key at the local repo root (gitignored)
```

Never commit the key. If the scraper ever logs 401/403 key-invalid mid-run,
data is safe — refresh the key, restart, it resumes.

## Runbook (deltas from the morning doc marked ★)

1. `gh codespace create -R SahilAshar/lol-meta-tracker -m basicLinux32gb
   --idle-timeout 240m` — the 240m matters; the default 30m is what makes
   stale codespaces useless for detached runs.
2. Verify the secret arrived:
   `gh codespace ssh -c <NAME> -- 'echo ${RIOT_API_KEY:0:8}'` → `RGAPI-`.
3. ★ **Seed the codespace with the local db so it resumes instead of
   re-fetching 78k matches.** `gh codespace cp` failed with a bare scp
   error tonight; streaming over ssh stdin works and the db gzips 121MB →
   17MB:
   ```
   cd ~/Documents/repos/lol-meta-tracker
   python3 scripts/riot_soloq_scrape.py --snapshot   # consistent backup.db
   gzip -kf data/raw/soloq/backup.db
   gh codespace ssh -c <NAME> -- \
     'mkdir -p /workspaces/lol-meta-tracker/data/raw/soloq &&
      gunzip -c > /workspaces/lol-meta-tracker/data/raw/soloq/soloq.db' \
     < data/raw/soloq/backup.db.gz
   ```
   Verify the count on the far side (~78k done) before launching. md5 the
   .gz both sides if paranoid.
4. Launch detached (log inside the gitignored soloq dir):
   `gh codespace ssh -c <NAME> -- 'cd /workspaces/lol-meta-tracker && nohup
   python3 scripts/riot_soloq_scrape.py --hours 10 >
   data/raw/soloq/scrape.log 2>&1 & echo launched'`
5. Keep-alive + telemetry every ~20-30 min (ssh counts as activity; a
   running process does NOT). Use the Monitor tool (foreground sleep is
   blocked). Tail `scrape.log` + `scrape_status.json`. ★ Make the check
   loop alert on `FATAL|Traceback|No Riot API key`, not just progress —
   tonight's failure mode was the scraper exiting cleanly at t=0 with a
   one-line key error while the launch command reported success.
6. Hourly off-box backup, end-of-day retrieval, WAL checkpoint, sanity
   counts, then `gh codespace delete` — unchanged from the morning doc
   (§ runbook steps 4-6). Local `soloq.db` is overwritten by the pulled
   file only after counts verify.

## Repo etiquette

Never stage `docs/ROADMAP.md`, `artifact/*`, or anything you didn't create;
run `date` before writing dates; commit/push only when Sahil approves.
Everything under `data/raw/soloq/` stays gitignored.
