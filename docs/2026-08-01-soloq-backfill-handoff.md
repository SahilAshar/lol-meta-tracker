# Handoff: soloq calendar backfill — what landed, and the one blocker before it reaches the model

**Written 2026-08-01 ~13:25 ET. Audience: a fresh agent picking up the soloq
scrape / rung-2 lane.** This supersedes the scrape sections of
`docs/2026-07-25-soloq-scrape-status.md`. The rung-2 verdict in
`docs/2026-07-31-rung2-rerun-results.md` still stands unamended — nothing here
re-runs it. What changed is the input that doc named as the binding
constraint.

## TL;DR

The 2026-06-08 coverage floor is gone. The db now spans **2026-03-03 →
present**, 877k games. Under the new floor, live (non-placeholder) **train**
decisions go from **1,360 → 19,277 — 14.2×**. That is the exact quantity the
rung-2 re-run identified as the reason arm 1's null was sparsity-muted.

**None of it reaches the model yet.** `SOLOQ_START` is hardcoded to
`2026-06-08` in three scripts, and the lift tables have not been rebuilt.
Until both happen, the backfill is invisible to training. That is step 1
below and it is the whole ballgame.

## What was run and why

The prior direction — extend the same window forward — was exhausted. The
last same-window run spent ~11,400 calls/region per 4h segment for ~3,500–4,500
new matches (**~0.3 new matches per call**). The window 2026-06-08 → 07-31 was
saturated.

The floor was an artifact, not a limit: `riot_soloq_scrape.py` computes
`start_time = now − start_days × 86400`, `--start-days` defaulted to 45, and
the first scrape ran 2026-07-25. Raising it backfills earlier.

Run: `--hours 23 --start-days 150`, launched 2026-07-31 13:36 ET, finished
clean 2026-08-01 12:36 ET (full 23h, all three regions reported `finished`,
0 FATAL — the key outlasted the run, unlike the 21.4h death the run before).

| | before | after |
|---|---|---|
| matches | 688,180 | **876,689** (+188,509) |
| coverage | 2026-06-08 → 07-31 | **2026-03-03 → 08-01** |
| pre-Jun-08 games | 0 | **93,897** |
| efficiency | ~0.3 new/call | **~0.97 new/call** (194,559 calls) |

Per region, near-identical — kr 63,269 · na 63,349 · euw 61,891 — so no
region silently underperformed. Backfilling was ~3× more productive per unit
of rate limit than extending the window would have been.

## Verified facts about the Riot API (don't re-litigate these)

- **Riot does NOT cap match-history depth.** A pre-flight probe paginated to
  `start=100` and pulled a full second page reaching 2026-03-12. This was the
  stated risk going in; it is not real. 150-day history is genuinely
  retrievable.
- **The 100-ID cap is ours, not Riot's.** `discover()`
  (`scripts/riot_soloq_scrape.py:229-231`) requests `count=100` with no
  `start=` pagination. For ~7 of 12 sampled players that cap was already
  binding at 45 days, so `--start-days 150` hands those players back an
  identical list. It has not cost anything yet — throughput is pinned at the
  per-host rate limit either way and the pending queue stays drained (<1k).
  It becomes the binding constraint only if discovery starts starving the
  fetchers. **Left unchanged deliberately.**
- **`walked` is keyed by `(day, region)`.** A run started on a new calendar
  day sees an empty table and re-walks the full ladder. This is why the
  Jul 31 run only reached ~85% of cycle 1 — it inherited ~7.9k players/region
  already marked by that morning's 45-day run and skipped them, so those
  players are still covered only to Jun 8.

## Currently running

A second run is live, launched **2026-08-01 13:18 ET**, same command
(`--hours 23 --start-days 150`), ends **~12:18 ET Aug 2**.

- PID **47318** (`caffeinate -ims` wrapper 47320), log
  `logs/scrape-20260801T131806.log`.
- All three regions started with **`0 already walked`** — the full ~11k
  players/region get the 150-day window this time. This is the run that
  picks up the players the Jul 31 run skipped, and it is where the remaining
  backfill is.
- Key `RGAPI-c135b86d…` (in `.riot-api-key`, gitignored) verified against all
  three regional hosts before launch — no 401 propagation lag. Issued
  ~13:15 ET Aug 1, so it dies ~13:15 Aug 2; the run ends before that.

Health check, which implements the per-region rule below:

```
bash scripts/scrape_health.sh 120     # new, uncommitted
```

## Step 1 — the blocker: move the cutoff and rebuild the tables

The backfill is inert until this happens. `SOLOQ_START = pd.Timestamp("2026-06-08")`
appears in:

- `scripts/draft_dataset.py:80` (and the comment at :59)
- `scripts/experiment_v12_rung2_outcome.py:54` (plus prose at :19, :22, :256, :269)
- `scripts/experiment_v12_rung2_gbm.py` — prose at :6, :8, :252

Then rebuild the soloq lift tables from the enlarged db and regenerate
`data/processed/draft_decisions.parquet`.

**Two traps, both already paid for once:**

1. **`data/processed/expcache_rung2gbm/` keys on `(config, family, seed)` with
   no data hash.** A re-run against rebuilt tables will silently reuse stale
   fits and reproduce the previous numbers exactly. Clear it (the Jul 31 run
   moved the stale copy to `expcache_rung2gbm.pre-rebuild-20260725/`), or add
   a data hash to the key. See `2026-07-31-rung2-rerun-results.md`.
2. **`verify_backup`'s exempt-column logic** was narrowed on Jul 31 to let the
   three `SOLOQ_FEATURES` columns move while asserting the other 24
   byte-identical. That fix is in and should hold for this rebuild too.

## What step 1 is expected to buy (computed, not estimated)

Recomputed against `draft_decisions.parquet` using the real split
(`is_test = league=="EWC" & month==7`; `VAL_DAYS = 14`). This reproduces the
rung-2 doc's table exactly at the old floor, so the reconstruction is sound:

| split | rows | decisions | ≥06-08 | ≥03-03 | live decisions |
|---|---|---|---|---|---|
| train | 3,805,816 | 28,936 | 4.6% | **66.6%** | 1,360 → **19,277** |
| val | 135,540 | 1,080 | 100% | 100% | 1,080 → 1,080 |
| train+val | 3,941,356 | 30,016 | 7.9% | 67.8% | 2,440 → 20,357 |
| full | 4,076,256 | 31,016 | 11.0% | 68.8% | 3,440 → 21,357 |

The rung-2 re-run concluded: *"With 4.7% live train decisions, the binding
constraint is calendar coverage."* That constraint is now substantially
relieved on the train side.

**The honest caveat — coverage ≠ density.** The floor moved to 2026-03-03 but
the games are not evenly spread:

```
2026-03    9,346     2026-06  289,858
2026-04   16,779     2026-07  510,710
2026-05   48,215     2026-08    2,069
```

March carries ~3% of June's volume. Lift tables in the Mar–Apr window will be
built from thin samples, so early-window decisions gain *noisy* features
rather than good ones — a different failure mode from the neutral
placeholders they replace, and not obviously better for those rows.

**Therefore: do not just flip the constant to 03-03 and re-run once.** Run the
gate at **three floors — 2026-06-08 (control), 2026-05-01, 2026-03-03** — so
"more live rows" is separated from "noisier tables". If the effect is real it
should be monotone-ish or peak at the dense floor; if 03-03 underperforms
05-01, the answer is a density-weighted floor, not more calendar.

Also carried forward from the rung-2 doc and still true: the **1,080-decision
val set cannot resolve a sub-1pt top-3 effect**. A coverage re-run needs a
larger val window or an explicitly underpowered result stated up front.
Widening `VAL_DAYS` is now cheaper than it was, since val no longer has to sit
inside a narrow covered window.

## Operational rules (each learned by losing a run)

- **Plug the machine in; leave the lid open.** Clamshell sleep cost two runs
  (~12.7h and ~8h) and `caffeinate` cannot prevent it. `caffeinate -s` only
  engages on AC. Launch as
  `nohup caffeinate -ims python3 -u scripts/... &`.
  *Both Jul 31 and the current run were started on battery anyway — Jul 31
  survived 23h (88%→62%), but this is luck, not headroom. The current run
  began at ~62%.*
- **Kill by explicit PID, never `pkill -f "python3 ..."`.** The real command
  line is `/opt/homebrew/.../Python -u scripts/...` — no `python3` in it, so
  that pattern matches nothing while appearing to succeed. It once left a
  duplicate scraper running ~19h and halved NA/EUW throughput.
- **Verify all three regions started.** A fresh key can 401 on one regional
  host from propagation lag; that thread dies fatal while the others run for
  23h. Require `grep "ladder players"` to show kr, euw, and na, and
  `grep -c FATAL` to be 0. Better: probe all three hosts *before* launching
  (one call each) — that is what was done for the current run.
- **Monitor per region, not the total.** After a wake, a single thread can
  wedge at exactly 0 new matches while a healthy total hides it. Restart is
  lossless — the SQLite state machine resumes exactly where it stopped.
- **23h against a 24h key is tight.** Start promptly or use `--hours 20`.
- **Everything under `data/raw/soloq/` is gitignored.** Never commit data,
  never print the key.

## State of the repo (scrape lane only)

- `data/raw/soloq/soloq.db` — 1.3 GB, gitignored, **the authoritative store**,
  local to the Mac. 877k+ done, ~1 failed, pending drains continuously.
- `scripts/scrape_health.sh` — **new, uncommitted.** Per-region rate with a
  `<< WEDGED` flag, PID/power/FATAL/floor in one command.
- No code changes were made to `riot_soloq_scrape.py`. The `count=100`
  pagination gap above is a known, deliberate non-fix.
- Other lanes (model/experiments, column, viz) are untouched by this work —
  see `docs/2026-08-01-START-HERE.md`.

## Suggested order for the next session

1. Let the current run finish (~12:18 ET Aug 2); confirm clean exit and the
   new floor/density with `scripts/scrape_health.sh`.
2. Decide whether a third run is worth it — check whether cycle 1 completed
   this time. If it did, the ladder is fully walked at 150d and further runs
   revert to ~0.3/call territory. If not, one more run still buys real
   history.
3. Do **step 1** (cutoff constants + table rebuild + clear the expcache).
4. Run the three-floor sensitivity gate. Expect to state power limits
   explicitly whatever the result.

---

# Execution addendum — 2026-08-01 ~13:50 ET

Step 1's **code half is done and verified**; its **compute half has not been
run**. The three-floor gate is now a single command. What follows amends the
plan above where executing it proved the plan wrong.

## Correction: "flip the constant" was the wrong shape

The plan said move `SOLOQ_START` in three scripts. Two problems with that:

**There is a fourth site, and it is the one that matters.**
`scripts/soloq_lift_tables.py:43` carried `SOLOQ_COVERAGE_START = "2026-06-08"`.
It was never a filter — the table builder had **no lower bound at all** and
simply took everything before the 07-01 cutoff. That was harmless only while
the db began on 06-08. It no longer does. So:

> **A rebuild today does not produce the 06-08 control. It produces a
> 03-03 table wearing a `"2026-06-08"` label**, because that string was
> hardcoded metadata, not a computed one. Running the "control" arm without
> this fix would have silently contaminated it with 134k older games.

That was the single highest-consequence trap in this lane, and it was not on
the list of two.

**The floor is two knobs, not one.** Which soloq games build the tables, and
which pro decisions get live features, were independent — only one was ever
settable. The sweep the plan asks for is only interpretable if they move
together, so they are now one plumbed value.

## Correction: the risk is recency, not just density

The plan's caveat was thin samples in Mar–Apr. Measured against the db, the
sharper problem is **patch staleness**, because these tables are patch-blind:

| floor | clean games pre-07-01 | distinct patches | oldest patch |
|---|---:|---:|---|
| 2026-06-08 (control) | 224,064 | 3 | 16.11 |
| 2026-05-01 | ~332k | ~6 | 16.9 |
| 2026-03-03 | 358,130 | 10 | 16.4 |

Going to 03-03 adds 60% more games, but ~37% of the table then comes from
patches 1–8 balance changes behind the July pro games it scores. More evidence
about a *different game*. This strengthens the case for the three arms rather
than weakening it — just read the result as recency-vs-volume, not
density-vs-volume.

## What changed in the code

- `soloq_lift_tables.py` — `--floor` (default `2026-06-08`, the control),
  `--db`, `--out`. Records the floor **asked for** and the earliest game
  **actually present**, so the metadata can never lie again. Opens the db
  read-only; the scraper may be mid-run.
- `draft_dataset.py` — `--soloq-floor`, `--soloq-tables`, `--out-suffix`.
  **Refuses to run** if the floor it was asked for and the table's own recorded
  floor disagree; fails in 0.3s, before any loading.
- `experiment_v12_rung2_gbm.py` — `--dataset`, `--soloq-tables`, `--out`, and
  the expcache trap is **closed at the key**: the cache now includes a
  blake2b digest of the actual feature matrices (verified to change on a
  single 1e-9 perturbation). Stale reuse is no longer possible rather than
  merely discouraged — so trap 1 above needs no manual cache-clearing ritual.
  Side effect: the 40 existing entries use the old schema and are now inert.
  The first sweep run refits everything.
- `experiment_v12_rung2_outcome.py` — imports the shared floor instead of
  keeping a second hardcoded copy.
- `scripts/sweep_soloq_floor.sh` — **new.** Runs all three arms end to end.

## Canary: the refactor is provably faithful

Rebuilt at `--floor 2026-06-08` from the frozen `soloq-snapshot-20260731.db`
and compared to the production table:

```
clean_games 224064 · blue_wr 0.4777 · 173 champs · 27,648 ctr · 39,023 syn
role_wr 837/837 · ctr 27,648/27,648 · syn 39,023/39,023 exact-equal
```

**All 67,508 table values byte-identical.** The parameterization changes
nothing at the old floor.

## Running it

```bash
bash scripts/sweep_soloq_floor.sh          # freezes its own snapshot, then 3 arms
```

**All three arms must read one frozen snapshot** — the script enforces this by
taking a `VACUUM INTO` copy up front. The scraper adds ~200k games/day; arms
built from the live db hours apart would confound "floor" with "how far the
scrape had got". This was not in the original plan and it is not optional.

## Still open (unchanged by this session)

- The **1,080-decision val set still cannot resolve a sub-1pt top-3 effect.**
  Nothing here fixes that. State the power limit whatever comes back.
- The second scrape run is live and untouched (PID 47318, ends ~12:18 ET Aug 2).
  Nothing above depends on it; the sweep can run on today's db and be re-run
  later against the fuller one.
- **The machine was on battery at 61% for the third run running.** The rule at
  the top of this doc exists because clamshell sleep already cost two runs.

## Sweep executed — see `2026-08-01-soloq-floor-sweep-results.md`

Ran 13:50–14:25 ET, same day. **All three arms null, and best top-3 declines
monotonically as the floor drops** (0.3380 → 0.3324 → 0.3278). The
pre-registered signature of a real effect was "monotone-ish or peak at the
dense floor"; the observed ordering is monotone the *other* way.

The 14.2× live-decision gain this document was written to deliver was
delivered exactly — 1,360 → 19,277, verified against the real pipeline — and
**bought nothing**. So the rung-2 re-run's claim that "the binding constraint
is calendar coverage" is now tested and false.

Practical consequence for the two open items at the top of this doc: **step 2
(is a third scrape run worth it?) is answered "not for this lane."** More
calendar history is precisely what just showed negative value. The next lever
is patch-aware tables or a wider val set, not more scraping.
