# START HERE — repo entry point (2026-08-01)

**Written 2026-08-01 ~10:50 ET. This file is an index, not a handoff.** Three
lanes are running in parallel, each with its own detailed handoff written the
same morning. Nothing below is restated from them — read the lane doc for
depth. What this file adds is (a) which door to walk through, (b) two
cross-lane facts that no single lane owns, and (c) live state as of now.

## The three lanes and their entry points

| Lane | Read this first | State |
|---|---|---|
| **Model / experiments** | `docs/2026-08-01-state-of-the-repo-handoff.md` | Pocket-conditioning arc closed and committed through `dc2b39a`. v14 `ban_only` is the live GO. |
| **Column / LoLMetaWeekly** | `docs/2026-08-01-counter-swing-column-handoff.md` | Debut targeted **Aug 3–4**. Has a hard blocker (no inference path) and 5 open decisions. |
| **Soloq scrape / rung 2** | `docs/2026-08-01-soloq-floor-sweep-results.md` | **Closed 08-02.** Floor sweep ran: all three arms null, more calendar coverage is strictly worse. Floor stays 06-08. Scrape finished clean. |

The viz/artifact lane has its own handoff ladder (`Handoff v8`, commit
`18a4a48`) and is not covered here.

## Cross-lane fact 1 — the port bundle and the column blocker are the same problem

Neither lane doc connects these, because each is true on its own side:

- The column's entire differentiator is **model-graded predictions** (Ledger
  B). It is blocked because v0.8.1's weights are never persisted — there is no
  inference entry point in the repo. (`counter-swing-column-handoff` §4.)
- The model lane has a **proven-but-unported stack** — meta injection, v14
  `ban_only` pocket injection, phase-aware blend weights — none of it in
  `train_draft_model_v08.py`. (`state-of-the-repo-handoff` §"Where the model
  actually stands".)

Put together: **whatever gets frozen as the column's shipping artifact defines
what the public ledger is grading for the next 10–12 weeks.** Freezing the
current v0.8.1 means publishing weekly grades for a configuration already known
to be superseded, and re-freezing later breaks ledger comparability across
editions. Freezing the ported stack means the port and the blend re-sweep move
onto the debut's critical path.

That is a real trade with a two-day fuse, and it is a decision for Sahil, not
a task to pick up. It is not recorded anywhere else.

## Cross-lane fact 2 — the stale blend weight is sharper than recorded

The model handoff notes the Mission 1 blend weights are "conditional on which
transformer ships" and queues a re-sweep behind the v16 combination rung. The
specific number that makes this urgent is not stated there:

| blend cell | transformer weight | source |
|---|---|---|
| picks | 0.5 (production 0.75) | `blend_phase_sweep.json` |
| **bans B1–3** | **0.0 — GBM only** | ← the stale one |
| bans B4–5 | 0.75 | |

Production is `BLEND_W_PICKS = 0.75` / `BLEND_W_BANS = 0.0`
(`scripts/train_draft_model_v08.py:55-56`).

The B1–3 weight of 0.0 was selected on 2026-07-25 against the *meta* ensemble
(`blend_phase_sweep.py:44` hardcodes `CACHE_DIR = expcache_timesignal_blind`).
It was the correct call for that model: it scored 5.3 top-1 on B1–3 META and
1.7 on B1–3 POCKET. v14 `ban_only` scores **15.3 and 7.0** in those same cells
(`docs/2026-07-28-slot-gated-pocket-results.md`), and its solo ban top-1 of
13.59 is now at parity with the GBM's 13.3.

So the blend discards the transformer entirely in the one phase where the new
model's whole improvement lives. The sweep is ~30 minutes local and needs the
cache pointed at `expcache_slotgate/` (v14's per-seed val prob npz) instead of
the hardcoded blind-test dir. **Sequencing note:** the model handoff puts this
after v16; if the column freezes an artifact first, the re-sweep has to come
before that freeze regardless of v16, or the frozen blend is knowingly wrong.

## Live state as of 2026-08-02 19:50 ET

**Nothing is running.** No scraper, no experiment, no background job.

**The tree is clean and everything is committed** on `main` — the four commits
below plus the merge. Still **nothing pushed**; that approval has never been
given.

```
b76399f  Soloq floor sweep: all three arms null
4b18c8f  Parameterize the soloq coverage floor; close the expcache trap
4b41029  Land parallel sessions' uncommitted artifacts
c8ed3e9  Ignore experiment fit caches (~350MB regenerable .npy)
```

The "uncommitted, by owner" list that used to live here is gone because it is
empty. The parallel lanes' artifacts (counter-swing, rung-2 rerun, Edition 1
skeleton, `scrape_health.sh`, ROADMAP, logs) were committed in `4b41029`,
deliberately kept in their own commit so authorship stays legible.

**Scrape finished clean 2026-08-02 12:18 ET** — all three regions `finished`,
0 FATAL, 174,787 new matches on 196,525 calls. The db is **1,051,476 games**,
floor 2026-03-03, 123 pending/failed. Cycle 1 did *not* complete (kr 60%,
na 56%, euw 80% of the ladder walked), but see below — that no longer argues
for another run.

## The soloq lane's result, in one paragraph

The 07-31/08-01 backfill did exactly what it promised — live train decisions
went 1,360 → 19,277, verified against the real pipeline — and it **bought
nothing**. All three floor arms are null, and best top-3 *declines*
monotonically as the floor drops (0.3380 → 0.3324 → 0.3278). The rung-2
rerun's claim that "the binding constraint is calendar coverage" is now tested
and false. The cause is recency, not density: the tables are patch-blind, and
the 03-03 window spans 10 patches back to 16.4. Full detail and the power
caveat in `docs/2026-08-01-soloq-floor-sweep-results.md`.

**Consequences already applied:** production floor stays `2026-06-08` (now a
flag, one word to change); a third backfill run is *not* justified for this
lane.

## Where to pick up — the per-patch direction

Sahil's call at session close, not yet designed or scheduled:

> Per-patch lift tables, the way op.gg / u.gg do it — because the game changes
> every two weeks and a patch-blind table cannot represent that.

The scrape's *purpose* changes under this design, and that is the thing to
internalise before touching it. Under a patch-blind global table, more games
means more dilution. Under per-patch tables the binding resource is **games
inside the current patch**, which resets to zero every ~2 weeks. Feasibility
is already there for the patches that matter:

| patch | games | window |
|---|---:|---|
| 16.4–16.9 | 177 → 15,810 | Mar–May (too thin to use) |
| 16.10 | 26,565 | May 12–28 |
| 16.11 | 42,137 | May 27–Jun 10 |
| 16.12 | 155,636 | Jun 9–24 |
| 16.13 | 331,721 | Jun 23–Jul 15 |
| 16.14 | 233,050 | Jul 14–29 |
| **16.15** | **53,433** | Jul 29 – live |

This also explains the sweep result completely: the backfill's games landed in
16.4–16.11, patches individually too thin to build a table from *and* too old
to describe the July pro games being scored. Worst of both.

Three things follow, none decided:

1. **The next scrape run wants a different shape.** `--start-days 150` now
   buys the part that just demonstrated negative value. A forward window
   matched to patch length (~14d) collects the same useful games for far
   fewer calls.
2. **Per-patch has a cold-start problem with a clean fix.** Day 1 of a patch
   has no games. EB-shrink patch *N* toward patch *N−1* instead of toward the
   champion's global WR: usable table immediately, sharpens as the patch
   fills, degrades toward "last patch's meta" which is the right prior anyway.
3. **The val set is the cheaper unlock.** Every conclusion in the sweep is
   power-limited at 1,080 decisions. Per-patch tables could genuinely help and
   this gate still would not show it. Widening `VAL_DAYS` is cheap now that
   val no longer has to sit inside a narrow covered window.

## Tooling that now exists (did not on 08-01)

- `bash scripts/sweep_soloq_floor.sh [snapshot.db]` — full three-arm sweep,
  freezes its own snapshot, ~20 min. All arms read ONE snapshot by
  construction; do not "fix" that.
- `soloq_lift_tables.py --floor/--db/--out`, `draft_dataset.py
  --soloq-floor/--soloq-tables/--out-suffix`, `experiment_v12_rung2_gbm.py
  --dataset/--soloq-tables/--out`.
- `draft_dataset.py` **refuses to run** if its floor and the table's recorded
  floor disagree (fails in 0.3s). The GBM fit cache is keyed by a blake2b
  digest of the actual feature matrices, so stale reuse is impossible — the
  old cache-clearing ritual is obsolete.

## If you are an agent picking this up cold

1. Read the lane doc for whatever you were asked to do. Do not read all three.
2. Re-read the rails in `state-of-the-repo-handoff` §1 before touching any
   experiment — the frozen EWC test set is never scored again, canaries are
   hard-asserted (baseline 3.5855, meta 3.4623, v14 `ban_only` 3.3911), CPU
   only, and `date` gets run before any date is written into any file.
3. The approvals ledger lives in `state-of-the-repo-handoff` §"Open approvals".
   Sahil has green-lit none of it. Commits are fine; pushes are not.
