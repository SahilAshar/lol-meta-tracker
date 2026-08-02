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
| **Soloq scrape / rung 2** | `docs/2026-07-31-rung2-rerun-results.md` | NO-GO holds on denser tables. Scrape is **running right now** (see below). |

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

## Live state as of 2026-08-01 10:50 ET

**Scrape is running** — `riot_soloq_scrape.py --hours 23 --start-days 150`
under caffeinate (pid 14078). `scrape_status.json` at 10:49:15: **861,778 done**
(euw 285,474 · kr 296,570 · na 279,734), 580 pending, 1 failed. This is up from
the 688,180 in the 07-31 frozen snapshot. Two consequences: the rung-2 follow-up
("re-run when soloq spans a full split") is accumulating toward its condition on
its own, and anything reading `soloq.db` must open it read-only and should
prefer `soloq-snapshot-20260731.db` for reproducibility.

**Committed** (HEAD `dc2b39a`): the full pocket-conditioning arc — v13 NO-GO,
v14 GO, v15 NO-GO — with docs, scripts, and result JSONs. Nothing pushed.

**Uncommitted, by owner** — do not fold these into unrelated commits:

- rung-2 density re-run: `docs/2026-07-31-rung2-rerun-results.md`, modified
  `rung2_gbm.json`, `soloq_lift_tables.json`, `experiment_v12_rung2_gbm.py`
- counter-swing + column: `scripts/analysis_counter_swing.py`,
  `counter_swing.json`, `docs/2026-07-28-counter-swing-analysis.md`,
  `drafts/2026-07-28-lolmetaweekly-edition1-skeleton.md`,
  `docs/2026-08-01-counter-swing-column-handoff.md`
- scrape ops: `scripts/scrape_health.sh`, `logs/`
- `docs/ROADMAP.md` — carries the uncommitted North Star section naming the
  LoLMetaWeekly debut. Another session owns it; never stage it.
- this file

## If you are an agent picking this up cold

1. Read the lane doc for whatever you were asked to do. Do not read all three.
2. Re-read the rails in `state-of-the-repo-handoff` §1 before touching any
   experiment — the frozen EWC test set is never scored again, canaries are
   hard-asserted (baseline 3.5855, meta 3.4623, v14 `ban_only` 3.3911), CPU
   only, and `date` gets run before any date is written into any file.
3. The approvals ledger lives in `state-of-the-repo-handoff` §"Open approvals".
   Sahil has green-lit none of it. Commits are fine; pushes are not.
