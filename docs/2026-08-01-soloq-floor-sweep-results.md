# Soloq coverage-floor sensitivity gate — results

**Run 2026-08-01 ~13:50–14:25 ET.** Executes the three-floor sweep specified in
`docs/2026-08-01-soloq-backfill-handoff.md`. Val-only; the frozen EWC test set
was never loaded. All three arms read one frozen snapshot
(`sweep-snapshot-20260801T135046.db`, 880,381 games, floor 2026-03-03).

## Verdict

**All three arms fail the gate — and the best top-3 declines monotonically as
the floor drops.** The 14.2× increase in live train decisions bought nothing,
and the deepest floor actively harms the additive variant.

| floor | games | patches | live train decisions | best top-3 | Δ vs baseline | gate |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-08 (control) | 265,270 | 3 | 1,360 (4.7%) | 0.3380 (V3) | +1.02 | null |
| 2026-05-01 | 334,093 | 5 | 10,518 (36.3%) | 0.3324 (V3) | +0.46 | null |
| 2026-03-03 | 359,886 | 10 | 19,277 (66.6%) | 0.3278 (V3) | +0.00 | null |

Baseline is identical across all three arms (top-1 0.1324, top-3 0.3278, top-5
0.4389) — it carries no soloq features, so this is a built-in control
confirming the arms differ *only* in the soloq columns.

## The pre-registered signature failed in a readable direction

The handoff pre-registered the shape of a real effect:

> *"If the effect is real it should be monotone-ish or peak at the dense floor;
> if 03-03 underperforms 05-01, the answer is a density-weighted floor, not
> more calendar."*

Observed: **monotone in the opposite direction.** More calendar coverage is
strictly worse at every step. 03-03 underperforms 05-01, which underperforms
the 06-08 control.

This is the recency-vs-volume question resolving against volume. The added
games are real games, but they are old: the 03-03 window spans 10 patches back
to 16.4, and the tables are patch-blind, so a third of the table describes
champion strength from behind up to eight balance changes. Diluting a
16.11–16.13 prior with 16.4–16.10 evidence costs more than the extra rows buy.

## Per-variant detail (ensemble, paired bootstrap on 1,080 val decisions)

| floor | config | top-1 | top-3 | top-5 | Δtop-3 CI95 |
|---|---|---:|---:|---:|---|
| 06-08 | V1 | 0.1213 | 0.3278 | 0.4620 | +0.0000 [−0.0185, +0.0194] |
| | V2 | 0.1222 | 0.3241 | 0.4759 | −0.0037 [−0.0231, +0.0157] |
| | V3 | 0.1231 | **0.3380** | 0.4583 | +0.0102 [−0.0074, +0.0278] |
| 05-01 | V1 | 0.1296 | 0.3287 | 0.4491 | +0.0009 [−0.0176, +0.0194] |
| | V2 | 0.1185 | 0.3250 | 0.4491 | −0.0028 [−0.0222, +0.0167] |
| | V3 | 0.1343 | 0.3324 | 0.4648 | +0.0046 [−0.0148, +0.0231] |
| 03-03 | V1 | 0.1269 | 0.3083 | 0.4611 | **−0.0194 [−0.0370, −0.0009]** |
| | V2 | 0.1287 | 0.3167 | 0.4565 | −0.0111 [−0.0287, +0.0065] |
| | V3 | 0.1352 | 0.3278 | 0.4630 | +0.0000 [−0.0185, +0.0185] |

Two CIs exclude zero in top-3, both at the deepest floor and both pointing at
harm or at V3 rescuing V1:

- **V1 at 03-03: −1.94 pts, CI [−3.70, −0.09].** The additive soloq features
  make the model *worse* once the table reaches back to March.
- **syn increment (V3 vs V1) at 03-03: +1.94 pts, CI [+0.65, +3.33],
  p=0.9978.** `soloq_syn_score` exactly offsets what `soloq_ctr_score` +
  `soloq_role_wr` break. V3 lands back on the baseline, not above it.

## Power — read this before quoting any number above

Unchanged from the rung-2 re-run and **the dominant fact about this sweep**:
the val set is **1,080 decisions**, and it cannot resolve a sub-1pt top-3
effect. This run made 27 comparisons (3 floors × 3 variants × 3 metrics); at
that count roughly one CI excluding zero is expected by chance alone, and two
were observed.

Therefore:

- The **monotone decline is the trustworthy result**, because it is an ordering
  across a pre-registered axis rather than a single significance test, and
  because the baseline is pinned identical across arms.
- The **individual per-variant CIs are not trustworthy** on their own. In
  particular, do not report "soloq features significantly harm the model" on
  the strength of V1 at 03-03.
- A top-5 pattern visible in the 06-08 arm (V2 +3.7 pts) attenuated at 05-01
  (+1.0) and did not reproduce. It was noise.

## What this means for the lane

1. **The backfill did not unblock rung 2.** The re-run's conclusion — *"with
   4.7% live train decisions, the binding constraint is calendar coverage"* —
   is now testable and **false**. Coverage went to 66.6% and the gate did not
   move. Sparsity was not what was muting arm 1's null.
2. **Do not lower `SOLOQ_START` in production.** The default stays
   `2026-06-08`. The floor is now a flag, so this is a one-word change if that
   ever reverses.
3. **A third scrape run is not justified by this lane.** More calendar history
   is the thing that just demonstrated negative value here. (It may still be
   justified for other lanes — this says nothing about them.)
4. **The next lever is the table, not the calendar.** If soloq transfer is to
   work, the tables likely need to stop being patch-blind: recency weighting,
   a patch-decay, or per-patch tables. That is a new rung, not a re-run.
5. **Or the val set.** Every conclusion here is power-limited. Widening
   `VAL_DAYS` is now cheap — val no longer has to sit inside a narrow covered
   window — and would do more for this lane's decidability than any amount of
   additional scraping.

## Reproduction

```bash
bash scripts/sweep_soloq_floor.sh data/raw/soloq/sweep-snapshot-20260801T135046.db
```

Arms: `data/processed/rung2_gbm_floor{20260608,20260501,20260303}.json`,
tables `soloq_lift_tables_floor*.json`, datasets `draft_decisions_floor*.parquet`.
Runtimes 491s / 357s / 392s.

**Canaries that held:** the 06-08 table rebuilt from the Jul-31 snapshot
reproduced the production table with all 67,508 values byte-identical; the
06-08 arm reproduced the handoff's 1,360 live train decisions exactly; the
03-03 arm reproduced its predicted 19,277 exactly; baseline metrics identical
across all three arms; blue-WR assert held on every table build.
