# Results: rung 2 arm 1 re-run on denser soloq tables (same cutoff)

**Executed 2026-07-31 ~13:40 ET. Verdict: NO-GO holds — no variant passed
the pre-registered top-3 gate.** This is a density re-run, not a new window:
the soloq lift tables were rebuilt from a 1.8× larger sample *inside the same
legal window*, cutoff untouched. Follow-up #1 of
`docs/2026-07-25-rung2-transfer-results.md` asked for a re-run "when soloq
spans a full split"; that condition is **not** met and this run does not
claim it. What is answered here is the narrower question density can answer.

Scripts: `scripts/soloq_lift_tables.py` (unmodified), `scripts/draft_dataset.py`
(unmodified), `scripts/experiment_v12_rung2_gbm.py` (one check widened, below).
Outputs: `data/processed/rung2_gbm.json`, `data/processed/soloq_lift_tables.json`.

## Provenance

| item | value |
|---|---|
| soloq source | `data/raw/soloq/soloq-snapshot-20260731.db` (frozen point-in-time copy, 688,180 done games) |
| why a snapshot | a scrape may write to `soloq.db` concurrently; the snapshot makes the build reproducible. `soloq_lift_tables.DB` was overridden at call time — the file was not edited |
| table cutoff | `2026-07-01T00:00:00Z` (`CUTOFF_MS = 1_782_864_000_000`) — **deliberately held, not advanced** |
| clean games in table | **224,064** (224,076 rows pre-cutoff in the snapshot, 12 dropped `bad_roles`) |
| previous build | 124,731 clean (9 dropped) → **+99,333 games, 1.796×** |
| blue WR | 0.4777 (prev 0.4754; 1b anchor 0.4755, assertion band ±0.01 — passes) |
| patch mix | 16.11 3,120 · 16.12 130,598 · 16.13 90,346 (prev 3,120 / 74,814 / 46,797) |
| bridge coverage | 150/150 pro champs matched, zero misses |
| val decisions | 1,080 (spec anchor asserted) · cutoff `2026-07-15 09:41:28` · VAL_DAYS 14 |
| bootstrap | 10,000 resamples, paired per decision, seed 20260725 |
| frozen EWC test | never loaded into any scorer |
| runtime | 547 s (20 fits, no cache reuse) |

### Why the cutoff was held

`CUTOFF_MS` is pinned to the pro val window start, which is *derived* from
pro data (first EWC main-event game − `VAL_DAYS`), not chosen. Per the
look-ahead rule (`docs/2026-07-25-rung2-transfer-spec.md:40`), a table
scoring a pro decision may only contain soloq games strictly before that
decision. **Only the soloq side grew — no new pro games — so advancing the
cutoff would fold post-hoc soloq information into val-window pro decisions
and silently invalidate the result.** The legitimate win is density inside
the existing legal window, and that is all that was taken.

### Table density delta

| | 124,731-game build | 224,064-game build | Δ |
|---|---|---|---|
| champs | 173 | 173 | — |
| champ@role combos | 816 | 837 | +21 |
| counter pairs | 23,717 | 27,648 | +3,931 (+16.6%) |
| duo pairs | 33,546 | 39,023 | +5,477 (+16.3%) |
| mean \|ctr lift\|, pairs in both | 0.005721 | 0.006843 | **+19.6%** |
| mean \|syn lift\|, pairs in both | 0.005965 | 0.007025 | **+17.8%** |

The old key set is a strict subset of the new one. On shared pairs the lifts
correlate 0.78 (ctr) / 0.77 (syn) with the old build and are ~19% larger in
magnitude — exactly the signature of less EB shrinkage toward zero at
unchanged prior (200 games), not of a different signal.

## Result: gate FAIL on all three variants

Ensemble val (1,080 decisions), paired bootstrap vs the refit baseline. The
baseline refit **reproduced the 2026-07-25 run bit-for-bit** (top-1 .1324 /
top-3 .3278 / top-5 .4389), which is the expected consequence of the pinned
columns being byte-identical — it makes the two builds properly paired.

| variant | top-1 | top-3 (gate) | top-5 |
|---|---|---|---|
| V1 (+ctr, +role_wr) | .1241, Δ −0.83pt [−2.50, +0.74] | .3278, **Δ +0.00pt [−1.76, +1.76] → FAIL** | .4713, Δ +3.24pt [+1.30, +5.19] |
| V2 (pair_ctr → soloq) | .1231, Δ −0.93pt [−2.50, +0.65] | .3278, **Δ +0.00pt [−1.85, +1.85] → FAIL** | .4565, Δ +1.76pt [−0.09, +3.61] |
| V3 (V1 + syn_score) | .1259, Δ −0.65pt [−2.31, +1.02] | .3361, **Δ +0.83pt [−0.93, +2.50] → FAIL** | .4676, Δ +2.87pt [+1.02, +4.72] |

Per-family (ensemble is the equal-weight rank blend of the two):

| variant | clf top-1/3/5 | ranker top-1/3/5 |
|---|---|---|
| baseline | .1259 / .3278 / .4472 | .1389 / .3278 / .4333 |
| V1 | .1120 / .3083 / .4324 | .1296 / .3269 / .4546 |
| V2 | .1120 / .3093 / .4370 | .1269 / .3167 / .4583 |
| V3 | .1139 / .3213 / .4389 | .1185 / .3352 / .4759 |

The families disagree in sign at top-5: the ranker gains (baseline .4333 →
V3 .4759, +4.3pt) while the classifier loses (.4472 → .4389). The ensemble's
top-5 gain is carried almost entirely by the lambdarank family.

## The two informative questions (effect size, CI width)

Pass/fail did not move. These did:

| comparison | 124,731 build | 224,064 build | CI width |
|---|---|---|---|
| V1 top-3 | −0.65pt [−2.50, +1.20] | +0.00pt [−1.76, +1.76] | 3.70 → 3.52 |
| V2 top-3 | −0.37pt [−2.31, +1.57] | +0.00pt [−1.85, +1.85] | 3.89 → 3.70 |
| V3 top-3 | +0.19pt [−1.67, +2.04] | +0.83pt [−0.93, +2.50] | 3.70 → 3.43 |

**Effect size moved, consistently upward, on the gated metric.** All three
variants improved their top-3 delta (+0.65, +0.37, +0.64 pt respectively);
V1 and V2 moved from mildly negative to exactly zero, V3 from +0.19 to
+0.83. The direction is what denser tables predict. The magnitude is not
close to clearing the gate — V3, the best variant, sits at roughly a third
of the width of its own CI.

**The CI barely narrowed (~5–7%), and that is structural.** Bootstrap width
here is set by the 1,080 val decisions, not by soloq table density. Denser
tables sharpen the *feature values*; they do nothing to the *sample size of
the test*. No amount of soloq scraping will narrow this interval
meaningfully — only more val decisions will. That is worth stating plainly
because it bounds what any future density re-run can buy: this specific test
cannot resolve a sub-1pt top-3 effect at n=1,080 regardless of how good the
tables get.

### Ungated observation: top-5 now clears zero

V1's top-5 delta is +3.24pt with CI [+1.30, +5.19] and V3's is +2.87pt
[+1.02, +4.72] — both exclude 0, where on the previous build the same
comparison was +1.48pt [−0.46, +3.43] and did not. The prior doc flagged a
"recurring ungated top-5 lift (+1.5–1.7pts, p≈0.93+)" as consistent with a
weak real signal; on denser tables that lift roughly doubled and became
statistically distinguishable from zero.

**This gets no verdict weight.** Top-5 is not the pre-registered gate, and
promoting it after seeing the numbers is exactly the move the pre-registration
exists to prevent. Recording it as the strongest surviving texture: the soloq
features appear to help the model get the right champion *into a shortlist*
without helping it rank within that shortlist.

### The prior run's SYN texture did not replicate

The 2026-07-25 doc's "most suggestive texture in the whole rung" was the
V3-vs-V1 SYN increment at top-1: +1.11pt, CI [+0.28, +2.04], excluding 0.
On denser tables that collapses to **+0.19pt [−0.83, +1.20]** (p 0.995 →
0.608). The top-3 increment is unchanged (+0.83pt [−0.37, +2.04]).

Denser tables are a strictly better estimate of the same quantity, so the
honest reading is that the top-1 SYN result was noise — an ungated
comparison that happened to clear zero once. It should not be carried
forward as evidence.

## Correction: train-side sparsity is worse than recorded

The prior doc reported "only 11% of train decisions postdate soloq
coverage" / "89% of train rows carry neutral placeholder values". That
figure is computed over the **full parquet**, which includes the val split
(100% live by construction) and the frozen EWC test rows. Measured on the
train split alone:

| split | rows | decisions | ≥ 2026-06-08 |
|---|---|---|---|
| train | 3,805,816 | 28,936 | **4.6% of rows, 4.7% of decisions** |
| val | 135,540 | 1,080 | 100% |
| train+val | 3,941,356 | 30,016 | 7.9% / 8.1% |
| full parquet | 4,076,256 | 31,016 | 11.0% / 11.1% |

The GBM has **1,360 live train decisions**, not ~3,200. The sparsity
diagnosis in the prior doc is not merely intact — it is about 2.4× more
severe than the number it cited. This strengthens, rather than weakens, the
"sparsity-muted null" reading, and it is the single best argument that the
real follow-up remains calendar coverage, not table density. This re-run
confirms density was never the binding constraint.

Feature-level behavior is otherwise stable and non-inert: `soloq_ctr_score`
remains essentially uncorrelated with the pro `pair_ctr` on live rows
(r = −0.0038, prev −0.0076 — new information, not a re-estimate), and
picked/banned candidates still carry a higher mean counter lift than
non-picked ones (+0.001471 vs +0.000463; the ratio compressed from 4.9× to
3.2× because the non-picked mean rose faster under reduced shrinkage).

## Code change (one, and why it was necessary)

`verify_backup` in `scripts/experiment_v12_rung2_gbm.py` compared *every*
column in the backup. On 2026-07-25 the backup predated the soloq columns,
so it only ever covered non-soloq ones. This time the backup already carried
`soloq_role_wr` / `soloq_ctr_score` / `soloq_syn_score` — the three columns
the rebuild is *supposed* to change — so the check aborted the run on its
own intended outcome.

The check was narrowed to its documented intent: the three table-derived
columns are exempted by name (`SOLOQ_FEATURES`), **all 24 other columns are
still asserted byte-identical** across all 3,941,356 pre-cutoff rows, along
with row count and row identity. The exempted columns' drift is recorded
descriptively in provenance instead (`soloq_column_shift`: corr with backup
0.932 / 0.798 / 0.784, rows changed 7.9% / 4.7% / 4.3%). When the backup
predates the soloq columns the exempt set is empty and the check behaves
exactly as before. The guarantee the gate depends on — nothing but the soloq
features moved — is fully preserved and now explicitly reported.

**Operational note for the next re-run:** `data/processed/expcache_rung2gbm/`
keys on `(config, family, seed)` with **no data hash**. A re-run against
rebuilt tables will silently reuse stale fits and reproduce the previous
numbers exactly. The stale cache was moved to
`data/processed/expcache_rung2gbm.pre-rebuild-20260725/` before this run;
all 20 fits are fresh. Clear that directory on any future rebuild, or add a
data hash to the cache key.

## Verdict and what it gates

- **NO-GO holds.** Denser tables did not flip arm 1. The prior verdict
  stands unamended.
- **The sparsity-muted reading is confirmed, and sharpened.** The prior null
  was diagnosed as train-time sparsity rather than falsified transfer; this
  run tested the competing explanation (noisy tables) and it lost — the
  tables got 1.8× more data and ~19% larger lifts, and the gate did not move
  materially. With 4.7% live train decisions, the binding constraint is
  calendar coverage.
- **Follow-up #1 is unchanged and still open:** re-run when soloq spans a
  full split. This run does not consume it. What has changed is the
  expectation attached to it — the CI-width analysis above shows the
  1,080-decision val set cannot resolve a sub-1pt top-3 effect, so a
  coverage re-run should plan on a larger val window or an explicitly
  underpowered result.
- **Rung 3 stays closed** (contingent on a GO plus the unhad mimic-vs-coach
  conversation). The top-5 texture is a further input to that conversation,
  not a substitute for it.
- Arm 2 (outcome head) was **not** re-run — it was not in scope for this
  task, and its null was diagnosed as an underpowered holdout (374 games),
  which denser tables cannot fix either.
