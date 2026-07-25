# Results: rung 2 — transferring role-aware soloq lifts into the pro models

**Executed 2026-07-25 ~14:30 ET per `docs/2026-07-25-rung2-transfer-spec.md`;
updated ~14:50 ET to fold in rung 1c's GO
(`docs/2026-07-25-synergy-rung1c-results.md`, commit `72df364`, run in a
parallel session). Verdict: NO-GO — no arm/variant passed its gate.** All
nulls land inside their pre-registered readings: arm 1 is a *sparsity-muted*
null (only 11% of train decisions postdate soloq coverage), arm 2 is an
*underpowered* null (the gate's CI is ~5× wider than the effect size the
transfer could plausibly carry). Neither arm falsified the transfer payload —
the payload (champ×role strength, lane counters, and per 1c the thin
aggregate duo-synergy layer) remains real in soloq; what failed is detecting
its marginal value in the pro models on the current windows. The most
suggestive texture in the whole rung is SYN-flavored: adding the duo-lift
aggregate is the only change that nudged the GBM above baseline.

Scripts: `scripts/experiment_v12_rung2_gbm.py`,
`scripts/experiment_v12_rung2_outcome.py`, shared tables in
`scripts/soloq_lift_tables.py`. Outputs: `data/processed/rung2_gbm.json`,
`data/processed/rung2_outcome.json`,
`data/processed/soloq_lift_tables.json`.

## Rung 1c fold-in (PASS branch)

1c passed both gates mid-session: the 1-df duo-synergy scalar is real
(+0.000061 holdout LL, CI [+0.000016, +0.000105]), orthogonal to the counter
scalar, ~1/10 the counter effect, and real *only as an aggregate* — 1b's
per-pair falsification stands. Consequences applied here:

- The SYN scalar joined both arms as the optional feature (arm 1: V3 with
  `soloq_syn_score`; arm 2: model `M2a_soloq_syn`).
- The pro GBM's `pair_syn` is **not** a deletion candidate (that flag was
  contingent on a 1c FAIL).
- **No leave-one-out anywhere in rung 2:** 1c's LOO repair applies only when
  scoring games *inside* the table sample. Every pro game rung 2 scores is
  disjoint from the soloq games the tables are built from, so full-count
  lifts are correct. Stated in both provenance blocks.

## Shared plumbing (as specced)

- **One static lift table**, soloq games with `game_creation` <
  2026-07-01 00:00 UTC (the GBM's pro val window start): 124,731 clean games
  (drops: 9 bad-roles), blue WR 0.4754 (matches the 1b anchor), 173 champs,
  816 champ@role combos, 23,717 counter pairs, 33,546 duo pairs over the
  four priority vectors. Construction identical to 1b's D table / the 1c
  tables: champ@role WR EB-shrunk toward champ overall (prior 200),
  antisymmetric same-role counter lift shrunk 200, slot-ordered duo lift
  shrunk 200, unobserved → 0. Used by both arms.
- **Name bridge:** 100% coverage both arms — 150/150 champs in the 2026 GBM
  dataset, 168/168 in the 2024–26 outcome vocab. Zero misses.
- **Role bridge:** all 10,296 pro (game, side) groups carry clean 5-role
  position sets; no fallback needed.

## Arm 1 — draft GBM (mimic): **gate FAIL (all three variants)**

`draft_dataset.py` regenerated with `soloq_role_wr` (role-aware strength
prior: expected champ@role WR under P(role) ∝ candidate role-shares ×
ref-team open-role probs), `soloq_ctr_score` (expected same-role counter
lift vs the other side's locked picks, locked roles marginalized by the same
permutation enumeration as `role_open_probs`), and post-1c
`soloq_syn_score` (expected priority-vector duo lift vs the *reference*
team's locked picks, candidate in either slot, same weighting). Decisions
before 2026-06-08 hold neutral values — the table cannot causally inform
them. **Pre-existing columns verified byte-identical** on all 3,941,356
pre-cutoff rows at each rebuild (26 columns checked on the final one).

Baseline refit on this machine (stored v0.7 was Linux-trained) reproduced
the stored val numbers on the nose: ensemble top-1 .1324 / top-3 .3278
(top-5 .4389 vs stored .4407 — platform-level wobble).

Ensemble val (1,080 decisions), paired bootstrap deltas vs baseline:

| variant | top-1 | top-3 (gate) | top-5 |
|---|---|---|---|
| V1 (+ctr, +role_wr) | .1278, Δ −0.46pt [−1.9, +1.0] | .3213, **Δ −0.65pt [−2.5, +1.2] → FAIL** | .4537, Δ +1.48pt [−0.5, +3.4] |
| V2 (pair_ctr → soloq) | .1324, Δ 0.0 [−1.5, +1.6] | .3241, **Δ −0.37pt [−2.3, +1.6] → FAIL** | .4546, Δ +1.57pt [−0.4, +3.5] |
| V3 (V1 + syn_score) | .1389, Δ +0.65pt [−0.9, +2.2] | .3296, **Δ +0.19pt [−1.7, +2.0] → FAIL** | .4556, Δ +1.67pt [−0.2, +3.5] |

SYN increment (V3 vs V1, descriptive — not a pre-registered gate): top-3
Δ +0.83pt [−0.4, +2.0] (p≈0.89); top-1 Δ +1.11pt **[+0.3, +2.0]** — the CI
excludes 0, but this comparison was not the declared gate and gets no
verdict weight. Read it as: the duo-lift aggregate is the one addition that
moved the ensemble in the right direction, consistent in sign with 1c's
soloq-side finding.

**Pre-registered reading (the only one allowed for this shape of null):**
ambiguous/sparsity-muted, *not* falsified. 89% of train rows carry neutral
placeholder values; the GBM had roughly three weeks of live-feature train
data to learn from. The recurring ungated top-5 lift (+1.5–1.7pts, p≈0.93+
in all three variants) is consistent with a weak real signal the model can
only express coarsely. Follow-up: **re-run when soloq spans a full split**
— not a rerun of the same window.

Texture worth keeping: the soloq counter score is essentially uncorrelated
with the pro `pair_ctr` (r = −0.006 on live rows) — new information, not a
re-estimate; and candidates actually picked/banned carry ~5× the mean
counter lift of non-picked candidates (+0.0014 vs +0.0003), so the raw
feature tracks pro behavior — the gate failure is about marginal value
inside the ensemble on this window, not about the feature being inert.

## Arm 2 — outcome head (coach): **gate FAIL (both variants)**

v0.9 protocol reproduced exactly — same splits (train 4,099 / val 625 /
holdout 374, frozen EWC excluded), stored Elo K=10, C swept on val (all
models chose 0.01), one holdout look. M2a reproduced its stored holdout LL
0.60235 exactly. CTR, ROLE_WR, SYN standardized on train; features are
**patch-blind** as pre-registered (the static table is applied to every game
regardless of date — champ meta shift is the noise term).

- **M2a vs M2a+soloq (CTR, ROLE_WR):** holdout LL 0.60230 vs 0.60235;
  Δ = +0.00004, CI **[−0.0024, +0.0025]** → FAIL.
- **M2a vs M2a+soloq+SYN:** holdout LL 0.60275; Δ = −0.0004,
  CI [−0.0031, +0.0023] → FAIL.
- **SYN increment** (M2a_soloq vs M2a_soloq_syn): Δ = −0.0004,
  CI [−0.0013, +0.0004] — null, as expected: the soloq-side SYN effect
  (+0.00006 LL) is two orders of magnitude below this test's resolution.
- **Secondary slice** (holdout ≥ 2026-06-08, 122 games; reported, not
  gated): all three comparisons null (CTR/ROLE_WR Δ = −0.0002,
  CI [−0.0045, +0.0039]) — not the "positive-where-aligned" pattern.
- Fitted coefficients tiny (per SD: CTR +0.028, ROLE_WR +0.060,
  SYN +0.021).

**Pre-registered reading:** the timing caveat allowed "misaligned window"
for the pre-06-08 holdout; what the numbers add is a power statement. Rung
1b/1c measured the counter effect in soloq at +0.0003 to +0.0007 LL per
game and the synergy aggregate at +0.00006 — at 374 holdout games the CI
here is ±0.0025, and on the aligned slice ±0.004. This gate can only detect
transfer if the pro-side effect is ~5× the soloq-side effect; it isn't, or
it's smaller. What we *can* now say: soloq comp features do not improve pro
win prediction by more than ~0.0025 LL on this holdout. A soloq-magnitude
transfer remains untestable at pro sample sizes — an underpowered null, not
a falsification, and also not a license to keep re-testing the same window.

## Verdict and what it gates

- **NO-GO.** No arm/variant passed; rung 2 does not confirm the transfer on
  current windows.
- **Rung 3 (counterfactual critic) stays closed.** The spec made rung 3
  contingent on a GO *plus* the mimic-vs-coach conversation; there is no GO.
  The mimic-vs-coach conversation itself remains unhad on the ROADMAP and is
  still worth having on its own terms — arm 1's top-5/SYN texture and arm
  2's power ceiling are both inputs to it.
- **`pair_syn` stays** in the pro GBM (1c PASS resolved the contingent
  deletion flag). The correct framing after 1b+1c: same-team synergy is a
  thin real *aggregate* layer — ~6 cents on the dollar of claimed lift —
  and never worth per-pair modeling capacity.
- The soloq db is closed out (329,703 done games, coverage 2026-06-08 →
  07-25). Both follow-ups below require *more calendar*, not more compute.

## Follow-ups (in order of leverage)

1. **Arm 1 re-run when soloq spans a full split** — resume the scrape next
   window so live features cover most of the train era, then rerun
   `experiment_v12_rung2_gbm.py` unchanged. The parquet plumbing, all three
   variants, and the gate are already in place. The V3/SYN top-1 texture is
   the specific thing a properly-covered re-run would confirm or kill.
2. **Arm 2 re-split only after coverage grows** — a holdout entirely inside
   soloq coverage with ~10× the games is the first version of this test
   with real power at soloq-magnitude effects.

## Provenance

Both output JSONs carry full provenance blocks mirroring 1b (splits,
C-grids, 10k bootstrap resamples, soloq table cutoff + patch mix
16.11–16.13, bridge coverage, per-arm timing caveats, byte-identity check,
no-LOO note, 1c fold-in note). Frozen EWC test set: untouched by every
scorer in both arms. Runtime: arm 2 12 s; arm 1 354 s initial + 128 s for
V3 (cached families reused) + two ~4 min dataset rebuilds. Platform: local
darwin (libomp installed for lightgbm); stored v0.7 comparisons annotated
accordingly.
