# Synergy rung 1c results: the single-scalar duo-lift test — GO

**2026-07-25 ~14:15 ET.** Executes `docs/2026-07-25-synergy-rung1c-spec.md`
after rung 1b's split verdict (slotting GO, counters GO, same-team pair
synergy null at 323k games). Script:
`scripts/experiment_v10_synergy_rung1c.py`; full output:
`data/processed/synergy_rung1c.json`. Same final db (329,703 done games,
closed 2026-07-25), same loader/discipline imported verbatim from 1b.
Runtime 36 s.

## Question

Collapse each team's claimed duo synergy into one train-built scalar (sum
of EB-shrunk lifts over the four priority vectors, blue minus red) and ask
the maximum-power 1-df question: does that single number improve held-out
prediction over champ×role main effects? A same-construction counter
scalar is the positive control.

## A mechanism bug the tripwire caught (and the fix)

The first run died on `fit_eval`'s sweep-edge assertion: A+SYN's val
log-loss rose monotonically in C (0.694 → 0.708 vs A's 0.687), pinning the
sweep at the smallest C. Cause: the scalars are **outcome-derived**, so
inside the train split each game's own result leaks into its own pairs'
lifts; the regression sees an in-sample-inflated feature, overweights it,
and val collapses. (1b never had this failure mode — its pair features
were indicators fitted by the regression, not precomputed scores.)

Fix: train-game scalars are computed **leave-one-out on the pair counts**
(drop the game's own outcome: `n−1` games, `w−won` wins, shrink
`(n−1)/(n−1+200)`). Val/holdout scalars use the full train tables — those
games are disjoint from the tables, so they were never contaminated, and
the holdout gate is untouched by the repair. Residual self-influence
through the champ@role means is O(1/n_combo), negligible, and symmetric
across SYN and CTR; the control gate validates the repaired mechanism
end-to-end. After the fix every model's C optimum was interior (0.03 for
all four).

## Data

Identical to 1b, re-asserted: 323,401 clean games (23 dropped), blue WR
0.4755, 173 champions, chronological 70/15/15 (train → Jul 13, val →
Jul 19, holdout → Jul 25), 605 champ@role columns. Lift tables (train
only): 8.8–11.2k duo pairs per vector, 3.9–8.2k counter matchups per role.
Both scalars nonzero in 100% of games; train sd 0.0499 (SYN), 0.0413 (CTR).

## Result: GO — a thin real synergy layer exists at 1 df

| model | cols | holdout LL | AUC | acc |
|-------|-----:|-----------:|----:|----:|
| A (champ×role mains) | 778 | 0.68873 | .5478 | 53.9% |
| A+SYN | 779 | 0.68867 | .5482 | 54.1% |
| A+CTR | 779 | 0.68811 | .5516 | 54.2% |
| A+SYN+CTR | 780 | 0.68805 | .5519 | 54.2% |

Paired 10k-bootstrap on per-game holdout log-loss (positive = challenger
better):

- **Control gate, A vs A+CTR: PASS** — mean +0.000624, 95% CI
  [+0.000366, +0.000894]. Consistent with 1b's independent B-vs-C
  estimate (+0.000496, CI [+0.000274, +0.000713]); a single scalar
  captures the counter signal that took 4.6k columns in 1b. Mechanism
  validated; the SYN result is interpretable.
- **Primary gate, A vs A+SYN: PASS** — mean +0.000061, 95% CI
  [+0.000016, +0.000105]. Small — a tenth of the counter effect — but
  the CI excludes 0.
- **Interaction check, A+CTR vs A+SYN+CTR: PASS** — mean +0.000060, CI
  [+0.000018, +0.000103]. SYN's contribution is essentially unchanged
  with CTR in the model: the two scalars are orthogonal signals.

## Reconciling with rung 1b's null

No contradiction. 1b's A-vs-B CI [−0.000246, +0.000568] comfortably
contains +0.000061 — an effect this small, spread over ~8k regularized
pair columns, was undetectable in that design (reading 2 of the spec's
two readings). The precise statement is now: **same-team duo synergy is
not learnable as individual pair columns, but its train-split aggregate
carries a thin real signal.** 1b's per-pair falsification stands for any
use that needs pair-level effects (drafting around a specific duo); the
1-df aggregate is real but tiny.

## The U.GG readout

Fitted effects, translated to WR at p≈0.5: the standardized SYN
coefficient is 0.0111 vs CTR's 0.0635. Per point (0.01 WR) of *claimed*
shrunk lift, the realized holdout effect is **+0.056 WR points for SYN vs
+0.385 for CTR** — claimed duo-synergy lift realizes at roughly **6 cents
on the dollar** once champ@role is known, while claimed counter lift
realizes at ~38 cents. A U.GG-style "+3 synergy" duo is worth ~0.17 real
WR points beyond the champs' own role-specific strength.

Holdout decile table of the SYN score (equal-count buckets):

| decile | mean SYN | actual blue WR | A-predicted WR | residual |
|-------:|---------:|---------------:|---------------:|---------:|
| 1 | −0.086 | .4343 | .4400 | −.006 |
| 2 | −0.051 | .4529 | .4546 | −.002 |
| 3 | −0.033 | .4597 | .4610 | −.001 |
| 4 | −0.019 | .4638 | .4673 | −.004 |
| 5 | −0.006 | .4745 | .4730 | +.002 |
| 6 | +0.006 | .4766 | .4770 | −.000 |
| 7 | +0.019 | .4945 | .4822 | +.012 |
| 8 | +0.033 | .4947 | .4893 | +.005 |
| 9 | +0.051 | .4842 | .4956 | −.011 |
| 10 | +0.086 | .5098 | .5106 | −.001 |

The publishable answer to "is a duo-synergy column informative once you
know champ@role": the SYN score's apparent gradient is large (7.6 WR
points, decile 1 → 10) **but actual WR tracks the champ@role-only
prediction almost exactly** — the residual column is within per-bucket
noise (SE ≈ 0.7 pts) and shows no clean monotone trend. The duo-synergy
column *looks* informative because it proxies team strength; its
information beyond role effects is real only in aggregate at ~6% of face
value, invisible at any decile. A counter column, by contrast, would show
genuine residual slope.

## Consequence

- **Hand the SYN scalar to rung 2 as an optional feature** (spec's PASS
  branch): a single train-built aggregate, cheap to compute, orthogonal
  to the counter features. Expectations should stay modest — one tenth
  of the counter effect.
- The pro GBM's existing `pair_syn` feature is **not** a deletion
  candidate (that flag was contingent on a FAIL).
- The roadmap framing shifts from "same-team synergy is dead" to
  "same-team synergy exists only as a thin aggregate — never spend
  per-pair modeling capacity on it."
