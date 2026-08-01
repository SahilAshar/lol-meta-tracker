# Results: val-only transformer capacity sweep — NO-GO, but the incumbent shape is wrong

**Run 2026-07-28 ~10:25–13:05 ET. Script:
`scripts/experiment_v15_capacity_sweep.py` (pre-registration is in its module
docstring, written before the sweep ran). Raw:
`data/processed/capacity_sweep.json`. Run on top of the `ban_only` injection
that GOed earlier today (`docs/2026-07-28-slot-gated-pocket-results.md`), per
the session brief. Val only — the frozen EWC July-2026 test set was not
touched.**

## Verdict

| Stage | Result |
|---|---|
| Stage 1 sweep (18 configs × 3 seeds) | winner **`d128x6L_do0.2`** on mean val loss, outright (nothing inside the 0.005 tie band) |
| Stage 2 confirm (winner vs incumbent, 5 paired seeds) | **NO-GO** — d_top1 +0.19 pts, threshold was +1.0 |

The winner reliably improves *loss* (+0.048, 5/5 seeds, bootstrap 95% CI
[0.034, 0.066]) but not the metric that matters: mean d_top1 +0.19 pts, with
bans −0.07 and picks +0.44. Both guards passed; the primary did not. Nothing
is promoted.

Harness integrity: the incumbent shape at `--arm ban_only` seed 16 reproduced
v14's cached run exactly (3.3911, ban 13.70, pick 18.52), which is a stronger
canary than the 3.5855 baseline number because it also pins the feature tensor
and the injection code. Split invariants asserted on every run.

## The sweep table (3 seeds each, sorted by mean val loss)

| tag | params | loss | top-1 | ban top-1 | pick top-1 | epochs |
|---|---|---|---|---|---|---|
| d128x6L_do0.2 | 871k | **3.3393** | 15.59 | 13.70 | 17.47 | 72.7 |
| d192x4L_do0.2 | 1.34M | 3.3445 | 16.36 | 13.95 | 18.76 | 53.3 |
| d192x6L_do0.2 | 1.93M | 3.3453 | 15.34 | 13.77 | 16.91 | 49.0 |
| d128x4L_do0.2 | 607k | 3.3454 | **16.76** | 15.37 | 18.15 | 65.7 |
| d128x4L_do0.1 | 607k | 3.3457 | 15.40 | 13.76 | 17.04 | 59.7 |
| d192x3L_do0.2 | 1.04M | 3.3690 | 15.31 | 13.45 | 17.16 | 53.3 |
| d128x3L_do0.2 | 474k | 3.3707 | 15.92 | 15.19 | 16.67 | 65.3 |
| d192x3L_do0.1 | 1.04M | 3.3759 | 14.66 | 13.21 | 16.11 | 41.7 |
| d256x3L_do0.2 | 1.83M | 3.3776 | 15.56 | 13.46 | 17.65 | 37.0 |
| d256x4L_do0.2 | 2.36M | 3.3804 | 15.34 | 13.39 | 17.28 | 46.0 |
| d256x6L_do0.2 | 3.41M | 3.3841 | 15.31 | 14.26 | 16.36 | 40.3 |
| d128x6L_do0.1 | 871k | 3.3863 | 15.25 | 13.15 | 17.35 | 51.7 |
| **d192x4L_do0.1 (incumbent)** | 1.34M | **3.3923** | 14.66 | 12.78 | 16.54 | 39.7 |
| d128x3L_do0.1 | 474k | 3.4046 | 15.74 | 14.75 | 16.73 | 41.3 |
| d192x6L_do0.1 | 1.93M | 3.4052 | 14.78 | 13.52 | 16.05 | 38.0 |
| d256x4L_do0.1 | 2.36M | 3.4091 | 15.31 | 14.75 | 15.86 | 29.7 |
| d256x3L_do0.1 | 1.83M | 3.4258 | 15.03 | 14.07 | 15.99 | 31.3 |
| d256x6L_do0.1 | 3.41M | 3.4323 | 14.10 | 13.33 | 14.88 | 29.3 |

### What the sweep actually says

1. **Dropout 0.2 beats dropout 0.1 in all nine config pairs. No exceptions.**
   That is by far the most consistent effect here — bigger and more reliable
   than any width or depth effect — and it is the one result from this sweep
   worth acting on. Margins run from 0.0003 (d128x4L, a tie) to 0.048
   (d192x4L, the incumbent's own shape).
2. **The incumbent is a bad config.** `d192x4L_do0.1` ranks **13th of 18** on
   loss and is tied for worst on top-1 (14.66). The shape has been fixed since
   v0.7 and was never swept; on this evidence that was costing real
   performance. Changing one number — dropout 0.1 → 0.2 at the same shape —
   is worth 0.048 nats and +1.7 pts top-1 at the sweep's 3-seed resolution.
3. **Under-regularization is visible in the training dynamics**, not just the
   scores: dropout-0.2 runs train ~1.8× longer before early stopping (72.7 vs
   39.7 epochs at the extremes). The 0.1 models are stopping early because
   they start overfitting, not because they have converged.
4. **Width above 192 never pays.** All six d256 configs rank 9th or worse
   despite up to 2.5× the parameters, and the two worst configs in the sweep
   are the two largest at low dropout. Depth 6 helps only when paired with
   dropout 0.2. The task is data-limited (5,043 training games), not
   capacity-limited.
5. **Loss and top-1 disagree again.** This is the third time today. The loss
   winner (`d128x6L_do0.2`, top-1 15.59) is only 5th on top-1, while
   `d128x4L_do0.2` leads top-1 at 16.76 with 30% fewer parameters and a loss
   0.0061 worse. `d256x4L_do0.1` is another instance: 4th-best ban top-1 in
   the sweep on the 3rd-worst loss.

## The selection rule misfired, and I am not going to pretend otherwise

The pre-registered rule was: select on mean val loss, ties within 0.005 broken
toward the smaller model. It nominated `d128x6L_do0.2`, which then failed the
confirm gate on top-1.

`d128x4L_do0.2` lost by **0.0061 nats** — a hair outside the tie band that
would have handed it the win on the parameter-count tiebreak — while being
**+1.17 pts better on top-1** and 264k parameters smaller. Had the band been
0.007, the sweep would have nominated a different, cheaper, more accurate
model.

I chose loss as the selection metric for a defensible reason (top-1 on 1,080
decisions over 3 seeds has a standard error near a point; loss does not), and
that reasoning still holds. But the outcome shows the cost: a noise-level loss
difference selected against a config that was better on the thing being
optimized for. The honest conclusion is that **stage 1's nomination was not
informative**, not that large capacity changes are worthless.

Because the pre-registered path is now spent, `d128x4L_do0.2` was run against
the incumbent on the same 5 paired seeds as an explicitly **post-hoc**
comparison — chosen after seeing the sweep, written to
`data/processed/capacity_sweep_posthoc.json`, and **not promotable on this
run** regardless of what it shows. It is reported in the section below for
information. Treat any number there as a hypothesis for a fresh pre-registered
test, not as a result.

## Post-hoc comparison — `d128x4L_do0.2` vs incumbent (NOT a result)

Raw: `data/processed/capacity_sweep_posthoc.json`, which carries a `posthoc`
field saying the same thing this section does.

| | mean | per seed (16/17/42/7/23) |
|---|---|---|
| d_top1 | **+1.46 pts** | +0.46, +3.42, +2.41, +0.46, +0.56 |
| d_ban_top1 | +1.19 pts | +0.93, +4.08, +2.78, −1.30, −0.56 |
| d_pick_top1 | +1.74 pts | 0.00, +2.78, +2.04, +2.22, +1.66 |
| d_loss | +0.068, CI [0.033, 0.103] | 5/5 positive |

It clears the same gate the pre-registered winner failed (+1.46 vs the +1.0
threshold, both guards passed, and unusually it improves bans *and* picks
together — picks are positive on 5/5 seeds). It is also 30% smaller and ~25%
faster to train than the incumbent.

**This is not a GO and must not be promoted.** The config was chosen after
seeing the sweep table, on the same 54 val games the sweep ranked it with, and
selected precisely because it looked good on the metric now being reported.
That is textbook selection bias; the +1.46 is an upward-biased estimate of
whatever the true effect is. The gate it "passed" was pre-registered for a
config nominated by a rule, and it was not that config.

What it is good for: strong pilot evidence that a shape+dropout change in this
direction is worth one clean pre-registered test.

## Assessment and what to do next

**No config change is promoted.** The pre-registered candidate failed its
gate, and the alternative that looks better was selected post-hoc.

The finding that survives all of this is the dropout one, and it does not
depend on the selection rule at all: 9 out of 9 pairs, in the same direction,
including a 0.048-nat gain at the incumbent's own shape. That is a
pre-registerable hypothesis with an obvious test:

1. **Dropout 0.1 → 0.2 at the current shape** (`d192x4L`), 5 paired seeds,
   gated on top-1 the same way this confirm stage was. One df, a-priori
   motivated by nine unanimous pairs plus the epoch-count evidence, and it
   changes a single scalar rather than the architecture. This is the cheapest
   real lead left. Note the sweep's 3-seed numbers for it (+1.7 pts top-1) are
   *in-sample for the sweep* and cannot be quoted as the result — hence a
   fresh 5-seed run.
2. **Re-test `d128x4L_do0.2` properly** if a shape change is wanted, treating
   today's post-hoc run as pilot data only. If both (1) and (2) are wanted,
   run them as one pre-registered comparison with a single declared primary —
   not two bites at the same val set.
3. Nothing here changes the `ban_only` recommendation from earlier today; the
   sweep was run on top of it and the injection's gain is not a capacity
   artifact — every config in the table carries the injection.

Honesty notes:

- Val-only throughout, as the rails allow for hyperparameter selection. 54
  games / 1,080 decisions. The frozen test set was not touched.
- 18 configs were compared on 54 val games. Even with loss as the metric, the
  spread across the middle of that table (3.339–3.345) is well inside what
  three seeds can resolve; only the top-vs-bottom contrast and the
  dropout-pairing result are safe to lean on.
- Accumulated val selection pressure across this lineage is now substantial
  (phase blend split, v13, v14, this sweep). The test set remains the only
  clean read left, which is exactly why it stays frozen.
