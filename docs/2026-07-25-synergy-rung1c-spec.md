# Spec: synergy rung 1c — the single-scalar duo-lift test

**Written 2026-07-25 ~13:05 ET, after rung 1b's split verdict
(`docs/2026-07-25-synergy-rung1b-results.md`: slotting GO, counters GO,
same-team pair synergy NULL at 323k games). Audience: the agent that will
execute rung 1c in a fresh session. Status: SPECCED, READY — the db is
final (329,703 done games, closed out 2026-07-25); no trigger to wait
for. Runtime: CPU-minutes. Independent of rung 2 — do not block either
on the other; 1c's only coupling is that a PASS adds one optional feature
to rung 2's arsenal.**

## Why this rung exists

Rung 1b falsified same-team role-pair synergy as *thousands of individual
pair columns* (A-vs-B CI [−0.000246, +0.000568] straddling 0). But sites
like U.GG publish duo tables with "synergy factors" of +1 to +5 points,
and our own descriptive D tables show similar lifts. Two readings are
compatible with 1b's null:

1. Those tables are noise-plus-role-effects — champ@role main effects
   plus leaderboard selection on ~1-pt standard errors.
2. There is a thin real synergy layer, but spreading it over ~7,400
   regularized columns diluted it below detectability even at 323k games.

Rung 1c distinguishes these with the **maximum-power version of the
test: one degree of freedom.** Collapse each team's claimed duo synergy
into a single scalar (the sum of EB-shrunk train-split duo lifts over the
four priority vectors) and ask whether that one number improves held-out
prediction over champ×role main effects. If even the 1-df test fails —
while a same-construction counter scalar (the positive control) passes —
the U.GG-style column is confirmed uninformative beyond role effects,
and the falsification is airtight.

## Design

Reuse rung 1b's loader and discipline verbatim
(`scripts/experiment_v10_synergy_rung1b.py`): clean-game filter (queue
420, duration ≥300, `creation > 0`, no blank/dup roles), chronological
70/15/15 by `game_creation`, C swept per model on val only, holdout
scored once, 10k paired bootstrap on per-game log-loss. Expected
anchors: 323,401 clean games, blue WR 0.4755, 173 champs, split
boundaries Jul 13 / Jul 19 / Jul 25.

**Lift tables (train split ONLY — no val/holdout games touch any
table).** Identical construction to 1b's D table:

- `wr(c)` = champ train WR; `wr(c@r)` = champ@role WR EB-shrunk toward
  `wr(c)` with prior 200 games.
- Duo lift, per priority vector (BOT+UTIL, MID+JG, TOP+JG, JG+UTIL):
  `syn_lift(a@ra, b@rb) = shrink_200( WR_obs(pair) − mean(wr(a@ra), wr(b@rb)) )`.
  Unobserved pairs → 0.
- Counter lift, per same-role matchup (5 lanes, bot 2v2 as ADC-vs-ADC +
  SUP-vs-SUP): `ctr_lift(a, b | r) = shrink_200( WR_obs(a beats b in r)
  − (wr(a@r) + 1 − wr(b@r))/2 )`, antisymmetric key as rung 1. Unobserved → 0.

**Scalar features per game** (signed, blue minus red):

- `SYN = Σ_vectors [syn_lift(blue duo) − syn_lift(red duo)]` (4 terms/side)
- `CTR = Σ_roles ctr_lift(blue champ vs red champ | role)` (5 terms, blue
  perspective — already signed)

Standardize both scalars on train statistics before fitting.

## Models and gates

| model | features | role |
|-------|----------|------|
| A | champ×role mains (1b's A design, refit) | baseline |
| A+SYN | A + the synergy scalar | **the test** |
| A+CTR | A + the counter scalar | **positive control** |
| A+SYN+CTR | both | interaction check |

- **Primary gate:** A+SYN beats A on holdout log-loss, 95% CI excluding
  0 → a thin real synergy layer exists; hand the SYN scalar to rung 2 as
  an optional feature and say so in the results doc.
- **Control gate:** A+CTR must beat A. Counters are already proven real
  (1b's B-vs-C: CI [+0.000274, +0.000713]), so if the scalar mechanism
  can't detect them, the mechanism is broken (leakage bug, shrinkage too
  aggressive, standardization error) and the SYN result is
  **uninterpretable — a design error, not a falsification.** Debug
  before writing any verdict.
- If SYN fails and CTR passes: same-team duo synergy is falsified at
  1 df. Write it as the definitive answer to "why do U.GG's tables not
  contradict us."

## Extra deliverable: the U.GG readout

Descriptive, not gated: decile-bucket the holdout games by their SYN
score and report actual blue WR per decile vs predicted-from-A. A flat
line is the direct, publishable answer to "is a duo-synergy column
informative once you know champ@role?" Also report the fitted SYN and
CTR coefficients with the implied WR effect per point of claimed lift.

## Protocol

- Script `scripts/experiment_v10_synergy_rung1c.py`, output
  `data/processed/synergy_rung1c.json` — mirror 1b's provenance block
  (counts, drops, split boundaries, patch mix, blue WR, runtime) and add
  the scalar summary stats (train mean/sd, holdout decile table).
- Import/copy 1b's loader; do not re-derive cleaning rules.
- Runtime should be minutes (1b's full ladder ran in 41 s; the tables
  are the only new cost). If a fit exceeds ~5 min something is wrong.
- Never stage `docs/ROADMAP.md` or `artifact/*`; run `date` before
  writing dates; commit/push only on Sahil's approval.

## Relation to the ladder

- **Rung 2 does not wait for this.** Its transfer payload (champ×role
  strength + role-aware counters) is already justified by 1b. A 1c PASS
  adds the SYN scalar to rung 2's optional features; a 1c FAIL removes
  same-team synergy from the roadmap entirely, including the pro GBM's
  existing `pair_syn` feature (flag it as a deletion candidate).
- Results doc: `docs/2026-07-25-synergy-rung1c-results.md` (or the date
  it actually runs — check `date`).
