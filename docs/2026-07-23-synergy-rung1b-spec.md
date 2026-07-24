# Spec: synergy rung 1b — role-aware pairs and slot permutations

**Written 2026-07-23 ~22:25 ET, after rung 1's NO-GO
(`docs/2026-07-23-synergy-rung1-results.md`). Audience: the agent that will
execute rung 1b. Status: SPECCED, NOT SCHEDULED — preferred trigger is the
soloq db reaching ~150-200k clean games (see §6), currently ~77k.**

## Why this rung exists

Rung 1 tested *role-blind* unordered pairs: 14,878 columns averaging ~100
games each — a design that dilutes signal across an almost entirely
unmeasurable feature space. It returned a dead null for same-team pairs and
a lean-positive-but-underpowered CI for counter pairs. Sahil's hypothesis,
which this rung encodes: synergy is not a property of arbitrary pairs, it
lives in specific **role vectors** — how duos move on the map, stack
pressure, and create vision:

- **BOTTOM + UTILITY** (bot duo) — the canonical 2v2 lane
- **MIDDLE + JUNGLE** — roam/gank timing, mid-priority-into-jungle-tempo
- **TOP + JUNGLE** — dive/pressure top side
- **JUNGLE + UTILITY** — vision control, invade duos

And a second, distinct hypothesis: for flexible champions the **same duo's
value depends on its slotting**. Camille+Galio might be Camille-top/
Galio-mid, Galio-top/Camille-sup, or Galio-mid/Camille-sup — different
permutations, plausibly different lifts. If permutation matters, late-draft
slot-shuffling ("I have 4 picks locked; slotting Galio and moving bodies
around") is a real lever for win percentage, and a draft tool should know it.

Data supports the design: `pos` is present and clean on ~100% of
participants (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY; ~3 blanks per 3,000
entries — drop those games).

## Ladder within the rung (each step falsifiable on its own)

Same house discipline as rung 1: one row per game, predict blue win, signed
indicators (+1 blue / −1 red), chronological 70/15/15 by `game_creation`,
C swept per model on val only (start the grid at 3e-5 — rung 1 optima sat
at 0.003 and pair blocks want heavy shrinkage), holdout scored once, 10k
paired bootstrap on per-game log-loss, GO = 95% CI excluding 0.

- **A. Champ×role main effects vs champ main effects.** Columns = observed
  (champion, role) combos only (min ~50 train games; everything rarer pools
  into the champ-only column). This is the cheapest test of "slotting
  matters at all": Galio-mid and Galio-top become separate effects. If A
  itself fails, permutation-aware synergy is unlikely to pass.
- **B. A + same-team role-pair interactions.** Feature = (champA@roleA,
  champB@roleB) same team, restricted to the four priority vectors above.
  Role-conditioning shrinks the space enormously: each vector draws from
  ~40-80 champs per role, and min-games pooling (~50 train games per
  feature, rest pooled) keeps every retained column measurable. Report the
  four vectors' contributions separately (fit B with each vector ablated)
  so "bot duo matters, top-jungle doesn't" is an available answer.
- **C. B + lane counters.** Cross-team same-role matchups (top-vs-top,
  mid-vs-mid, jungle-vs-jungle, and the bot 2v2 collapsed to
  ADC-vs-ADC + SUP-vs-SUP to start). Rung 1's only whisper of signal was
  counter pairs; this is where it should concentrate if real.
- **D. Permutation contrast (descriptive, not gated).** For every champion
  pair observed in ≥2 distinct role-slottings with ≥150 train games each,
  report per-permutation EB-shrunk lifts side by side (prior strength ~200
  games, as rung 1). Camille+Galio is the motivating example — check it
  explicitly in the writeup. This table is the draft-tool payload even if
  the B/C gates fail.

## Gate

GO if B beats A, or C beats B, on holdout log-loss with the bootstrap CI
excluding 0. A-vs-champ-only is reported but not a GO/NO-GO for the rung
(it gates *interpretation*: if A fails, treat any B/C pass with suspicion).
A null is a real answer — report it straight.

## Protocol

- Script `scripts/experiment_v10_synergy_rung1b.py`, output
  `data/processed/synergy_rung1b.json` — mirror rung 1's provenance block
  (counts, split boundaries, patch mix, blue WR) and reuse its loader
  pattern; add the `pos` field and drop games with blank/dup roles per team.
- Runtime is still CPU-minutes; if a fit exceeds ~5 min something is wrong.
- Never stage `docs/ROADMAP.md` or `artifact/*`; `date` before writing
  dates; commit/push only on Sahil's approval.

## §6 When to run

Power was rung 1's binding constraint and role-restriction only partially
fixes it (bot-duo pairs still spread ~77k games over ~40×60 combos).
Preferred trigger: **db ≥ 150k clean games** (scrape-resume handoff:
`docs/2026-07-23-soloq-scrape-resume-handoff.md`), which roughly doubles
per-feature counts and puts a 2-point duo lift within detection range for
common duos. Running earlier is acceptable if Sahil wants the answer now —
but then a null on B must be labeled "underpowered", not "falsified", and
step D's tables should still ship (they degrade gracefully with shrinkage).

## Relation to the rest of the ladder

- The ban time-signal / trailing meta-rate direction (`96a2b2b` handoff) is
  independent of this rung and currently the higher-expected-value use of
  the soloq asset — do not block it on 1b.
- If B or C passes, rung 2 (transferring role-aware lifts into the pro
  GBM's `pair_syn`/`pair_ctr`, or comp features into the outcome model)
  reopens with the role-aware features replacing the role-blind ones that
  died tonight.
