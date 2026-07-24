# Results: ban time-signal rung — trailing meta-rates into the transformer

**Written 2026-07-23 ~22:45 ET. Executes
`docs/2026-07-23-ban-timesignal-handoff.md` (ROADMAP open loop #2).**
Script: `scripts/experiment_v11_ban_timesignal.py` (v09 scaffolding);
raw results: `data/processed/timesignal_rung.json`; per-run cache:
`data/processed/expcache_timesignal/`. Val only — the frozen EWC July-2026
test set was dropped at load and never evaluated.

## Verdict: GO — all three criteria met

| Criterion | Bar | Observed | Met |
|---|---|---|---|
| Mean ban val top-1 delta | ≥ +1.5 pts | **+1.93 pts** (7.70 → 9.63) | yes |
| Bootstrap 95% CI of paired val-loss delta | excludes 0 favorably | **[+0.095, +0.115]**, mean +0.105, 5/5 seeds | yes |
| Pick top-1 guard | no degradation beyond ±1.5 pts | **+0.15 pts** (mean) | yes |

Per-seed deltas (meta − baseline), seeds [16, 17, 42, 7, 23]:

- ban top-1: +2.41, +2.40, +0.93, +2.41, +1.48 — **all positive**
- pick top-1: 0.00, +0.18, +0.37, −0.37, +0.56 — noise, no cost to the
  picks specialist
- val loss (baseline − meta): +0.123, +0.088, +0.096, +0.109, +0.108 —
  a large, uniform effect; for scale, the soloq-transfer rung's deltas were
  ±0.03 and straddled zero

## What was run

Design exactly per the handoff's minimal version: the 28-day trailing global
`pick_rate`/`ban_rate` columns of `draft_decisions_multi.parquet` (causal by
construction, constant per (gameid, candidate)) were scattered into one
(vocab, 2) matrix per game, standardized with **train-game statistics only**,
and added to the output logits through 4 learnable scalars —
`meta_w[slot_type, feature]`, zero-initialized:

    logits[g, slot, champ] += meta_w[slot_type(slot), f] · meta[g, champ, f]

RNG hygiene held: `meta_w` is created deterministically after the base
modules, so baseline and meta conditions share identical init/dropout/shuffle
streams per seed — strictly paired. The baseline condition (no meta tensor)
reproduced the seed-16 anchor **3.5855 exactly** (hard-asserted canary).
Production config d192x4L6H, cutoff 2026-07-15, train 5,043 g / 100,836 d,
val 54 g / 1,080 d (all asserted), CPU, ~2.5 min/run.

Learned weights are all positive and stable across seeds (~0.13–0.23), with
ban slots weighting both features slightly heavier than pick slots and
`ban_rate` heaviest for ban slots — the model reads the meta prior most where
the handoff predicted it was blindest.

## One handoff correction

The claim "every game's candidate table covers all 168 candidates" is false
for fearless games: series-prior champions are excluded from the candidate
table (e.g. 128 = 168 − 40 in game 3 of a fearless series). Harmless for
this design — those champions are never available, so their logit cells are
masked to −inf everywhere and their zero meta rows are never read. The
script asserts exact per-game coverage of 168 − |prior-banned candidates|
instead, plus the rates-constant-across-slots invariant.

## Interpretation & next steps

- Output-layer rate injection — the cheapest possible time signal — works:
  +1.9 ban top-1 points and a val-loss drop an order of magnitude beyond
  seed noise, at a cost of 4 parameters.
- The transformer's val ban top-1 (9.6) still trails the pure GBM's 14.3 val
  sweep number: rates close part of the date-blindness gap, not all of it.
  The natural next rungs are input-side conditioning (let attention see the
  rates, not just the head) and re-sweeping the v0.8.1 production blend
  weights with the meta-aware transformer.
- **Promotion to a blind-test look requires Sahil's explicit approval** — it
  would be the third look at the frozen EWC test set. Nothing here touched it.
