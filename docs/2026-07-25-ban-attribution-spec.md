# Spec: denial-vector attribution for the pick/ban phase

**Written 2026-07-25 ~13:35 ET, from Sahil's framing. Status: DRAFT for
discussion — the proxy definitions in §3 are the part to argue about before
anything runs.**

## 1. The reframe

Every model so far predicts the next decision; none of them explain *why* a
decision is worth making. Sahil's frame: picks and bans are both **denial
tools**, along three vectors —

- **V1 Meta denial**: remove a champion that is currently very strong for
  everyone (ban it, or first-pick it away).
- **V2 Pocket denial**: remove a specific opposing player's comfort pick
  ("their ADC is scary on Lucian — ban it or pick it away").
- **V3 Comp denial**: remove what is strong against our composition, or
  pick something that fits us *well enough* while denying them something
  that fits them better.

These motives are tangled inside one predicted probability. This study
untangles them descriptively: classify historical bans by dominant vector,
measure the mixture by draft slot, and condition each model's ban accuracy
on the vector — so the next accuracy rung is chosen by evidence, not vibes.
Prediction ≠ valuation; this is the bridge.

## 2. Verified data anchors

`draft_decisions_multi.parquet` ban rows already carry opponent-referenced
features (`draft_dataset.py:13-32`, `560-623`):

- `presence` = 28-day trailing pick+ban rate (global meta strength).
- `opp_usage` = opponent team's 56-day usage of the candidate.
- `player_pool` / `player_wr` = **for bans, the opponent's** inferred
  roster: how much of their players' 180-day pools the candidate is, and
  how well they perform on it, weighted by their open-role probabilities.
- `pair_ctr` = candidate's shrunk WR **versus the banning team's own locked
  picks** ("good against our comp"); `pair_syn` = candidate alongside the
  **opponent's** locked picks ("slots into their comp").

Pick rows are scored from the picking team's perspective only — pick-denial
has no ready-made columns (see Stage C).

## 3. Proxy definitions (v0 — argue here)

All scores are **within-decision percentiles** across that slot's available
candidates, so vectors are comparable on one scale. For each ban target b:

- `META(b)` = pct(presence)
- `POCKET(b)` = pct(opp_usage) ⊕ pct(player_pool · max(player_wr − 0.5, 0))
  — ⊕ = max of the two percentiles for v0 (team habit and player pool are
  both "they play this a lot / well"; max avoids double-counting scale)
- `COMP(b)` = pct(pair_ctr), with pct(pair_syn) reported alongside but not
  fused in v0

Dominant vector = argmax; **multi-motive** if the top two are within ε = 10
percentile points (report sensitivity at ε = 5, 15). Known mechanical
artifact: B1–B3 have no locked picks, so pair features are neutral and
COMP is undefined early — the slot-mixture table will show this by
construction; early bans can only be V1/V2. That matches the game reality.

Open questions for Sahil:
1. Should POCKET require player-level signal (pure `player_pool`·`wr`),
   with `opp_usage` as a separate "team habit" sub-vector?
2. Is max the right fusion, or should multi-motive be the object of
   interest rather than a tie to break?
3. Does COMP include "fits their comp" (`pair_syn`) or only "threatens
   ours" (`pair_ctr`)? Sahil's framing includes both.

## 4. Stages

**A — mixture (descriptive; local, pure pandas, minutes).** All train+val
bans (~51k). Outputs: vector mixture by ban slot (expect V1-heavy B1–B3,
V2/V3 rise in B4–B5), multi-motive rate by slot, per-vector *lift* (proxy
percentile of the actually-banned champ vs the availability-pool base
rate), mixture over time and by league.

**B — model-gap conditioning (val's 540 bans; directional, small n).**
Per-ban top-1/top-5 hits for (a) baseline transformer, (b) meta
transformer, (c) production GBM, each conditioned on the ban's dominant
vector. Hypothesis: the ~7-point blind ban gap concentrates in V2 (the
transformer has no team/player inputs — pocket denial is structurally
invisible to it), is partially closed in V1 by the time-signal rung, and
is smallest in V3. Infra notes: meta-transformer ensemble val probs are
cached (`expcache_timesignal_blind/`, val slice); baseline ensemble needs
a 5-seed local retrain (~15 min, canary 3.5855); GBM val scores need one
codespace run (lightgbm) — **cache the GBM val scores to an npz this
time**. Val only; the frozen EWC test set is not touched.

**C — pick-denial counterfactual (optional, after A/B).** For each early
pick, opponent demand = the meta transformer's predicted probability of
that champion at the opponent's next pick slot with the champion left
available (one forward pass with edited availability masks). Denial score
= demand percentile. v0 limitation to flag: conditioning on the actual
later draft is teacher-forcing — score only the *immediately next*
opponent pick slot to keep the approximation honest. This is the seed of
decision *valuation* (denial value ≈ demand × strength); full valuation
needs win-prob linkage and is explicitly out of scope here.

## 5. Guardrails & sequencing

- Descriptive study: no test look, no production change, no weight tuning.
- Deliverable: `data/processed/ban_attribution.json` + a results doc with
  the three tables; decision output is *which rung next* — team/player
  conditioning (if V2 dominates the gap), input-side rate conditioning
  (if V1 residual), or wait for soloq synergy tables (if V3 — rung 1b,
  spec'd, blocked on the other session's scrape).
- Related open thread: the soloq scrape currently running in a parallel
  session feeds V3 directly (role-aware synergy/counter tables).
