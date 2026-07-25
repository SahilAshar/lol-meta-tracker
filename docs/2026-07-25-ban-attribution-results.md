# Results: denial-vector attribution (Stages A–C)

**Written 2026-07-25 ~14:00–15:00 ET. Spec:
`docs/2026-07-25-ban-attribution-spec.md`. Descriptive study — no test look,
no production change. Raw tables: `data/processed/ban_attribution.json`.
Scripts: `ban_attribution_stageA.py` (mixture), `ban_attribution_train.py`
(ensembles for B/C, canaries 3.5855/3.4623 both hit, meta val probs asserted
bit-identical to the blind cache), `gbm_val_scores.py` (codespace),
`ban_attribution_stageB.py`, `ban_attribution_stageC.py`.**

## Stage A — what bans are made of (50,945 train+val bans)

Proxies are within-decision percentiles (spec §3 v0 definitions; the three
open questions were resolved empirically below rather than by fiat). One ban
of 50,946 targeted a champion outside its availability pool (fearless data
quirk) and is unattributable.

### Mixture by ban slot (dominant vector share)

| Slot | META | POCKET | COMP |
|---|---|---|---|
| B1 | 0.63 | 0.37 | 0.00 |
| B2 | 0.59 | 0.41 | 0.00 |
| B3 | 0.59 | 0.41 | 0.00 |
| B4 | 0.31 | 0.56 | 0.13 |
| B5 | 0.31 | 0.56 | 0.13 |

COMP is undefined in B1–B3 by construction (no picks locked — pair features
neutral), which matches the game. The headline: **phase-2 bans flip to
pocket-dominant** (56%), exactly where the draft has revealed who plays what.

### The pocket signal is the *player*, not the team

4-vector sensitivity split (spec open question 1): at B4–B5 the pocket share
decomposes into PLAYER 0.46–0.47 vs TEAM habit 0.09; even in B1–B3 PLAYER
(0.22) edges TEAM (0.15–0.19). Team habit is a minor sub-vector everywhere —
"their midlaner is scary on it" is the motive, not "that org likes it".

### Lift: how far above random the banned champ sits on each vector

Mean percentile of the actually-banned champion minus the 50 base rate:

| Slot | META | POCKET | COMP (pair_ctr) | SYN (pair_syn) |
|---|---|---|---|---|
| B1–B3 | +43 to +44 | +36 to +38 | — | — |
| B4 | +38.9 | +37.3 | +2.1 | +6.0 |
| B5 | +38.5 | +36.2 | +0.5 | +4.6 |

Two findings here:

1. **META and POCKET are both huge and nearly co-equal** (banned champs sit
   ~87–94th percentile on both) — the motives are deeply entangled: strong
   meta champs are also in strong players' pools. Multi-motive rate at
   ε=10 is 59–71% by slot (37–49% at ε=5, 74–81% at ε=15). Spec open
   question 2 resolves itself: multi-motive is not a tie to break, it is
   the normal case; single-motive bans are the minority.
2. **`pair_ctr` ("counters our locked comp") is nearly uninformative**
   (+0.5–2.1 lift ≈ random), while `pair_syn` ("fits their comp") carries
   real signal (+4.6–6.0). Phase-2 bans target what the opponent's draft
   *wants next*, not what abstractly beats ours. Spec open question 3:
   comp-denial as practiced is denial of THEIR comp-fit — if COMP is kept
   as a vector it should be built on pair_syn, not pair_ctr. (With the
   fused max(ctr, syn) variant, COMP's phase-2 share rises 13%→24%, taken
   mostly from META.)

Mixture is stable across leagues (META 0.46–0.55 everywhere; slightly more
meta-flavored at international events EWC/FST) and across 2026 months
(META 0.49–0.58) — this is a structural property of drafting, not an era
artifact.

## Stage B — where each model's ban accuracy lives (540 val bans)

Top-1/top-5 per model, conditioned on dominant vector. GBM = codespace
refit; comparable only within this table (its overall bans 13.3/46.5 and
the meta-alone 8.9/40.9 exactly reproduce the blend re-sweep's w=0.0/w=1.0
rows — reconstruction verified). Baseline transformer = v0.8 lineage.

| Cell | n | baseline TF | meta TF | GBM |
|---|---|---|---|---|
| all bans | 540 | 8.3/36.9 | 8.9/40.9 | **13.3**/46.5 |
| META | 291 | 10.0/37.5 | 11.3/45.4 | **15.5**/50.2 |
| POCKET | 221 | 6.3/34.4 | 5.4/34.4 | **12.2**/45.2 |
| COMP | 28 | 7.1/50.0 | **10.7**/46.4 | 0.0/17.9 |
| B1–3 × META | 209 | 5.3/32.1 | 6.2/42.1 | **17.7**/52.1 |
| B1–3 × POCKET | 115 | 1.7/24.3 | 2.6/24.3 | **13.9**/46.1 |
| B4–5 × META | 82 | 21.9/51.2 | **24.4**/53.7 | 9.8/45.1 |
| B4–5 × POCKET | 106 | 11.3/45.3 | 8.5/45.3 | **10.4**/44.3 |
| B4–5 × COMP | 28 | 7.1/50.0 | **10.7**/46.4 | 0.0/17.9 |
| multi-motive | 367 | 9.8/40.6 | 10.9/45.2 | **17.4**/55.6 |
| single-motive | 173 | 5.2/28.9 | 4.6/31.8 | 4.6/27.2 |

Reading it (directional — cells are small):

1. **The spec's hypothesis was half right.** The POCKET gap is real (12.2
   vs 5.4, ~7 pts) and the meta injection didn't touch it. But conditioning
   on phase reveals the sharper truth: **the transformer's entire ban
   deficit lives in B1–B3**, across BOTH vectors (META 17.7 vs 6.2, POCKET
   13.9 vs 2.6). With no draft context to attend over, a sequence model at
   slots 1–6 is running on league/side embeddings and four meta scalars;
   the GBM's tabular features (presence, player pools) own that regime.
2. **In phase 2 the relationship inverts.** B4–5 META bans: transformer
   24.4 vs GBM 9.8 — once twelve slots of context exist, attention beats
   tables, even on meta-flavored bans. POCKET is at parity; COMP (small n)
   the GBM literally never gets right (0/28 top-1) while the transformer
   does. The production ban blend (w=0.0, GBM-only) is leaving the
   transformer's phase-2 advantage entirely on the table.
3. The time-signal rung improved phase-1 META top-5 (32→42) but not top-1 —
   a 4-scalar output bias can re-rank the shortlist, not sharpen the #1.
4. Multi-motive bans are ~3× more predictable than single-motive for every
   model (GBM 17.4 vs 4.6). Redundant motive = predictable ban; the hard
   residual is the single-motive ban, where all three models collapse to
   ~5%.

## Stage C — pick-denial counterfactual (486 val picks)

Counterfactual opponent demand at their immediately-next pick slot, meta
ensemble (edit: pick token → MISSED, champion re-opened at slot u; teacher-
forced elsewhere; MISSED-at-pick-slot is mildly out-of-distribution — v0).

| Pick | mean demand pct | in opp top-5 | above p90 |
|---|---|---|---|
| B-P1 | 91.7 | 37% | 67% |
| R-P1 | 88.7 | 46% | 67% |
| R-P2 / B-P2 | 78.2 / 77.8 | 19% / 22% | 37% / 48% |
| B-P3 / R-P3 | 83.7 / 73.5 | 24% / 15% | 48% / 31% |
| R-P4 / B-P4 | 71.8 / 69.9 | 15% / 20% | 39% / 30% |
| B-P5 | 74.9 | 24% | 37% |
| **overall** | **78.9** | **25%** | **45%** |

Sahil's "pick it away" intuition is quantitatively real: the average pro
pick sits at the ~79th percentile of what the opponent wanted next, and
first-picks at ~90th (two-thirds above p90). Picks ARE denial moves —
roughly a quarter of all picks take something out of the opponent's top-5
demand at their very next selection. Denial value decays through the draft
but never approaches neutral (50).

## Decision — which rung next

The evidence re-ranks the roadmap:

1. **Phase-aware ban blend (cheapest, immediate).** Extend the v0.8.1
   per-type blend to per-type-×-phase: bans B1–3 → GBM, bans B4–5 →
   transformer-weighted. Naive cell arithmetic says ~+2.4 val ban top-1
   points over GBM-only, but that's post-hoc on these same 540 bans — it
   needs a proper val sweep (1 extra df, a-priori motivated by "phase 2 has
   context"). No new model, one codespace sweep.
2. **Team/player conditioning rung (the structural fix).** POCKET is 41%
   of all bans, the transformer's worst vector, and B1–3 POCKET (2.6 top-1)
   is structurally invisible without roster identity. Team embeddings (or
   player-pool features injected like the meta rates) target the largest
   remaining coherent deficit. This also feeds on the soloq scrape
   (player-pool priors) running in the parallel session.
3. **Input-side rate conditioning** (richer than 4 output scalars) is now
   subsumed under #2's injection question — output-layer bias demonstrably
   can't fix phase-1 top-1.
4. **COMP/synergy rung 1b** stays queued behind the soloq tables; Stage A
   says if a comp vector is built it should be `pair_syn`-flavored
   ("denies their comp-fit"), not `pair_ctr`.

Stage C, longer-term: pick-denial scoring is the seed of draft *valuation*
(demand × strength → win-prob linkage), Sahil's original "value of each
part of the pick/ban phase" question. Out of scope here, spec'd as future
work.
