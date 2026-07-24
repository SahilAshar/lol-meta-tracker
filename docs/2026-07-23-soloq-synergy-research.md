# Soloq Champion Synergy: From Co-occurrence to Win-Rate Lift

*Research memo, 2026-07-23. Question from Sahil, after the Step-3 embedding
transfer NO-GO (`docs/2026-07-23-step3-results.md`): the soloq data's real
value is measuring which champion combinations raise win rate — pairs first,
comps eventually — because pro drafts are near-optimal but not optimal, and
the north-star question is "would a different pick have improved this pro
draft's win probability?" His framing: soloq estimates the interaction
structure, pro data validates and calibrates it. Master+/apex tiers (esp. KR)
are the closest soloq gets to pro-level play.*

*Think-and-write memo plus a few read-only queries against the scraped db.
No training run, no test set touched, no code changed.*

---

## 1. TL;DR

**The direction is sound and it is a continuation, not a pivot.** The
embedding-transfer thread is *concluded* (NO-GO, written up); this reuses the
same scraped asset, the same rung/gate discipline, and plugs into two things
the repo already has: the pro `pair_syn`/`pair_ctr` features (currently
estimated from only ~5k pro games) and the v0.9 outcome head, whose gate
("draft signal in win prediction is real") already passed. The original soloq
memo (§5, 2026-07-22) explicitly predicted this: *"the v0.9 outcome head is
where soloq's outcome signal could matter most."*

**One reframe before anything else:** rung 0 never used the `win` field. The
embeddings encoded *what gets picked together* (mostly role structure). This
direction uses *what wins together* — a different, unused column of the same
data.

**Two statistical corrections to the raw idea:**

1. **Full 5-champ comps are unmeasurable directly.** ~10^9 possible comps vs
   ~154k team-observations: essentially every comp in the data is unique.
   The measurable unit is the **pair** (14,878 possible; ~1.5M same-team pair
   observations in hand, ~103 games per pair on average, heavily skewed).
   Anything larger than pairs must come from a *model* that generalizes,
   validated on held-out win prediction — never from lookup tables.
2. **Synergy is the interaction, not the win rate.** Two strong champions
   winning together is expected. The signal is the lift *above* what their
   individual (patch-aware) strengths predict. Skip this and every analysis
   "discovers" that strong champions synergize with everything.

**The cheap falsifiable gate (rung 1):** on held-out soloq games, does a
win-prediction model with pair-interaction terms beat the same model with
champion main effects only? If pair terms add nothing *in-domain*, the whole
direction is falsified for CPU-minutes. Spec: §5; execution:
`docs/2026-07-23-synergy-rung1-handoff.md`.

---

## 2. What the data supports (verified 2026-07-23)

Scraped db (`data/raw/soloq/soloq.db`, gitignored): **78,171 done matches**;
after cleaning (queue 420, duration ≥ 300s i.e. no remakes, 10 participants):
**76,995 games**, NA/KR/EUW roughly even thirds. Payload per participant:
champion, team, role (`pos`), and **win** — plus bans, patch
(`game_version`), and timestamp per match.

- Window: 2026-06-08 → 2026-07-23 (scraper still running), patches
  **16.11–16.14** (16.13 alone is 40k games).
- Coverage: all **173** champions appear; median 3,813 games per champion,
  minimum 691. Main effects are estimable for everyone.
- **Blue side wins only 47.9%** at Master+ — the well-known high-elo red-side
  (counter-pick) advantage. Side must be in every model; a synergy analysis
  that ignores it inherits a 2-point bias.
- Matchmaking equalizes team skill, which *removes* the biggest confounder pro
  data has (team strength). The remaining confounders are champion strength,
  patch, and side — all modelable.
- 1,176 sub-5-minute games (remakes) and 7 malformed rows get dropped.

## 3. Why pairs, and what "lift" means precisely

For champions A, B on the same team, the synergy lift is the effect of
*A-and-B-together* on win probability after accounting for A's strength, B's
strength, side, and patch. Operationally that is an interaction coefficient in
a win model, or equivalently an empirical-Bayes shrunk deviation:

    lift(A,B) = shrink( WR_observed(A+B) − WR_expected(A, B) )

Counters are the same construction across teams: `ctr(A,B)` = effect of A
*facing* B, antisymmetric by definition. The db's `pos` tags allow role-aware
refinement later (jungle+mid synergy ≠ bot-lane synergy), but rung 1 stays
role-blind to keep the gate simple.

Honesty about power: ±2-point pair lifts need ~2k games/pair to detect
individually; most pairs have far fewer. So rung 1's gate is **aggregate** —
"do pair terms improve held-out prediction at all?" — not per-pair
significance. Individual pair estimates are reported shrunk, as artifacts for
face-validity, not as claims.

## 4. Continuation or new direction? (the routing question)

Asked directly: *does this build on existing work, or is it new work that
should wait until the previous thread concludes?*

- **The previous soloq thread IS concluded.** Rung 0 (purity gate) passed;
  Step 3 (transfer) ran to a clean NO-GO with a written verdict. Nothing is
  half-open. (The Step-3 artifacts await one approval-gated commit.)
- **This reuses, not replaces:** same db and scraper, same PRO_TO_SOLOQ name
  bridge, same rung methodology, same frozen-test discipline. Rung 2 would
  feed the *existing* `pair_syn`/`pair_ctr` features
  (`draft_dataset.py:597-622`) and the *existing* outcome-head lineage.
- **The one genuine fork:** "inform better pro decisions" is the **coach**
  north star, and the ROADMAP notes that mimic-vs-coach conversation as
  unhad and gating any full outcome build. Rung 1 does not force the fork —
  it is soloq-in-domain, touches no pro model, and its lift tables are useful
  to either north star. The fork gets real at rung 3. Recommendation: run
  rung 1 now; treat its results note as the opening exhibit of the
  mimic-vs-coach conversation rather than front-running it.

## 5. The rung ladder

**Rung 1 — in-domain gate (cheap, decisive, no pro data).** One row per game,
predict blue-side win. M0: side only (intercept). M1: + signed champion main
effects (+1 blue / −1 red). M2: M1 + signed same-team pair features.
M3: M2 + cross-team counter pairs. Chronological splits; hyperparameters on
val; holdout scored once. **GO** if M2 (or M3) beats M1 on holdout log-loss
with a 10k-bootstrap 95% CI excluding 0. **NO-GO** if within noise — which
would falsify the premise cheaply and redirect the soloq asset to meta-rate
features (the other surviving use from the 07-22 memo). Artifacts: metrics
JSON + top-20 shrunk synergy and counter pairs with game counts.

**Rung 2 — transfer gate (touches pro models, mimic-compatible).** Recompute
the GBM's `pair_syn`/`pair_ctr` from soloq lifts (or blend soloq prior with
pro observations — EB with soloq as the prior mean) and measure pro **val**
top-k against the current pro-only features. Same 5-seed/CI discipline as
Step 3. Independently: add comp-lift features to the outcome model and measure
val log-loss. Either improvement is a GO.

**Rung 3 — the counterfactual critic (coach north star, gated).** A
win-probability model over both comps, queried at each pro decision point:
"which available champion maximizes win probability, and how far from it was
the actual pick?" Requires rungs 1–2 GO, the mimic-vs-coach decision, and
serious calibration care (a critic that is confidently wrong is worse than
none). Not specified further here on purpose.

## 6. Open questions

- **Region stratification.** Sahil's hypothesis that KR Master+ tracks pro
  best is testable: fit rung 1 per region, compare lift structure and
  transfer (rung 2) per region. Deferred to keep rung 1 small; the `region`
  column makes it a one-line ablation later.
- **Patch pooling.** Four adjacent patches are pooled in rung 1 (main effects
  absorb most drift). If the scraper accumulates across a big balance patch,
  recency weighting or per-patch effects become necessary.
- **Role-aware pairs.** `pos` tags support (champ,role) tokens — sharper
  synergy definitions at the cost of sample per cell. A rung-1.5 if the gate
  passes.
- **Comps beyond pairs.** Only via a generalizing model (factorization/
  set encoder). Do not attempt before rung 2 proves pair-level transfer.
- **Soloq→pro attenuation.** Coordination-dependent synergies may be
  *understated* in soloq. Rung 2's blend (soloq prior + pro update) is the
  hedge; pure replacement is the ablation.
