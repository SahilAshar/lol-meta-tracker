# Should Solo Queue Draft Data Go Into the Model?

*Research memo, 2026-07-22. Question from Sahil: the v0.8 transformer loses to
the GBM because it is starved for data (102,916 pro decisions). Solo queue
(ranked) could add millions of drafts and maybe better champion embeddings.
His worry, verbatim: "is it worth the potential bad data? And very famously,
soloq does not track pro league very effectively... so I may dilute the draft."*

*This is a think-and-write memo only. No training was run, no test set touched,
no code changed. Where I am speculating or could not verify, I say so.*

---

## 1. TL;DR recommendation

**Bring soloq in, but not the way the "millions of examples" framing implies.**
Two facts collapse the option space before we start:

1. **You cannot rebuild a soloq draft as an ordered 20-token sequence from
   Riot's historical API.** Match-V5 gives you the *bans with their pick turns*
   but the *picks* come back sorted by team + role, not by draft turn. There is
   no field for pick order, and it has been an open feature request since 2019.
   (Verified below — this is the load-bearing fact.) So the transformer's whole
   trick — next-token prediction over an interleaved draft — is not trainable on
   bulk historical soloq. You would only have the *unordered* set of 10 picks +
   bans per game.
2. Even if you could, **soloq meta famously diverges from pro** (agency/carry
   champs vs scaling/teamplay champs), so naive mixing would dilute exactly the
   thing you are trying to predict.

That kills the naive "pretrain the sequence model on soloq" idea. But it does
**not** kill the two things soloq is genuinely good for:

- **(A) Champion embeddings from co-occurrence.** Role and synergy geometry
  come from *which champions appear together and in which roles*, which needs
  **no pick order at all**. Millions of soloq games would give a denser, lower-
  noise embedding than 5,100 pro games — this part of Sahil's hypothesis is
  probably right. Transfer it by initializing `champ_emb` from a soloq model,
  then retrain the transformer body on pro.
- **(B) The time signal the transformer is missing.** The ROADMAP's #1 open rung
  is that the transformer is *date-blind* and loses bans to the GBM (7.6 vs 15.0
  top-1). Soloq **trailing pick/ban rates** are an aggregate — no pick order
  needed — and they are the closest thing to a live meta clock we can get. This
  may serve the ban rung better than it serves the scale rung.

**Do first (cheap, no API key, no test set):** train champion embeddings on a
public soloq co-occurrence dump, measure role purity against our pro-trained
0.704, and measure the pro **validation** loss delta from soloq-initialized vs
random-initialized embeddings. Go/no-go criterion in §6.

**Do not:** naively concatenate soloq and pro drafts and train one model.

---

## 2. The domain gap, with evidence

The gap is real and well-documented, not folklore:

- **Different champion pools.** Solo-carry champions (Katarina, Yasuo, Master Yi,
  Darius) are staples on the ladder and near-absent in pro. Dignitas frames the
  split directly: soloq rewards champions that "solo carry games regardless of
  teamplay," which "hardly ever get spotlight in pro play" because they flourish
  in uncoordinated settings.
  <https://dignitas.gg/articles/the-difference-between-pro-play-and-solo-queue>
- **Pro is more rigid, and the causality runs pro → soloq more than the reverse.**
  Riot's lead designer Phroxzon has said pros are "pretty rigid" in how they
  approach the game; and when a champion hits high presence in pro, its soloq
  pick rate climbs almost immediately.
  <https://www.oneesports.gg/league-of-legends/league-of-legends-meta-riot/>
  Patch-level breakdowns routinely list the two metas side by side as *different*
  lists. <https://www.dtgre.com/2026/01/league-of-legends-patch-26-1-solo-queue-vs-pro-play-meta.html>
- **Different ban logic.** Soloq bans are often "ban the champion I personally
  hate / can't play against" (Yuumi topping global ban rate is the classic
  example), whereas pro bans are targeted at a specific opponent's comfort or a
  system pick. <https://www.sportskeeda.com/esports/yuumi-highest-overall-ban-rate-league-legends-ranked-solo-queue-11-21>
- **Structural differences:** no fearless rules, autofill (players off-role),
  hovers/dodges, ~simultaneous bans per side, no coordinated team strategy, and
  individual rather than team-planned pick-order dynamics.

**What transfers vs what doesn't:**

| Signal | Transfers to pro? | Why |
|---|---|---|
| Champion identity | Yes | A champion's kit is the same in both |
| Role geometry (who is a jungler, a support) | **Yes, strongly** | Role is a property of the champion, not the elo |
| Synergy fundamentals (Rakan↔Xayah, engage+follow) | **Partially** | Real synergies hold; but soloq under-weights coordination-dependent combos |
| Counter fundamentals | Partially | Lane counters hold; teamfight counters weaker in soloq |
| Ban logic | **No** | Different objective function entirely |
| Timing / pick-order priority | **No** | Individual vs team-planned; and see §4, not even reconstructable |
| Meta rates (who is strong *now*) | **No** — different by construction | The two metas are literally different lists |

The honest reading of Sahil's worry: he is right that meta rates and pick timing
would dilute, and right that identity and role geometry would transfer. The
design job is to import the second without importing the first.

---

## 3. The design menu, with a verdict on each

**a. Naive data mixing (train on pro + soloq together). Verdict: no.** This is
the worst option and worth being crisp about. The transformer's loss is
dominated by frequency: soloq would outnumber pro ~100:1, so the model would
learn the *soloq* next-pick distribution and treat pro as noise. You would be
optimizing the wrong meta. It also imports the ban logic we specifically don't
want. This maximizes dilution for minimal benefit.

**b. Pretrain on soloq, fine-tune on pro (the standard low-resource recipe).**
The full sequence version is **blocked** — you cannot build soloq draft
sequences from historical data (§4), so there is no soloq "sentence" to pretrain
next-token prediction on. **But the embedding-only variant is viable and is the
most promising idea here:** train champion embeddings on soloq via a model that
does *not* need order (co-occurrence / skip-gram over team comps, or a set-based
win-prediction MLP), then initialize the transformer's `champ_emb` from those and
retrain the encoder body on pro. This transfers identity + role + synergy
geometry (the parts that transfer) while letting the pro data set all the timing
and sequence behavior. **Verdict: the primary recommendation.** Test it cheaply
(§6).

**c. Domain token ("this is soloq" / "this is pro").** The clean way to let one
model learn both distributions without confusing them — add a domain embedding
the way `league_emb`/`fearless_emb` already work in `draft_transformer.py`.
**But it requires soloq sequences to be worth doing** (the whole point is one
*sequence* model over both domains), and those don't exist for historical soloq.
**Verdict: shelved unless we ever get ordered soloq (live capture, §4).** If we
did, this is how I'd combine them rather than option (a).

**d. Soloq as auxiliary features, especially the time signal. Verdict: do this,
it is the cheapest win.** Soloq **trailing pick/ban rates** are aggregates —
no order needed, available from any dump or a light API pull — and they are a
proxy for "what is the meta right now," which is precisely the transformer's
blind spot. Injected at the output layer (or as extra tokens) they attack the
ROADMAP's #1 rung directly. Note this reframes the whole question: soloq may be
more valuable as a **meta clock** than as a **scale multiplier**. See §5.

**e. From the literature.**
- **DraftRec (KAIST, WWW 2022)** is the closest prior art and it is striking how
  much it prefigures the v0.9 direction. It trains on ~280,000 LoL soloq matches,
  uses two transformers (a per-player network over match history + a match
  network), and recommends champions by **predicted win probability**, not by
  mimicry. <https://arxiv.org/abs/2204.12750> · code:
  <https://github.com/dojeon-ai/DraftRec>. Two lessons: (1) the win-probability
  head over soloq drafts is a proven, published design — validation for v0.9;
  (2) its power comes largely from **per-player personalization** (your soloq
  history), which **does not exist in pro data** the same way, so we can't lift
  their exact architecture. It also implicitly assumes a canonical draft order to
  build its sequence — a reminder that soloq "order" in published work is usually
  an *assumption*, not ground truth (I could not fetch the full methods section —
  ACM returned 403 — so treat the order-handling detail as inferred, not
  verified).
- **OpenAI Five (Dota 2)** drafted via a **win-probability minimax** over the
  hero pool, not sequence mimicry. Same lesson as v0.9: at the frontier, draft
  tools optimize *winning*, not *copying*. (General background; no ordered-
  sequence pretraining involved.)

---

## 4. Data acquisition reality check — including the pick-order verification

**The load-bearing claim, verified: you cannot reconstruct a soloq draft's pick
sequence from Riot's historical API.**

- Match-V5's match object contains a `bans` array *with* `pickTurn` and
  `championId` per ban, so **ban order is recoverable**. But the **picks** live
  in the `participants` array, which is "sorted by `team` + `individual position`
  and that does not reflect the draft pick turn." There is no pick-turn field for
  picks. Riot developer-relations issue #739 (Feb 2023) states this exactly and
  has no Riot fix; issue #192 (2019) requested pick order and was closed without
  one. <https://github.com/RiotGames/developer-relations/issues/739> ·
  <https://github.com/RiotGames/developer-relations/issues/192>
- The only place pick order exists live is **Spectator-v4/v5**, which serves
  *ongoing* games only. You'd have to be watching each game in real time to
  capture order — not viable for bulk historical collection.
  <https://darkintaqt.com/blog/spectator-v4>

**Consequence for the model:** from historical soloq you get, per game, the
*unordered* set of 10 picked champions (with roles), the bans (ordered), and the
outcome. That is enough for **co-occurrence embeddings, synergy/counter priors,
role geometry, aggregate pick/ban rates, and a set-based win model** — i.e.
everything in options (b-embedding), (d), and DraftRec's outcome idea. It is
**not** enough for sequence next-token pretraining (options a full-sequence, c).
This is why the recommendation lands where it does; it is a data constraint, not
a preference.

**Throughput, if we do pull the API** (personal key, per riot-api-assessment.md):
100 req / 2 min sustained ≈ ~72,000 calls/day ceiling. A match needs ~1 call
(plus timeline if wanted). Realistically, with sampling and politeness, **tens of
thousands of games/day** per region is easy — a few days gets you to millions.
The key has a few days' approval lead time and is already on the roadmap for the
Challenger leading-indicator feature, so this pull is nearly free rider on work
we planned anyway.

**Third-party datasets (faster, no key, good enough for the first rung):**
- "Patch 25.14+ LoL Ranked Games" — 100k+ full match records from Riot's API.
  <https://www.kaggle.com/datasets/californianbill/patch-25-14-lol-league-of-legends-ranked-games>
- "Challenger's Ranked Games" (~1,000 games) and 2024/2025 champion/ranked dumps.
  <https://www.kaggle.com/datasets/gabisato/league-of-legends-challengerss-ranked-games> ·
  <https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025>

These carry ban/pick detail and outcomes but inherit the **same pick-order
limitation** (they were pulled from the same API) and can be stale by a few
patches. For a co-occurrence embedding and role-purity check, staleness barely
matters — champion identity is stable. For a *meta-rate* feature, recency
matters, so prefer a fresh API pull there.

---

## 5. How this interacts with the roadmap rungs

The most important interaction: **soloq's best fit may be the ban time-signal
rung, not the scale rung.**

- ROADMAP open loop #2 lists, in order, "give the transformer time signal for
  bans: patch embedding or trailing meta-rate features injected at the output
  layer — its 7.6 ban top-1 vs GBM's 15.0 is entirely current-meta blindness."
  Soloq trailing pick/ban rates are exactly a trailing meta-rate feature, and
  they are **aggregate, so the pick-order wall doesn't apply.** This is the
  lowest-friction way soloq data enters the model, and it targets the single
  biggest measured weakness.
- The **v0.9 outcome head** (rung passed 2026-07-22) is where soloq's *outcome*
  signal could matter most. A set-based soloq win model (DraftRec-style, minus
  personalization) could pretrain the champion representations the outcome head
  needs — soloq has millions of labeled win/loss drafts; pro has ~5,100. But
  this is downstream of the north-star conversation (mimic vs. coach), which the
  ROADMAP says is still unhad and should gate the full build. Don't front-run it.
- The **Challenger leading-indicator feature** (riot-api-assessment.md #4)
  already plans a soloq pull. If we're building that pipeline anyway, harvesting
  co-occurrence embeddings + trailing rates from the same data is nearly free.
  Two features, one ingestion path.

Net: soloq is additive to at least two existing rungs *without* touching the
sequence model. That is a much safer entry point than "retrain the transformer on
soloq."

---

## 6. The falsifiable first experiment (validation only)

**House-style rung 0: settle the embedding-quality claim for the price of a
laptop afternoon, before committing to any pipeline. The EWC test set stays
FROZEN — every number here is train/val only.**

**Step 1 — Build soloq champion embeddings, no order needed.** Take a public
soloq dump (Kaggle, ~100k games is plenty to start). For each game, treat the 10
picked champions (tagged by role) as a "bag." Train embeddings by either
(i) skip-gram / PMI over champion co-occurrence within team and across the game,
or (ii) a small set-based win-prediction MLP whose input layer *is* the
embedding table. Either yields a champion embedding matrix with no pick order.

**Step 2 — Measure role purity on the soloq embeddings.** Same metric as our pro
model: 5-NN role purity (pro-trained baseline = **0.704**; chance ≈ 0.20). This
answers "does soloq geometry even encode role?" in isolation.

**Step 3 — Measure transfer into the pro transformer, on validation only.**
Initialize `draft_transformer.py`'s `champ_emb` from the soloq embeddings
(projected to d=192), then train the transformer body on the existing pro train
split exactly as production does. Compare against the current random-init
baseline on the **validation** split (val = 1,080 decisions, per ROADMAP), using
the same 5-seed protocol so we're above the ±1.5-pt single-fit noise band.
Compare val loss and val top-1/3/5.

**Go / no-go criterion (numerical):**
- **GO** if soloq-init beats random-init on pro **val loss** with a 5-seed mean
  improvement whose bootstrap CI excludes 0, **or** lifts val top-1 by **≥ 1.5
  points** (clearing the documented seed-noise band). AND soloq role purity is
  **≥ 0.55** (clearly above chance; ideally near the pro 0.704).
- **NO-GO / falsified cheaply** if soloq-init is within noise of random-init on
  val loss *and* val top-1, or if soloq role purity is < 0.40. Either result is
  publishable and worth writing up — it would mean the pro data already learns
  everything the embedding can carry, and soloq's scale doesn't help *this* model.

**Cost:** one soloq dump download, a co-occurrence/skip-gram fit (minutes on
CPU), plus 5 pro-transformer refits we already know how to run. No API key, no
new infra, no test-set spend. If it passes, the next rung is the trailing-rate
ban feature (§5); only after both would a full soloq ingestion pipeline be
justified.

---

## 7. Open questions

- **Is a co-occurrence embedding actually a good initializer for a
  *sequence-attention* body?** The pro transformer learns embeddings *jointly*
  with attention; a co-occurrence embedding optimizes a different objective.
  Warm-start might help, or the body might just overwrite it. The rung-0 val test
  answers this empirically before we over-invest.
- **Should we freeze or fine-tune the transferred embeddings?** Freezing protects
  the soloq geometry but blocks pro-specific adjustment; fine-tuning risks
  washing it out on 5,100 games. Worth trying both in step 3 (cheap).
- **Elo/rank filtering.** Challenger-only soloq is closer to pro than Diamond/
  Emerald. Does filtering to top elo tighten the domain gap enough to matter, or
  just shrink the sample? The leading-indicator pull would let us test both.
- **The north-star gate.** If the project pivots to the coach/outcome tool, a
  soloq win-model pretrain (DraftRec-minus-personalization) becomes far more
  valuable than anything in the mimicry lineage — but that decision should
  precede any big soloq build, per the ROADMAP.
- **Unverified detail:** I could not fetch DraftRec's full methods section (ACM
  403), so its exact pick-order handling is inferred. Low stakes — the
  recommendation doesn't depend on it — but flagging it per house rules.
