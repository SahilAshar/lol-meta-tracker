# How much does the draft actually matter? A win-probability swing, measured

**2026-07-28.** A draft at the 90th percentile of counter advantage wins **51.8%**
of its games; the same draft at the 10th percentile wins **43.1%** — an **8.7
win-rate point swing** from lane matchups alone. Put the whole composition
together (champion strength in role + counters + duo synergy) and the top decile
wins **55.6%** against the bottom decile's **40.0%**: a **15.6-point swing**.
For scale, the single largest uncontested edge in this sample — which side of the
map you were assigned — is worth **4.9 points**. The draft's swing is about
**3.2× side selection**, and counters alone are **1.8×** it. The honest caveats,
stated once and meant: this is soloq Master+ on patches 16.11–16.13, not pro; it
is correlational, not causal; and roughly a third of the counter swing is
champion strength wearing a counter costume (details in "What survives", below).
The scored games are patches 16.13–16.14; the matchup tables they are scored
against were built on 16.11–16.13.

Script: `scripts/analysis_counter_swing.py`. Every number in this document comes
from `data/processed/counter_swing.json` — the JSON key is named next to each
table. Runtime 5 s, deterministic, no model training.

## What was measured, in one paragraph

Three earlier experiments (rungs 1b, 1c, 2) established *that* draft-composition
effects are real in soloq, but reported them in per-game log-loss: counters
+0.000624, duo synergy +0.000061. Those are correct and unreadable. This analysis
takes the same lift tables and asks the reader's question instead: if a draft has
a good matchup profile, how much more often does it win? Each game gets three
blue-minus-red numbers from the pre-built tables — **CTR** (sum of the five
same-role counter lifts), **SYN** (sum of duo lifts over the four priority
vectors), **ROLE** (sum of champion-in-role win rates) — and then we count.

## The trap this analysis had to avoid

The lift tables are built *from game outcomes*. Score a game that is inside the
table's own sample and that game's result leaks into its own lift, inflating
every gradient you then measure. This killed rung 1c's first run.

The fix here is structural rather than statistical: the tables are built from
124,731 games **before 2026-07-01**, and this analysis scores 195,684 clean games
**on or after 2026-07-01**. No game contributes to its own lift, so full-count
lifts are correct and no leave-one-out correction is needed. The script asserts
the table file's own recorded cutoff matches the window start and refuses to run
otherwise (`provenance.leave_one_out`).

**Both windows are the same population.** This is worth stating plainly because
the split is by date and nothing else: every game on both sides of the cutoff
comes from `data/raw/soloq/soloq.db`, queue 420 (ranked solo/duo), the same
Master+ scrape, the same three regions (na/kr/euw), through the same cleaning
rules. Neither window contains a single professional game. This is **not** a
soloq-to-pro transfer study — that was rung 2, which scored pro games with these
tables and failed its gates. Two independent signatures confirm the population:
blue-side win rate is .4754 before the cutoff and .4758 after (pro blue-side WR
on 16.13 internationals was 61%), and the pro corpus for all of 2024–2026 is
about 5,148 games — 124,731 pro games in a three-week window do not exist.

The scoring window is pinned at both ends — `[2026-07-01, 2026-07-25)` — because
the soloq scrape resumed on 07-25 and is appending live. Pinning fixes the
*window*, not the *sample*: the scrape also backfills games inside already-covered
dates, so the counts here describe the db as of **2026-07-28 ~09:15 ET**. By that
evening the same query returned 148,157 pre-cutoff and 234,518 in-window raw
matches. Re-running against a later db state will move the sample sizes and
therefore the third decimal of every rate. The effect sizes are not sensitive to
this; the exact counts are.

## The headline numbers

Raw counted win rates. No model is involved in this table at all — these are the
observed win rates of games in the top and bottom tails of each score
(`raw_win_rate_bands`).

| score | band | bottom WR | top WR | swing (pts) | 95% CI |
|---|---|---:|---:|---:|---|
| **Counters (CTR)** | p90 vs p10 | .4309 | .5181 | **+8.71** | [7.73, 9.70] |
| | p99 vs p1 | .4006 | .5324 | +13.18 | [10.09, 16.28] |
| | p75 vs p25 | .4446 | .5066 | +6.20 | [5.58, 6.83] |
| **Champion strength (ROLE)** | p90 vs p10 | .4039 | .5446 | **+14.07** | [13.09, 15.05] |
| | p99 vs p1 | .3735 | .5861 | +21.26 | [18.20, 24.32] |
| | p75 vs p25 | .4239 | .5265 | +10.27 | [9.64, 10.89] |
| **Duo synergy (SYN)** | p90 vs p10 | .4455 | .5047 | **+5.93** | [4.94, 6.92] |
| | p99 vs p1 | .4262 | .5176 | +9.15 | [6.03, 12.26] |
| | p75 vs p25 | .4555 | .4986 | +4.32 | [3.69, 4.94] |

And the whole composition at once — a three-input logistic fitted on the first
70% of the window and **evaluated only on the last 30%** (58,706 games), so this
is not a model reading its own fit (`combined_draft_score`):

| band | bottom WR | top WR | swing (pts) | 95% CI |
|---|---:|---:|---:|---|
| p90 vs p10 | .4003 | .5558 | **+15.55** | [13.77, 17.34] |
| p99 vs p1 | .3826 | .5952 | +21.26 | [15.68, 26.84] |
| p75 vs p25 | .4271 | .5297 | +10.26 | [9.12, 11.40] |

The held-out decile table is well behaved: predicted and actual win rates agree
within ±1.2 points at every decile with no trend in the residuals, so the swing
is a real gradient and not a tail artifact (`combined_draft_score.deciles`).
Discrimination is nonetheless modest — AUC .5469. Both facts are true at once:
draft moves win probability by a *lot* at the tails and explains very little of
*which* game you are looking at. Soloq outcomes are player-dominated.

## Raw counts vs the logistic fit — where they disagree, and why

Comparing a counted band to a fitted curve is easy to get wrong. A band like
"top 10%" averages the whole tail, so its mean score sits well beyond the 90th
percentile *cut*. Read at the cut, the fit looks 2+ points smaller than the raw
gap; read at the band's own mean score, the two agree closely
(`raw_vs_fit_matched`):

| score | band | raw counted | fit at the same scores |
|---|---|---:|---:|
| CTR | p90 vs p10 | +8.71 | +8.93 |
| CTR | p99 vs p1 | +13.18 | +13.74 |
| ROLE | p90 vs p10 | +14.07 | +14.67 |
| ROLE | p99 vs p1 | +21.26 | +22.49 |
| SYN | p90 vs p10 | +5.93 | +5.96 |

The pattern in the disagreement is consistent and worth stating: the fit runs
slightly **above** the counts, and the gap widens at the extremes (ROLE: +1.2
points at p99 vs p1). The linear-in-logit model mildly over-extrapolates the
tails; reality saturates a little. Where this document quotes one number, it
quotes the counted one.

## What survives once you know how strong the champions are

The important honesty check. A big counter differential usually also means
somebody got a strong champion in their role — so how much of the counter swing
is *counters*, and how much is champion strength wearing a counter costume?

Two readings, same answer. First, conditioning inside the joint fit
(`logistic_implied_joint.*.at_raw_band_means`) — the swing that remains with the
other two scores held at their means:

| score | raw p90 vs p10 | with champion strength known | share surviving |
|---|---:|---:|---:|
| Counters | +8.71 | **+5.71** | ~66% |
| Duo synergy | +5.93 | **+1.11** | ~19% |
| Champion strength | +14.07 | +12.97 | ~92% |

Second, the residual decile tables on held-out games (`decile_tables`), which
need no model of the counter score itself — just actual win rate minus what a
champion-strength-only model expected:

- **Counters show a clean residual slope**: −2.6 points in the bottom decile
  rising to +2.6 in the top, monotone across the middle. Counter advantage
  predicts wins that champion strength alone does not.
- **Duo synergy shows none**: residuals bounce between −1.5 and +1.1 with no
  trend. Its 5.9-point apparent gradient is almost entirely champion strength
  showing through.

That second finding independently reproduces rung 1c — which predicted exactly
this shape ("a counter column, by contrast, would show genuine residual slope")
and priced claimed synergy lift at ~6 cents on the dollar — on games that
experiment never touched. A U.GG-style synergy rating is mostly telling you the
champions are good, not that the pair is.

## Anchors: is 8.7 points a lot?

**Side selection** (`anchors.side_selection`). Blue side wins 47.58% in this
sample, so blue is spotting 2.4 points and the red-minus-blue gap is **4.85
points**. Nobody drafts around this, because nobody can — side is assigned. It is
the reference class for "an edge everybody accepts as material."

- Counter differential, raw p90 vs p10: **1.8× the side gap**
- Counter differential, champion-strength-conditioned: **1.18×**
- Whole composition, p90 vs p10: **3.21×**

**Team strength** (`anchors.elo_equivalents`, `anchors.team_strength_v09`). Under
the standard 400-point Elo curve, each swing equals the rating edge that would
produce the same expected score against an even opponent:

| edge | swing (pts) | Elo-equivalent |
|---|---:|---:|
| Side selection | 4.85 | ~17 |
| Counters, p90 vs p10 (raw) | 8.71 | ~30 |
| Counters, p90 vs p10 (conditioned) | 5.71 | ~20 |
| Whole composition, p90 vs p10 | 15.55 | ~55 |
| Whole composition, p99 vs p1 | 21.26 | ~75 |

Scale reference from the pro side: v0.9's outcome baseline (Elo K=10, 374 holdout
games) reached AUC .708 / 65.8% accuracy from **team strength alone**, and adding
the completed draft moved holdout log-loss 0.6144 → 0.6024. Team identity remains
by far the bigger lever in pro play; the draft is a real second-order term.

One deliberate omission: the pro Elo-*gap* distribution was not recomputed, so
there is no "the median LCK game has an X-point gap" line here. Computing it
would mean loading the pro game table, and the frozen EWC July test set stays
untouched by this analysis. The Elo conversions above are analytic curve math,
nothing more.

## Caveats, in plain language

1. **Correlational.** These are win rates conditional on champions that got
   picked, not the effect of intervening on a draft. Players who pick the
   countering champion may differ from players who don't.
2. **Soloq, not pro.** Master+ ranked solo/duo, regions NA/KR/EUW, scored games
   on patches 16.13 (112,533) and 16.14 (83,151). Pro play has a ban phase that
   removes the extremes, coordinated
   play that changes what a counter is worth, and a different champion pool.
   Rung 2 tried to transfer these tables into the pro models and failed its
   gates — an underpowered failure, not a falsification, but the transfer is
   **not** established.
3. **Static tables, later games, and a patch boundary.** A 16.11–16.13 snapshot
   scores games that are 43% 16.14. Meta drift across that boundary is
   unmodelled noise, which biases toward *under*stating the effect — the swings
   here are a floor, not a ceiling.
4. **Tails are not typical.** The honest middle-of-the-road number is the
   quartile band: counters +6.2 points, whole composition +10.3.
5. **No skill term exists in this data.** Every comp feature together reaches
   AUC .547; rung 1b's ceiling with far more columns was .551.

## Is this strong enough to carry a public piece?

Yes, with the framing kept honest. The publishable spine is the comparison, not
the raw magnitude: **a draft edge available to anyone reading a matchup table is
larger than the side-selection handicap the entire community treats as
significant** — 1.8× raw, 1.2× after conditioning on champion strength. The
supporting result is nearly as good: the most-marketed draft statistic in the
game (duo synergy ratings) survives conditioning at ~19%, while counters survive
at ~66%.

Both claims are reproducible from one deterministic script and one JSON, on
195,684 games no lift table ever saw.

## Provenance

| Item | Value |
|---|---|
| Population (both windows) | `data/raw/soloq/soloq.db`, queue 420 ranked solo/duo, Master+ scrape, na/kr/euw. No professional games on either side of the cutoff. |
| Lift tables | `data/processed/soloq_lift_tables.json`, 124,731 clean games before 2026-07-01, EB shrinkage prior 200 |
| Scoring window | 2026-07-01 00:00 → 2026-07-25 00:00 UTC, 195,684 clean games (14 dropped) |
| Db state | counts as of 2026-07-28 ~09:15 ET; the scrape backfills, so later re-runs see a larger sample |
| Blue WR (window / table sample) | .4758 / .4754 |
| Patches | scored games 16.13 (112,533) + 16.14 (83,151); tables 16.11–16.13 |
| Regions | euw 66,844 / kr 64,959 / na 63,881 (queue 420, duration ≥ 300s) |
| Fit / eval split | first 136,978 games fitted, last 58,706 evaluated |
| Table coverage | 98.6% of role matchups found a counter entry; 214 champ@role fallbacks out of 1.96M lookups |
| Leave-one-out | not applied and not needed — window is disjoint from the table sample, asserted at run time |
| Frozen EWC test set | never loaded or scored |
| Outputs | `scripts/analysis_counter_swing.py` → `data/processed/counter_swing.json` |
