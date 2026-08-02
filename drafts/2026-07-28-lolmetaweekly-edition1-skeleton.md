# LoLMetaWeekly — Edition 1 skeleton (debut ~Aug 3–4, 2026)

**Status: SKELETON — structure and rules, not copy.** Written 2026-07-28.
Defines the recurring format so Editions 2–12 are assembly, not invention.
The shelved EWC debut thread (`drafts/2026-07-20-meta-entering-summer-debut-thread.md`)
stays shelved; its *voice* and its stat-provenance appendix are the model to copy.

Week 1 coverage window: **LPL from Jul 22 · LEC from Jul 24 · LCS from Jul 25 ·
LCK from Jul 29**. Edition 1 is the first cross-league report.

---

## 1. The recurring format

**Title pattern**
- Thread + Reddit: `LoLMetaWeekly #N — <the week's one-line hook>`
- The hook is a claim with a number in it, never a topic. "Nobody has beaten
  the Poppy ban yet" over "Week 1 meta roundup".

**Section order** (identical every week — the cadence is the product):

| # | Section | Thread | Reddit | Notes |
|---|---|---|---|---|
| 1 | The hook + what happened | 1–2 tweets | 1 para | The week's headline claim |
| 2 | **Prediction ledger — last week graded** | 2 tweets | table | §2. Skipped in Ed. 1 (nothing to grade yet) |
| 3 | Meta movement (pick/ban shifts WoW) | 2–3 tweets | table + 1 para | From `daily_report.py` output |
| 4 | **Surprise detection — what the model couldn't see** | 2–3 tweets | 2–3 paras | §3 |
| 5 | **Regional predictability split** | 1–2 tweets | table + 1 para | §4 |
| 6 | **This week's predictions (pre-registered)** | 1–2 tweets | table | §2, posted BEFORE the games |
| 7 | Provenance footer | 1 tweet (link) | full block | §5 |

**Target length**
- Twitter thread: **9–12 tweets.** Ed. 1 runs short (no ledger to grade) — aim 9.
- Reddit (r/leagueoflegends, r/CompetitiveLoL): **700–1,100 words**, tables inline,
  provenance block at the bottom, link to the repo doc for anything longer.
- Rule inherited from the shelved thread: every stat is traceable to a committed
  file. If it cannot be cited, it does not ship.

**Cadence**: post the graded ledger + report on Mon/Tue; post the *next* week's
pre-registered predictions in the same thread, timestamped, before that week's
first game. One post per week, both platforms, same numbers.

---

## 2. Prediction ledger

The differentiator. Two ledgers, because the model can honestly do two different
things and blurring them would be the easiest way to lose credibility early.

### Ledger A — pre-registered, posted before the games

v0.8.1 predicts *the next decision in a draft*. Most of its predictions therefore
need the draft to already be underway — which cannot be published in advance. The
subset that genuinely can be:

- **Slot-1 predictions** (blue-side first ban) need zero draft context. Publish the
  model's top-5 for each league's week, plus its probability on each.
- **Aggregate pre-registrations** derivable from the model's opening distribution:
  "champion X appears in the first three bans of ≥60% of LCK games this week."

Publish as a timestamped table, before the week's first game:

| League | Prediction | Model p | Result | Hit? |
|---|---|---:|---|---|
| LCK | Blue first ban ∈ {A, B, C, D, E} | .— | | |
| LEC | Poppy first-3-banned in ≥60% of games | .— | | |

### Ledger B — frozen-weights retrospective, graded the following week

The full next-pick scorecard across every decision of the week's games. The
no-peeking guarantee is the **frozen model artifact and its data cutoff**, not the
posting timestamp — state it exactly that way, every week:

> "These predictions come from v0.8.1, weights frozen `<date>`, trained on data
> through `<cutoff>`. It has never seen a 2026 summer game."

**Grading rules** (fixed in Ed. 1, never renegotiated mid-season):

- **top-1 hit**: the realized pick/ban is the model's single highest-probability
  candidate.
- **top-3 hit**: the realized choice is in the model's top 3. (Report top-5 too;
  top-3 is the headline.)
- Picks and bans are graded **separately and always both shown** — v0.8.1 is two
  different models under the hood (picks: 0.75 transformer / 0.25 GBM; bans: pure
  GBM), and hiding that would be dishonest about where accuracy comes from.
- Decisions involving a champion outside the model's vocabulary are **excluded and
  counted separately**, never scored as misses.
- Two baselines are published alongside every ledger, computed on the same
  decisions: **meta baseline** (most-picked-recently) and **team baseline** (this
  team's own history). Beating the meta baseline is the bar. On the model's last
  blind test (EWC main event, 1,000 decisions): model **16.0 / 33.1 / 44.6**
  top-1/3/5 vs meta baseline **9.9 / 26.6 / 38.0** and team baseline
  **5.9 / 13.1 / 20.1**.

**Running season tally** — one table carried forward every edition, cumulative:

| Edition | Decisions | Top-1 | Top-3 | vs meta baseline (top-3) | Ledger A hits |
|---|---:|---:|---:|---:|---|
| #1 | | | | | |
| … | | | | | |
| **Season** | | | | | |

Never quietly drop a bad week. A ledger that only shows wins is a tier list.

---

## 3. Surprise detection — "what the model couldn't see"

Each week, the 2–3 drafts the model misjudged worst, framed as discovery rather
than model failure: a high-surprise draft is where a team did something the last
two years of pro play do not predict.

**The metric, concretely:**

- **Decision surprise** = −log₂ *p*(realized choice) in bits. A choice the model
  gave 50% is 1 bit; 6% is ~4 bits.
- **Game surprise** = mean decision surprise across that game's 20 decisions.
  Rank games by this; take the top 2–3.
- Alongside each, name **the single highest-surprise decision** in that game — that
  is the sentence people quote.
- Report the model's own top-3 for that decision next to what actually happened.
- Exclusions as in §2 (out-of-vocabulary champions are flagged, not scored).

**Editorial rule**: every surprise gets one sentence of "here's what the model was
expecting and why" — the value is the contrast, not the number. And when a surprise
turns out to be a *loss*, say so; "the model was surprised and the team lost" is a
better story than pretending every innovation works.

---

## 4. Regional predictability split

Which league does the model read best? Published as a fixed weekly table:

| League | Decisions | Top-1 | Top-3 | Mean surprise (bits) | Δ vs season avg |
|---|---:|---:|---:|---:|---:|

Editorially: the least predictable league is the most interesting one, and that
framing survives whichever league it is.

**⚠️ Honest flag — this does not exist yet.** There is no per-league model-accuracy
breakdown anywhere in the repo; the metrics JSONs are split by decision type and
test set, not by league. It is a small script (group the ledger's per-decision
scores by `league`), but it is **net-new work before Aug 3**, not a lookup.

What *does* exist per-league today, and is a legitimate Ed. 1 substitute if the
split slips: the **ban-motive mixture** from ban attribution
(`data/processed/ban_attribution.json` → `stage_A.mixture_by_league`) — LEC is the
most pocket-driven region (POCKET .483, META .456), LCS the most meta-driven
(META .504, POCKET .444), LCK .491/.461, LPL .483/.462. "LEC bans the player, LCS
bans the patch" is a real, citable Edition 1 section that needs zero new modeling.

---

## 5. Provenance footer pattern

Ends every edition, both platforms (Twitter: one tweet + link to the repo doc):

> **Model** v0.8.1 (picks 0.75 transformer / 0.25 GBM; bans pure GBM), weights
> frozen `<YYYY-MM-DD>`, trained on `<N>` pro draft decisions from `<years>`.
> **Data** Oracle's Elixir, games `<start>` → `<end>`, patch `<X.YZ>`.
> **This edition** `<N_games>` games / `<N_decisions>` decisions across
> `<leagues>`. Baselines: meta `<x/y/z>`, team `<x/y/z>`.
> **Method + every number**: `<repo link>`.

Rules: sample sizes always shown next to rates; any n < 20 is labelled "small
sample" inline; patch boundaries called out explicitly, because a week that spans
one invalidates week-over-week comparisons.

---

## 6. What must exist by Aug 3 — TODO, with blockers flagged

**🔴 BLOCKER 1 — there is no v0.8.1 artifact and no inference path.**
`data/processed/` holds `draft_model.joblib`, `draft_model_v05.joblib`,
`draft_model_v06.joblib` — the v0.7 multi-year ensemble and the v0.8 transformer
are trained inside their experiment scripts and only their *metrics* are saved.
Nothing in the repo can score a new game. Options, in order of preference:
1. Add a `--save` path to the v0.8.1 training run → persist GBM ensemble +
   5 transformer seeds + blend weights → write `scripts/predict_draft.py`
   (input: game/draft rows; output: per-decision top-k + probabilities).
2. Retrain weekly and accept the ±1.5pt single-fit noise band — cheaper to build,
   but it makes the ledger's "frozen weights" claim false. **Not acceptable for
   the differentiator**; option 1 or the column launches without Ledger B.
Also note the stored v0.8.1 was Linux-trained; local refits differ at the
~0.2–1.5pt level (documented platform wobble). Whichever machine produces the
frozen artifact, say so in the footer.

**🟡 BLOCKER 2 — no per-league accuracy breakdown** (§4). Net-new, small, but real.
Fallback content identified above.

**🟢 Ready — the data pull.** `scripts/daily_report.py` (download → analyze →
charts → report) already runs daily via GitHub Actions at 9:23 AM ET and writes
`reports/latest.md`, with Monday dated snapshots to `reports/`. Week-1 games flow
in automatically. Section 3 (meta movement) is essentially free.

**Ordered checklist:**

| # | Item | Est. | Owner note |
|---|---|---|---|
| 1 | Persist v0.8.1 artifact + write `predict_draft.py` | 3–5h | **Blocker for Ledger B** |
| 2 | `scripts/weekly_ledger.py`: score a date range, emit per-decision probs, top-k hits, baselines, per-league split, surprise ranking | 3–4h | Consumes #1; produces §2/§3/§4 in one pass |
| 3 | Ledger A generation (slot-1 / aggregate pre-registrations) | 1–2h | Needs #1 |
| 4 | **Post Ledger A before LCK week 1 (Jul 29)** | — | ⚠️ **Timing risk: 1 day away.** If #1 and #3 don't land by Jul 28 EOD, Ed. 1 ships with Ledger B only and Ledger A pre-registers *week 2* instead. Say that plainly in the post rather than backdating anything. |
| 5 | Confirm Oracle's Elixir has all four leagues' week-1 games by Aug 3 | 15m | LCK Jul 29 start → ~3 days of games; verify OE lag |
| 6 | Draft Ed. 1 copy from this skeleton | 2–3h | |
| 7 | Posting logistics: @lolmetatracker handle confirmed live; public home for the repo doc links (GitHub README anchor); Reddit account age/karma check for r/CompetitiveLoL self-posts | 1h | ⚠️ Handle status was still open in the shelved thread — **verify first**, everything else assumes it |

**Realistic read:** Ledger B + sections 1, 3, 4(fallback), 5 are achievable by
Aug 3. Ledger A pre-registration for week 1 is very likely to slip past Jul 29;
plan for it to pre-register week 2. Item #7 is the only thing that can stop the
debut outright.

---

## 7. Where the draft-swing ("heresy") piece slots in

Mission-1 result, `docs/2026-07-28-counter-swing-analysis.md`: a draft at the 90th
percentile of counter advantage wins 51.8% vs 43.1% at the 10th — an **8.7-point
swing, 1.8× the entire blue/red side handicap** (4.85 points) — and the whole
composition swings 15.6 points, ~3.2× side selection. Measured on 195,684 soloq
games no lift table had seen. **Strong enough to carry a feature.**

**Recommendation: Edition 3 (~Aug 17–18) as a standalone feature**, with a
one-tweet teaser in Edition 2. Reasoning:
- The claim is contrarian, so it should land on an account that has *already*
  published two weeks of self-grading. Credibility first, heresy second.
- It is soloq-only, and rung 2 failed to transfer the tables into the pro models
  (an underpowered null, not a falsification). The piece must be explicit that it
  is a soloq result — which reads as rigour after two graded editions, and as
  hedging if it is the debut.
- It needs a chart (decile curve: actual WR vs champion-strength-predicted WR,
  counters vs synergy side by side) — that is a build, not a write-up.

Do **not** write it this session. When it is written, the spine is the comparison
to side selection, not the raw magnitude, and the synergy result
(claimed duo-synergy lift survives conditioning at ~19%, counters at ~66%) is the
supporting act.
