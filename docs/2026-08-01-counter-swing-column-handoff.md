# Handoff: counter-swing analysis + LoLMetaWeekly Edition 1

**Written 2026-08-01.** Covers work done in a session on **2026-07-28** and its
state four days later. Written for a reader (human or agent) with no memory of
that session. Two independent workstreams; neither is blocked by the other.

**Read this first if you are picking up cold:** the analysis is finished and its
conclusion is solid, but its numbers were computed against lift tables that a
parallel session has since rebuilt at 1.8× density. Nothing is wrong; one
re-run is pending. See §3.

---

## 1. What the session was asked to do

1. **Counter-swing quantification.** Earlier experiments (rungs 1b, 1c, 2) proved
   draft-composition effects are real in soloq but reported them in per-game
   log-loss — numbers no reader can feel (counters +0.000624, synergy +0.000061).
   Translate the same effects into win-rate points, to produce the headline number
   for a public "how much does draft actually matter" piece.
2. **LoLMetaWeekly Edition 1 skeleton** against the ~Aug 3–4 debut named in
   `docs/ROADMAP.md` "North Star".

Both were delivered. Everything is uncommitted.

---

## 2. Mission 1 — counter-swing: what was built and what it found

### Files

| File | State |
|---|---|
| `scripts/analysis_counter_swing.py` | new, untracked, runs in ~5 s |
| `data/processed/counter_swing.json` | new, untracked — every number below traces here |
| `docs/2026-07-28-counter-swing-analysis.md` | new, untracked — the plain-language write-up that feeds the public piece |

### Method in one paragraph

Load the pre-built soloq lift tables. Score real soloq games with three
blue-minus-red scalars: **CTR** (sum of the five same-role counter lifts), **SYN**
(sum of duo lifts over the four priority vectors), **ROLE** (sum of champion-in-role
win rates). Then count win rates in the tails of each scalar. No model training.
Deterministic — no sampling, analytic binomial SEs. A three-input logistic is
fitted on the first 70% of the window and read only on the last 30%, so the
combined number is never a model reading its own fit.

### The epistemic trap, and how it was avoided

The lift tables are built *from game outcomes*. Score a game that is inside the
table's own sample and its own result leaks into its own lift, inflating every
gradient. This killed rung 1c's first run, which repaired it with leave-one-out
counts.

This analysis avoids it structurally instead: tables built from games **before
2026-07-01**, games scored **on or after 2026-07-01**. Disjoint by construction,
so full-count lifts are correct and no LOO is needed. The script asserts the table
file's own recorded cutoff equals the window start and refuses to run otherwise.
**If you change the window or the tables, that assert is the thing that protects
you — do not weaken it.**

### Population — settle this before re-deriving anything

Both windows are the **same** population: `data/raw/soloq/soloq.db`, queue 420
(ranked solo/duo), Master+ scrape, regions na/kr/euw, same cleaning rules. The
split is by date and nothing else. **Neither window contains a professional game.**

This came up mid-session as a genuine misreading — that the pre-07-01 games were
pro. They are not. Three confirmations: identical queue and regions on both sides;
blue-side WR .4754 before vs .4758 after (pro blue-side WR on 16.13 internationals
was 61% — a 47.5% rate is a soloq fingerprint); and the entire pro corpus for
2024–2026 is ~5,148 games, so 124,731 pro games in three weeks cannot exist.

The soloq-to-pro study is **rung 2** (`docs/2026-07-25-rung2-transfer-results.md`),
a different experiment, which scored pro games with these tables and failed its
gates. The soloq/pro meta gap limits **who this finding generalizes to**; it is not
a flaw inside the measurement.

### Results (tables of 124,731 games; window 195,684 games)

Raw counted win rates, top decile vs bottom decile — no model involved:

| Scalar | Bottom 10% | Top 10% | Swing | 95% CI |
|---|---:|---:|---:|---|
| Counters (CTR) | .4309 | .5181 | **+8.71 pts** | [7.73, 9.70] |
| Champion strength (ROLE) | .4039 | .5446 | +14.07 | [13.09, 15.05] |
| Duo synergy (SYN) | .4455 | .5047 | +5.93 | [4.94, 6.92] |
| **Combined draft score** (held-out 30%) | .4003 | .5558 | **+15.55** | [13.77, 17.34] |

Typical-game band (p75 vs p25): counters +6.20, combined +10.26.

**Anchor:** blue side wins 47.58% in this sample, a 4.85-point red-minus-blue gap.
Counters alone are **1.8×** that; the whole composition is **3.2×**. In Elo terms,
~30 and ~55 points.

**The honest deflation — this is the part that makes the piece credible.** Condition
on champion strength and ask what survives:

| Scalar | Raw swing | Strength-conditioned | Survives |
|---|---:|---:|---:|
| Counters | +8.71 | +5.71 | ~66% |
| Duo synergy | +5.93 | +1.11 | ~19% |
| Champion strength | +14.07 | +12.97 | ~92% |

A second, model-free check agrees: group held-out games by counter advantage and
compare actual wins to what a strength-only model expected. Counters show a clean
monotone residual (−2.6 → +2.6 points). Synergy shows no trend at all. That
independently reproduces rung 1c's "claimed synergy realizes at ~6 cents on the
dollar" on games rung 1c never touched.

Discrimination is modest throughout (combined AUC .5469). Both things are true:
draft moves win probability a lot at the tails, and explains little about which
game you are looking at. Soloq outcomes are player-dominated.

### Verdict for the public piece

**Strong enough to carry it.** The spine must be the comparison, not the
magnitude: *a draft edge anyone can read off a matchup table beats the
side-selection handicap the whole community treats as material.* The synergy
result is the supporting act. Mandatory caveats: soloq Master+ not pro;
correlational not causal; rung 2's transfer to pro failed its gates; the scored
games are patches 16.13–16.14 while the tables are 16.11–16.13 (this mismatch
biases the effect **down**, so the numbers are a floor).

---

## 3. ⚠️ The one open item on Mission 1: the tables moved

A parallel session re-ran rung 2 on 2026-07-31
(`docs/2026-07-31-rung2-rerun-results.md`) and **rebuilt
`data/processed/soloq_lift_tables.json` in place**:

| | at time of analysis | now |
|---|---:|---:|
| clean games in tables | 124,731 | **224,064** (1.8×) |
| blue WR | .4754 | .4777 |
| cutoff | 2026-07-01 | 2026-07-01 (**unchanged**) |

The cutoff is unchanged, so `analysis_counter_swing.py` **still runs and its
assert still passes**. But `counter_swing.json` and the write-up describe tables
that no longer exist on disk. Anyone re-running today gets different numbers and
will think something is broken.

That same session also created a frozen point-in-time copy,
**`data/raw/soloq/soloq-snapshot-20260731.db`** (688,180 done games), precisely to
make builds reproducible against a live scrape. This solves a problem the original
run flagged: the live `soloq.db` backfills into already-covered dates, so sample
counts drift between runs.

**Recommended refresh (not yet done, deliberately — it changes every published
number and should be a conscious choice):**

1. Point `analysis_counter_swing.py` at the snapshot instead of the live db —
   currently `DB = DATA_RAW / "soloq" / "soloq.db"` at module scope. Prefer adding
   a `--db` flag over editing the constant, matching how the rung 2 re-run
   overrode `soloq_lift_tables.DB` at call time without editing the file.
2. Re-run. Expected sample: **~362,759 raw matches** in the same
   `[2026-07-01, 2026-07-25)` window, up from 195,684 — 1.85× denser, scored
   against 1.8× denser tables.
3. Optionally extend the window to 2026-08-01, which adds ~86,523 more matches.
   Only do this if you also state the wider patch mix; the current write-up's
   patch caveat is specific.
4. Update the numbers in `docs/2026-07-28-counter-swing-analysis.md`. The doc
   already carries a db-state stamp explaining why counts move.

**Expectation, not a promise:** effect sizes should barely move — same population,
same window, more games — while confidence intervals tighten. If a headline number
shifts by more than ~1 point, something else changed and is worth investigating
before publishing.

---

## 4. Mission 2 — LoLMetaWeekly Edition 1

**File:** `drafts/2026-07-28-lolmetaweekly-edition1-skeleton.md` (new, untracked).

It specifies the recurring format so later editions are assembly, not invention:
title pattern, section order, Twitter (9–12 tweets) and Reddit (700–1,100 words)
targets, the prediction-ledger design and grading rules, the surprise metric, the
regional split, and a provenance footer. Voice and stat-provenance style are copied
from the shelved EWC debut thread
(`drafts/2026-07-20-meta-entering-summer-debut-thread.md`, which stays shelved).

### The three findings that matter more than the skeleton

**🔴 BLOCKER — v0.8.1 cannot make a prediction today.** Only `draft_model.joblib`,
`draft_model_v05.joblib`, and `draft_model_v06.joblib` are persisted. The v0.7
multi-year ensemble and the v0.8 transformer are trained *inside* their experiment
scripts, which save metrics and discard the weights. There is no inference entry
point anywhere in the repo. Fix: add a save path to the v0.8.1 training run
(GBM ensemble + 5 transformer seeds + blend weights), then write
`scripts/predict_draft.py`. Est. 3–5 h. **The prediction ledger is the column's
entire differentiator, so this gates the concept, not just one section.**
Note the stored v0.8.1 was Linux-trained and local refits wobble ~0.2–1.5 pts;
whichever machine produces the frozen artifact must be named in the footer.

**⚠️ DESIGN FLAW in "predictions published before games."** v0.8.1 predicts *the
next decision in a draft*, so most of its output requires the draft to be underway
and cannot be published in advance. The skeleton splits the ledger:
- **Ledger A** — genuinely pre-registrable: slot-1 (blue first ban) needs zero
  draft context, plus aggregate claims like "champion X is first-3-banned in ≥60%
  of LCK games this week."
- **Ledger B** — the full retrospective scorecard. Its no-peeking guarantee rests
  on **frozen weights and a stated data cutoff**, not on a posting timestamp. Say
  that explicitly every week.

Grading rules are fixed in the skeleton (top-1 / top-3 definitions, picks and bans
always reported separately because they are two different models, out-of-vocabulary
champions excluded and counted rather than scored as misses, meta and team
baselines published alongside every ledger).

**🟡 The regional predictability split does not exist.** No per-league accuracy
breakdown exists anywhere in the repo — metrics JSONs split by decision type and
test set, never by league. It is a small script (group per-decision scores by
league) but it is net-new work. Fallback that needs zero modeling:
`data/processed/ban_attribution.json` → `stage_A.mixture_by_league`, i.e. *"LEC
bans the player, LCS bans the patch"* (LEC POCKET .483 / META .456; LCS META .504 /
POCKET .444).

**🟢 Ready:** the data pull. `scripts/daily_report.py` already runs daily in CI at
9:23 AM ET and writes `reports/latest.md`. Week-1 games arrive on their own.

### Schedule reality as of 2026-08-01

Week 1 began **LPL Jul 22 · LEC Jul 24 · LCS Jul 25 · LCK Jul 29** — all four
leagues are already playing. The debut target is **Aug 3–4, two days out.**
Pre-registering week 1 via Ledger A is no longer possible; that window closed
Jul 29. Edition 1 should ship Ledger B only and pre-register **week 2**, stating
so plainly rather than backdating anything.

**The item that can stop the debut outright:** the `@lolmetatracker` handle was
still unconfirmed in the shelved thread and has not been verified since. Check it
first — everything else assumes it.

### Where the contrarian piece goes

Mission 1's result is strong enough for a feature. Recommendation: **Edition 3
(~Aug 17–18)** as a standalone, with a one-tweet teaser in Edition 2. A contrarian
soloq-only claim lands better on an account that has already published two weeks of
grading itself, and it needs a chart (decile curve: actual vs strength-predicted
WR, counters against synergy) which is a build, not a write-up. **Not written yet,
deliberately.**

---

## 5. Repo state and rails

**Uncommitted, ready to stage (this session's work only):**

```
scripts/analysis_counter_swing.py
data/processed/counter_swing.json
docs/2026-07-28-counter-swing-analysis.md
drafts/2026-07-28-lolmetaweekly-edition1-skeleton.md
docs/2026-08-01-counter-swing-column-handoff.md   (this file)
```

**Do not stage, not ours:** `docs/ROADMAP.md` (a parallel session owns it and grew
it during the session), `artifact/*`, and everything from the parallel lanes —
pocket conditioning, capacity sweep, rung 2 re-run, `expcache_*`, `logs/`,
`scripts/scrape_health.sh`. Parquet, npz, and db files stay untracked;
`data/raw/soloq/` is gitignored.

**Rails observed throughout:** the frozen EWC July test set was never loaded or
scored by anything written here. The soloq db was opened strictly read-only
(`mode=ro`) because a scrape was active. Nothing was pushed. Nothing was committed.

---

## 6. Open decisions for whoever picks this up

1. **Refresh the counter-swing numbers against the 07-31 snapshot?** (§3) Improves
   density and makes the run reproducible, at the cost of rewriting every number in
   the write-up. Do this **before** the piece is published, not after.
2. **Extend the scoring window to 2026-08-01?** Adds ~86.5k games and a wider patch
   mix. Only with an updated patch caveat.
3. **Build the v0.8.1 inference path, or ship Edition 1 without Ledger B?** (§4)
   This is the real fork in the road for the column.
4. **Ship the regional split, or use the ban-motive fallback for Edition 1?**
5. **Verify the `@lolmetatracker` handle.** Do this first; it is cheap and it gates
   everything.
