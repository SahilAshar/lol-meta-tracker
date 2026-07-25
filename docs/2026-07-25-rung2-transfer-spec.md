# Spec: rung 2 — transferring role-aware soloq lifts into the pro models

**Written 2026-07-25 ~13:05 ET, after rung 1b's GO
(`docs/2026-07-25-synergy-rung1b-results.md`). Audience: the agent that
will execute rung 2 in a fresh session. Status: SPECCED, READY — the
soloq db is final (329,703 done games) and rung 1b already justifies the
transfer payload. NOT dependent on rung 1c** (the 1-df same-team synergy
test, `docs/2026-07-25-synergy-rung1c-spec.md`): 1c only decides whether
one optional extra feature (the SYN scalar) joins the arsenal below. If
1c has produced results by the time this runs, read them; otherwise
proceed without.

## What rung 1b changed about this rung

The original rung-2 sketch (`docs/2026-07-23-soloq-synergy-research.md`
§5) was "recompute the GBM's `pair_syn`/`pair_ctr` from soloq lifts."
Rung 1b narrowed the payload:

- **Transfer-worthy:** champ×role main effects (M1→A passed) and
  same-role lane counters (B→C passed, CI [+0.000274, +0.000713]).
- **NOT transfer-worthy:** generic same-team pair synergy — falsified
  role-blind at 77k (rung 1) and role-restricted at 323k (rung 1b).
  Do not build a soloq-informed `pair_syn`; treat the existing pro
  `pair_syn` as a deletion candidate pending 1c.

Two independent arms; **either passing = rung 2 GO.** Arm 1 serves the
mimic north star, arm 2 the coach north star — a GO here makes the
mimic-vs-coach conversation (ROADMAP) live, per the research memo.

## Shared plumbing

- **Soloq lift tables**, computed once from the final db
  (`data/raw/soloq/soloq.db`, 329,703 done; loader/cleaning rules from
  `scripts/experiment_v10_synergy_rung1b.py` — 323,401 clean games):
  - `soloq_wr(c@r)`: champ@role WR, EB-shrunk toward the champ's overall
    WR, prior 200 games.
  - `soloq_ctr(a, b | r)`: same-role counter lift (construction as 1c
    spec: observed vs expected from the two champ@role WRs, shrink 200,
    antisymmetric). Unobserved → 0.
- **Look-ahead rule:** any table used to score a pro decision/game must
  be built only from soloq games with `game_creation` strictly before
  that pro game's date — in practice, build ONE static table from soloq
  games before the pro **val window start**, use it everywhere, and
  state this in provenance. Soloq coverage begins 2026-06-08; it cannot
  inform pro games before that (see per-arm notes).
- **Name bridge:** `PRO_TO_SOLOQ` in
  `scripts/experiment_v09_soloq_transfer.py:46` (168/168 champs matched
  there; verify coverage again and report misses).
- **Role bridge:** pro data has positions (Oracle's Elixir `position`:
  top/jng/mid/bot/sup) → map to soloq `pos`
  (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY).

## Arm 1 — draft GBM (mimic): soloq-informed candidate features

**Current state.** The production v0.7 ensemble
(`scripts/train_draft_model.py`: 5 seeds [16,17,42,7,23], hist-GBM clf +
LGBM lambdarank, equal-weight rank blend) scores val (14 days pre-cutoff
2026-07-15, 1,080 decisions) at top1 .1324 / top3 .3278 / top5 .4407.
Its `pair_ctr` feature (`draft_dataset.py:367-434` `PairStats`) is a
trailing-window pro pair WR with (w+2)/(g+4) shrinkage — estimated from
only ~5k pro games.

**Intervention.** Two new candidate-level features (computed in
`draft_dataset.py`'s decision loop, where `p_open`, `picks_so_far`, and
the pair features already live):

1. `soloq_ctr_score(cand)` = Σ over the opponent's locked picks of
   `soloq_ctr(cand, opp_pick | r)`, role-resolved via the candidate's
   role-share distribution × `p_open` (mirror how `role_need` handles
   role uncertainty; document the exact weighting chosen).
2. `soloq_role_wr(cand)` = Σ_r P(cand plays r | shares, p_open) ·
   `soloq_wr(cand@r)` — the role-aware champ-strength prior.

Optionally (only if 1c PASSED): `soloq_syn_score(cand)` analogous to 1,
over own locked picks.

**Variants** (each vs the v0.7 baseline, same seeds, same splits):
- V1: baseline + both new features (additive).
- V2: V1 with pro `pair_ctr` **replaced** by `soloq_ctr_score` (does
  soloq subsume the thin pro estimate?).
- Report per-family (clf vs ranker) and ensemble numbers.

**Timing caveat (mandatory in writeup):** soloq coverage starts
2026-06-08; pro train decisions before that get zero/neutral values for
the new features. The features can only differentiate on late-window
data — val is entirely post-07-01, so the gate is still clean, but
train-time sparsity may mute what the GBM learns. If the result is an
ambiguous null, note "re-run when soloq spans a full split" as the
follow-up, not a rerun of the same window.

**Gate:** ensemble val top-3 improvement over baseline with a paired
bootstrap 95% CI excluding 0 (bootstrap over the 1,080 val decisions,
paired per decision; 10k resamples). Report top-1/top-5 alongside.
Frozen EWC test set: untouched, as always.

## Arm 2 — outcome head (coach): soloq comp features in win prediction

**Current state.** v0.9 outcome baseline
(`scripts/experiment_v09_outcome_baseline.py`,
`data/processed/outcome_baseline_v09.json`): completed-draft win
prediction, splits train ≤03-15 / val ≤05-15 / holdout after (374
games), frozen EWC main event excluded. Best model M2a (Elo + champ
indicators): holdout LL 0.60235 vs Elo-only 0.61439; gate passed with CI
[+0.0026, +0.0215].

**Intervention.** Add to M2a's feature set, per game:

1. `CTR` = Σ over the 5 lanes of `soloq_ctr(blue champ, red champ | r)`
   (blue perspective, already signed).
2. `ROLE_WR` = Σ blue `soloq_wr(c@r)` − Σ red `soloq_wr(c@r)`.
3. Optionally (1c PASS only): the SYN scalar.

Standardize on train. Sweep the logistic C on val as v0.9 did; holdout
scored once; paired bootstrap on per-game holdout log-loss, M2a vs
M2a+soloq.

**Timing/patch caveat (mandatory, sharper than arm 1's):** the v0.9
holdout games run from mid-May; soloq lifts are measured on patches
16.11–16.14 (June 8 onward). There is real patch misalignment for the
early holdout and *zero* soloq-informed differentiation is possible for
games before 06-08 unless the features are computed patch-blind (they
are — the tables pool patches; champ meta shifts are the noise term).
Primary eval: the standard v0.9 splits, unchanged, for comparability.
Secondary slice (report, not gated): holdout restricted to games on or
after 2026-06-08. If primary is null but the secondary slice is
positive, say exactly that — it reads "transfer works where the data
overlaps" and the follow-up is a re-split, not a NO-GO.

**Gate:** M2a+soloq beats M2a on holdout log-loss, 95% CI excluding 0.

## Verdict rules

- **GO** = arm 1 gate OR arm 2 gate passes. Name which.
- Per-arm nulls are real answers. Distinguish in the writeup: "falsified
  transfer" vs "underpowered/misaligned window" — the timing caveats
  above pre-register which reading is allowed for which arm.
- A GO reopens rung 3 (counterfactual critic) *contingent on the
  mimic-vs-coach conversation*, which the ROADMAP marks as unhad. Do not
  start rung 3; end the session by putting that conversation on the
  table.

## Protocol

- Scripts: `scripts/experiment_v12_rung2_gbm.py` (arm 1) and
  `scripts/experiment_v12_rung2_outcome.py` (arm 2) — v11 is taken by
  the ban time-signal work. Outputs `data/processed/rung2_gbm.json`,
  `data/processed/rung2_outcome.json`, provenance blocks mirroring 1b
  (plus: soloq table cutoff date, bridge coverage, per-arm timing
  caveats).
- Arm 1 requires regenerating `draft_decisions.parquet` with the new
  columns — keep the existing columns byte-identical (spot-check a
  handful of rows against the current parquet before training).
- Expected runtime: arm 2 is CPU-minutes; arm 1 is the dataset rebuild
  (~tens of minutes historically) + 10 GBM fits. Nothing needs a GPU.
- Never stage `docs/ROADMAP.md` or `artifact/*`; run `date` before
  writing dates; commit/push only on Sahil's approval.
