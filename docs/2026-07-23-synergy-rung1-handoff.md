# Execution handoff: synergy rung 1 — do pair interactions predict soloq wins?

**Written 2026-07-23 ~21:45 ET by the Fable session that ran Step 3. Audience:
the agent that will EXECUTE rung 1.** Spec and rationale:
`docs/2026-07-23-soloq-synergy-research.md` §5. Every data anchor below was
verified against the live db tonight.

**Mission:** on held-out soloq games, does adding champion-pair interaction
features to a win-prediction model beat champion main effects alone? This is
the falsifiable gate for the whole "synergy lift" direction. Soloq-only — no
pro data, no pro models, and the frozen EWC test set is not even in the same
dataset. CPU-minutes of compute.

---

## 1. Data (verified anchors — recompute nothing, sanity-check against these)

Source: `data/raw/soloq/soloq.db` (gitignored, 122MB — never commit it),
table `matches`, `state='done'`, payload JSON has `participants` (10 of:
`champ`, `team` 100/200, `pos`, `win`) and `bans`; table columns carry
`game_creation` (ms epoch), `game_version`, `queue_id`, `duration`, `region`.

Clean filter: `queue_id=420 AND duration>=300 AND state='done' AND payload IS
NOT NULL`, then require exactly 10 participants. Expected counts:

- **76,995 clean games** (from 78,171 done; 1,176 remakes + malformed drop).
  The scraper may still be running — if the count is higher, fine; assert
  `>= 76,995` and record the actual number.
- Window 2026-06-08 → 2026-07-23+, patches 16.11 (1,365) / 16.12 (14,961) /
  16.13 (40,459) / 16.14 (21,379). Regions na/kr/euw ≈ even thirds.
- **173 champions**, median 3,813 games each, min 691 (soloq internal names —
  `MonkeyKing` etc.; no name bridge needed, this rung never touches pro).
- **Blue win rate 0.4789** — red-favored. If your loaded data shows blue
  ≈0.52, you have a team-id bug (`team==100` is blue, `win` is per
  participant and constant within a team).
- A handful of rows have `game_creation=0` (epoch 1970) — sort them out of
  the chronological split by dropping them (they are ≤7 games).

## 2. Design (one row per game, predict blue win)

Features are **signed indicators**, so one logistic regression sees both
teams symmetrically:

- **Champion main effects** (173 cols): +1 if champ on blue, −1 if on red,
  0 otherwise. (A champ can't be on both — no bans-of-picks in the same game.)
- **Same-team pair features** (14,878 cols = C(173,2)): +1 if the pair is on
  blue, −1 if on red, 0 otherwise. ~10 nonzero per team per game.
- **Cross-team counter pairs** (14,878 cols, unordered {A,B}): +1 if A on
  blue faces B on red where A < B alphabetically... do NOT hand-roll the
  sign convention loosely — define `ctr{A,B} = +1` when the alphabetically
  first champ is on blue, `−1` when on red. Antisymmetry is then encoded by
  the single signed column. 25 nonzero per game.
- Intercept captures side advantage. Use scipy sparse matrices
  (`csr_matrix`); the design is ~77k × ~30k with ~45 nonzeros/row — trivial.

Models (sklearn `LogisticRegression`, `penalty='l2'`, `solver='liblinear'`
or `lbfgs` with sparse input):

- **M0**: intercept only (side baseline; holdout log-loss ≈ 0.692 minus the
  side edge).
- **M1**: + champion main effects.
- **M2**: M1 + same-team pairs.
- **M3**: M2 + counter pairs.

Sweep `C ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}` **per model** on the val
split only (precedent: `experiment_v09_outcome_baseline.py` sweeps C on val).
Expect pairs to want much smaller C (heavier shrinkage) than main effects —
if the best C for M2/M3 is at the sweep edge, extend the sweep.

## 3. Splits (chronological, house discipline)

By `game_creation`: **train = first 70%, val = next 15%, holdout = last 15%**
of clean games in time order (record the resulting date boundaries in the
output). Hyperparameters (C) chosen on val; **holdout scored exactly once**
per final model. No games from the future in any feature — these are all
static indicators, so the only leakage risk is the split itself.

## 4. Gate (from research doc §5)

Compute per-game holdout log-loss for M1 and M2 (and M3), then the paired
per-game difference `d_i = ll_M1_i − ll_M2_i`:

- **GO** if the 10k-resample bootstrap 95% CI of mean `d` excludes 0 in M2's
  favor (same for M1 vs M3 — report both; either passing is a GO).
- **NO-GO** if within noise. Report AUC and accuracy as secondary color, but
  the gate is log-loss only. A null is a real answer — report it straight.

## 5. Face-validity artifact (report, not gate)

From the **train split only**, compute empirical-Bayes shrunk pair lifts:
observed pair win rate minus expected (from the two champs' train win rates,
via log-odds sum or simple average — state which), shrunk toward 0 with a
beta-ish prior of strength ~200 effective games (tune nothing here; it's
descriptive). Emit top-20 synergy pairs and top-20 counter pairs with lifts
and game counts. Sanity-read them: known bot-lane duos and famous counter
matchups should be recognizable. If the list is all sub-100-game pairs,
shrinkage is too weak — that's a bug, not a discovery.

## 6. Protocol & repo etiquette

1. Script: `scripts/experiment_v10_synergy_rung1.py`, follows the house
   pattern (docstring stating question/gate, verified-anchor asserts,
   incremental cache unnecessary — the whole thing is minutes).
2. Output: `data/processed/synergy_rung1.json` — per-model {C chosen, val
   and holdout log-loss/AUC/acc}, paired bootstrap CIs, verdict, top-pair
   tables, provenance (game count, date boundaries, patch mix, blue WR).
3. Runtime guardrail: if a fit exceeds ~5 min, stop and reconsider C grid /
   solver — nothing here should be slow.
4. `docs/ROADMAP.md` and `artifact/*` belong to other sessions — never stage
   them. Run `date` before writing dates. Commit/push only when Sahil
   approves; offer the results note first.
5. Region ablation, role-aware pairs, recency weighting: **out of scope**
   for rung 1 (research doc §6). Do not run them preemptively.
