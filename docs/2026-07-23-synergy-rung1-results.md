# Synergy rung 1 results: pair interactions in soloq win prediction — NO-GO

**2026-07-23 ~21:55 ET.** Executes
`docs/2026-07-23-synergy-rung1-handoff.md` (spec:
`docs/2026-07-23-soloq-synergy-research.md` §5). Script:
`scripts/experiment_v10_synergy_rung1.py`; full output:
`data/processed/synergy_rung1.json`.

## Question

On held-out soloq games, does adding champion-pair interaction features
(same-team synergy, cross-team counters) to a win-prediction model beat
champion main effects alone? The falsifiable gate for the "synergy lift"
direction. Soloq only — no pro data touched.

## Data (all handoff anchors verified)

76,995 clean games (exact match), blue WR 0.4789 (exact match), 173
champions, patches 16.11–16.14, regions na/kr/euw. Chronological 70/15/15
split: train 53,896 (→ Jul 14), val 11,549 (→ Jul 19), holdout 11,550
(→ Jul 23). C swept per model on val only; holdout scored once per model.
The original C grid bottomed out at 0.003 with M2's optimum on the edge, so
the sweep was extended down to 3e-5 per the handoff's instruction; every
final C (0.003 for M1's near-tie region and both pair models) is interior.

## Result: NO-GO

| model | features | cols | holdout LL | AUC | acc |
|-------|----------|-----:|-----------:|----:|----:|
| M0 | intercept (side) | 0 | 0.6926 | .500 | 52.4% |
| M1 | + champ main effects | 173 | 0.69010 | .5440 | 53.2% |
| M2 | + same-team pairs | 15,051 | 0.69008 | .5431 | 53.3% |
| M3 | + counter pairs | 29,929 | 0.68942 | .5483 | 53.6% |

Paired 10k-bootstrap on per-game holdout log-loss (positive = pairs better):

- **M1 vs M2**: mean +0.000017, 95% CI [−0.000711, +0.000739] — dead null.
- **M1 vs M3**: mean +0.000676, 95% CI [−0.000347, +0.001710] — leans
  positive, CI straddles 0.

Neither gate passes. The val sweeps agree: M2 never beat M1 on val at any
C — regularization wants the same-team pair weights at zero.

## Observations (not spin)

- **This is substantially a power problem, not proof synergy is fictional.**
  ~1.54M same-team pair slots spread over 14,878 pairs ≈ ~100 games per
  pair on average. Detecting a genuine 2-point pair WR lift needs a few
  thousand games *of that pair*; only the most common handful of pairs are
  even measurable at this scale. The design diluted whatever signal exists
  across an almost-entirely-unmeasurable feature space.
- **Counter pairs carry the whisper.** M3's lean-positive delta and its AUC
  bump (.5440 → .5483) fit the game-theoretic prior: lane counters are a
  real mechanism, and cross-team pairs are denser (25/game vs 20). A
  role-aware design (lane-vs-lane counters, bot-duo synergy using the
  `pos` field) would concentrate power where the mechanism lives instead of
  spreading it over C(173,2) — that's the natural rung 1b if this is
  revisited.
- **The ceiling was low all along.** Even perfect champion-level knowledge
  only gets AUC .544 in soloq — outcomes are dominated by the ten players,
  not the ten champions. Pair effects are second-order corrections on top
  of that already-small term.
- Face-validity tables (train-only, EB shrinkage ~200 games) are sane, not
  degenerate: top synergies are mostly 150–400-game pairs (Poppy+Syndra
  +.056/390g, Camille+Nidalee +.053/404g); top counter by volume is
  Jayce+Yone (+.065/712g). Not dominated by sub-100-game noise.

## Consequence

Per research doc §5, rung 2 (transferring soloq pair lifts into the pro
GBM's `pair_syn`/`pair_ctr`) does **not** proceed. Combined with today's
Step 3 embedding-transfer NO-GO, both soloq→pro *model-transfer* routes are
dead at current scale. The surviving soloq uses from the 07-22 memo:
**meta-rate / patch-strength features** (soloq as a leading indicator of
pro meta) and continued scraping for scale. Role-aware rung 1b is a cheap
optional revisit once the dataset is several times larger.
