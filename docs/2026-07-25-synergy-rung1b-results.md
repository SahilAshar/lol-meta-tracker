# Synergy rung 1b results: role-aware pairs and slot permutations — GO

**2026-07-25 ~12:57 ET.** Executes `docs/2026-07-23-synergy-rung1b-spec.md`
after rung 1's role-blind NO-GO. Script:
`scripts/experiment_v10_synergy_rung1b.py`; full output:
`data/processed/synergy_rung1b.json`. Run against the final scrape pull
(329,703 done games, db closed out 2026-07-25 16:18 UTC) — 2.2× the spec's
preferred ≥150k trigger, so nulls here read as *falsified*, not
"underpowered".

## Question

Does role-aware structure — champ×role slotting (A), same-team role-pair
synergy on the four priority vectors (B), lane counters (C) — predict
held-out soloq wins beyond champion main effects? Plus the descriptive
slot-permutation table (D).

## Data

323,401 clean games (23 dropped for blank/dup roles — `pos` is as clean as
the spec claimed), blue WR 0.4755, 173 champions, patches 16.11–16.14
(mode 16.13), regions na/kr/euw. Chronological 70/15/15: train 226,380
(→ Jul 13), val 48,510 (→ Jul 19), holdout 48,511 (→ Jul 25). C swept per
gated model on val only; all optima interior; holdout scored once per
model. Feature retention (≥50 train games) computed on train only:
605 champ@role columns, 8k same-team pair columns across the four vectors,
4.6k counter columns. Full ladder runtime: 41 s.

## Result: GO — slotting and counters are real; same-team pair synergy is not

| model | features | cols | holdout LL | AUC | acc |
|-------|----------|-----:|-----------:|----:|----:|
| M1 | champ main effects | 173 | 0.68971 | .5404 | 53.8% |
| A | champ×role main effects | 778 | 0.68873 | .5478 | 53.9% |
| B | + same-team role-pair vectors | 8,169 | 0.68857 | .5477 | 54.0% |
| C | + lane counters | 12,786 | 0.68807 | .5507 | 54.1% |

Paired 10k-bootstrap on per-game holdout log-loss (positive = challenger
better):

- **M1 vs A: PASS** — mean +0.00098, 95% CI [+0.000441, +0.001524].
  Slotting matters: Galio-mid and Galio-top are genuinely different
  effects. This clears the interpretation gate for B/C.
- **A vs B: FAIL** — mean +0.000164, CI [−0.000246, +0.000568]. At 323k
  games this is the real falsification rung 1 couldn't deliver: same-team
  pair synergy, even restricted to the four priority vectors, does not
  beat role-aware main effects overall.
- **B vs C: PASS** — mean +0.000496, CI [+0.000274, +0.000713]. Lane
  counters carry real signal, exactly where rung 1's whisper pointed
  (cross-team, not same-team). AUC .5477 → .5507.

Rung gate (B-over-A or C-over-B): **GO**, via counters.

## Vector ablations of B (descriptive; 3-point C sweep around B's optimum)

Positive delta = full B beats B-without-this-vector, i.e. the vector helps:

| vector | mean delta | 95% CI | reading |
|--------|-----------:|--------|---------|
| MIDDLE+JUNGLE | +0.000166 | [+0.000051, +0.000281] | only vector whose CI excludes 0 |
| TOP+JUNGLE | +0.000046 | [−0.000058, +0.000153] | null |
| BOTTOM+UTILITY | +0.000014 | [−0.000130, +0.000156] | null — the canonical duo, dead |
| JUNGLE+UTILITY | −0.000096 | [−0.000221, +0.000033] | null, leans harmful |

The one surviving same-team vector is **mid-jungle**, not the bot duo.
Caveat: four comparisons at 95%, one significant — treat as "the vector to
keep if any", not an independent discovery. The bot-duo null is the more
striking result: with ~453k bot-duo slots over 1,439 measurable pairs,
there was power to find it, and it isn't there beyond the champs' own
role-specific strength.

## D. Slot permutations (train only, EB shrinkage 200g, ≥150g per slotting)

1,306 champion pairs qualify with ≥2 slottings. Permutation clearly moves
lift for flex-relevant pairs — top spreads ~6–9 WR points:

- **Kled+Senna**: Senna-ADC +5.5 pts vs Senna-support −3.1 (spread .086)
- **Akali+Sylas**: Akali-top/Sylas-mid +3.8 vs Akali-top/Sylas-jg −4.2
- **Elise+Kaisa**: Elise-support +2.4 vs Elise-jungle −5.4
- **Riven+Sylas**: Riven-mid/Sylas-jg +6.7 vs Riven-top/Sylas-mid −0.9
- **Nautilus+Pantheon**: Pantheon-jg +1.9 vs Pantheon-mid −5.7

**Camille+Galio** (the motivating example): the dominant observed slotting
is Camille-support + Galio-mid (938 train games, +1.7 shrunk lift); the
"classic" Camille-top + Galio-mid is slightly *negative* (293g, −0.5);
Camille-support + Galio-top (+1.1/62g) and Camille-top + Galio-support
(+1.0/52g) are positive but thin. Directionally exactly the
slot-shuffling story: the pair's value concentrates in one permutation.

## Consequence

- **Rung 2 reopens with a narrower payload than hoped**: champ×role main
  effects and lane-counter features (mid-jungle same-team optionally) are
  the transferable objects — not generic same-team pair synergy, which is
  now falsified twice (role-blind at 77k, role-restricted at 323k).
- **The draft-tool lever is real**: the D table is directly consumable —
  permutation-dependent lifts exist at meaningful magnitude for flex
  pairs, and slot-shuffling late draft is worth encoding.
- Ceiling check stands: even model C only reaches AUC .551 — soloq
  outcomes remain player-dominated; these are second-order terms.
