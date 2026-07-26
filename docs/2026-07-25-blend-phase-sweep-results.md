# Results: phase-aware ban blend sweep (Mission 1)

**Written 2026-07-25 ~23:00 ET. Handoff:
`docs/2026-07-25-next-rungs-handoff.md` Mission 1; evidence base:
`docs/2026-07-25-ban-attribution-results.md` Stage B. Script:
`scripts/blend_phase_sweep.py`, run locally — GBM side from the cached
codespace refit (`gbm_val_scores.parquet`), transformer side from the
blind-test ensemble caches (`expcache_timesignal_blind`). Raw:
`data/processed/blend_phase_sweep.json`. VAL ONLY — no test look.**

**Harness check:** before reading any new cell, the sweep's all/picks/bans
rows were hard-asserted equal (all 15 rows, 4 decimals) to the stored
codespace re-sweep `blend_resweep_timesignal.json`. They reproduce exactly —
the local reconstruction is bit-faithful to the run that set the current
weights.

## Sweep (val top-1/3/5; bans split B1–3 n=324, B4–5 n=216)

| w_transformer | Bans (all) | Bans B1–3 | Bans B4–5 |
|---|---|---|---|
| 0.00 (GBM alone) | **13.3**/35.2/46.5 | **16.4**/38.9/50.0 | 8.8/29.6/41.2 |
| 0.25 | 12.6/34.8/49.1 | 13.0/35.5/49.1 | 12.0/33.8/49.1 |
| 0.50 | 11.3/35.2/50.0 | 9.6/32.1/50.0 | 13.9/39.8/50.0 |
| 0.75 | 11.5/31.3/48.9 | 7.1/25.9/47.2 | **18.1**/39.4/51.4 |
| 1.00 | 8.9/24.8/40.9 | 4.9/17.9/35.8 | 14.8/35.2/48.6 |

Picks re-verified in the same pass: selected w=0.5 (18.70), same
near-flatness across 0.5–1.0 as the re-sweep; no new information, 0.75
production stands per the re-sweep's reasoning.

## Selected per cell (ties → lower w)

| Cell | w | top-1 |
|---|---|---|
| bans B1–3 | **0.0** | 16.4 |
| bans B4–5 | **0.75** | 18.1 |
| composite bans (phase-aware) | — | **17.0** |
| reference: single-weight bans w=0.0 | — | 13.3 |

**+3.7 val ban top-1 points (20 of 540 bans) over the production
single-weight scheme**, beating the naive Stage-B cell arithmetic (+2.4).
The two cells move in opposite directions and each is close to
monotone in its preferred direction — B1–3 degrades steadily as transformer
weight rises (16.4 → 4.9), B4–5 rises steadily to a 0.75 peak (8.8 → 18.1
→ 14.8 at 1.0). That is the clean signature of Stage B's finding
(phase-1 bans belong to the tabular model, phase-2 bans to attention),
not a jagged cell-picking artifact.

## Reading it honestly

1. **Post-hoc split caveat (carry everywhere this number goes):** the
   B1–3/B4–5 split was chosen after seeing Stage B's conditional cells on
   these same 540 val bans, and the selected weights are chosen on the same
   540 bans they're evaluated on. The split has clean a-priori motivation
   ("phase 2 has twelve slots of context to attend over") and adds only
   1 df, but this is weaker evidence than a pre-registered split. The +3.7
   is an optimistic estimate of the true gain.
2. **The B4–5 peak at 0.75 sits on n=216**; 18.1 vs 14.8 (w=1.0) is 7
   decisions. The direction (blend in the transformer at phase 2) is
   well-supported; the exact weight 0.75 is soft, as blend weights here
   always have been.
3. Cross-platform caveat inherited from the re-sweep: all numbers in this
   file are on the codespace-refit GBM scale and compare only within this
   sweep family, not to stored local metric blocks.

## Recommendation

Adopt per-type-×-phase weights — picks 0.75 (unchanged), bans B1–3 w=0.0,
bans B4–5 w=0.75 — **bundled with the still-pending meta-injection port**
into `train_draft_model_v08.py` (`docs/2026-07-24-blend-resweep-results.md`
§Recommendation), since the transformer side of these weights IS the
meta-aware ensemble. Weekly north-star grading then measures the effect
prospectively. **Both remain on the open approvals ledger — nothing ships
without Sahil's sign-off.**
