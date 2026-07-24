# Results: v0.8.1 blend re-sweep with the meta-aware transformer

**Written 2026-07-24 ~14:00 ET.** Val only — no test look. Script:
`scripts/blend_resweep_timesignal.py`, run on a GitHub Codespace
(4-core/16GB Linux) because lightgbm cannot load locally (libomp missing).
Raw results: `data/processed/blend_resweep_timesignal.json`.

Transformer side: the 5-seed meta-aware ensemble's cached eval probs from
the blind-test run (`expcache_timesignal_blind/`), val slice only —
bit-identical to the tested lineage. GBM side: full production 10-model
ensemble (5 seeds × clf/ranker) refit on the train split in the codespace.
Grid and selection rule identical to `experiment_v08.py` §3–4
(w ∈ {0, .25, .5, .75, 1}, val top-1 per type, ties → lower w).

## Sweep (val top-1/3/5, n=1080; 540 picks / 540 bans)

| w_transformer | All | Picks | Bans |
|---|---|---|---|
| 0.00 (GBM alone) | 13.2/32.0/44.2 | 13.2/28.9/41.9 | **13.3**/35.2/46.5 |
| 0.25 | 14.5/36.1/48.7 | 16.5/37.4/48.3 | 12.6/34.8/49.1 |
| 0.50 | 15.0/36.6/50.6 | **18.7**/38.0/51.1 | 11.3/35.2/50.0 |
| 0.75 (current) | 14.9/35.6/50.2 | 18.3/40.0/51.5 | 11.5/31.3/48.9 |
| 1.00 | 13.7/31.6/45.1 | 18.5/38.3/49.3 | 8.9/24.8/40.9 |

Selected per type: **picks w=0.5 (18.70), bans w=0.0 (13.33)**.

## Reading it honestly

1. **Bans: confirmed at w=0.0.** GBM alone still dominates every mixed
   weight. No change.
2. **Picks: no promotion-grade evidence to move off 0.75.** The selected
   0.5 beats 0.75 by 0.37 points = **2 decisions out of 540**, with 1.0
   in between (18.5). The real finding is flatness: with the meta-aware
   transformer the pick blend is insensitive to w across 0.5–1.0
   (18.3–18.7), where the old transformer needed the GBM's help. Top-3
   actually prefers 0.75 (40.0 vs 38.0).
3. **Cross-platform caveat:** the codespace GBM refit (Linux, pandas 3.0,
   numpy 2.5) lands visibly different val numbers from the stored local
   blocks (GBM-alone val picks 13.2 here vs 11.9 stored; bans 13.3 vs
   13.9). Only within-sweep comparisons are meaningful here; nothing in
   this file should be compared directly against stored local metrics.

## Recommendation

Keep the v0.8.1 weights (picks 0.75, bans 0.0). The actionable upgrade
from this whole arc is **swapping the meta-aware transformer into the
v0.8.1 production pipeline** — already justified by the rung
(val, 5/5 seeds) and the blind test (picks 12.8 → 15.4, bans 7.6 → 8.2,
both components of the blend improve or hold). That means porting the
meta injection into `train_draft_model_v08.py`'s transformer lineage and
refitting locally; the blend weights stay put. Weekly north-star grading
then measures the effect prospectively — no further test-set looks.
