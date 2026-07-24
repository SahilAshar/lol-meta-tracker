# Blind test: meta-aware transformer on the frozen EWC July-2026 set

**Written 2026-07-23 ~23:10 ET.** Third and final look at the EWC test set
(v0.8 was the first, v0.8.1 the second) — gated on the
`timesignal_rung.json` GO and Sahil's explicit approval tonight. Scored
once; no further iteration against this test set. Script:
`scripts/blindtest_ban_timesignal.py`; raw results:
`data/processed/timesignal_blindtest.json`.

Protocol mirrors `train_draft_model_v08.py`'s transformer lineage exactly
(full-data vocab 171, 5-seed mean-softmax ensemble, train-split fit,
val early stop) plus the rung's meta injection. Every per-seed val loss
reproduced the rung's numbers to the fourth decimal, so the comparison to
the stored v0.8 blocks is clean; test set verified unchanged (50 games,
1,000 decisions).

## Test top-1 (n=1000; 500 picks / 500 bans)

| Model | All | Picks | Bans |
|---|---|---|---|
| v0.8 transformer (stored) | 10.2 | 12.8 | 7.6 |
| **meta transformer (this test)** | **11.8** | **15.4** | **8.2** |
| v0.7 GBM refit (stored) | 14.9 | 14.8 | 15.0 |
| v0.8.1 per-type blend (stored) | 16.0 | 17.0 | 15.0 |

Top-3/5 moved the same way: picks 28.6→30.0 / 37.4→41.4, bans
16.8→20.2 / 24.4→30.8.

## Reading it honestly

1. **The headline surprise: picks, not bans.** The rung was framed as a ban
   fix, but on blind test the meta injection moved picks +2.6 pts (12.8 →
   15.4, now ahead of the GBM's 14.8) while bans gained only +0.6 (7.6 →
   8.2, still less than half the GBM's 15.0).
2. **The ban result is not a val→test transfer failure — it's an ensembling
   effect.** The rung's +1.9 ban points were per-seed means. At the
   5-seed-ensemble level the val ban gain is +0.56 (8.33 → 8.89), almost
   exactly what test shows (+0.6). Ensembling the baseline seeds already
   recovers most of what the rate features add for bans; the per-seed and
   ensemble pictures diverge, and production runs the ensemble.
3. Ban top-3/5 gains (+3.4 / +6.4 pts) are larger than top-1, consistent
   with the rate prior improving calibration of the ban candidate set more
   than its argmax.
4. The transformer remains the picks specialist and the GBM the bans
   specialist — the v0.8.1 per-type split is still right, but the pick side
   of the blend now has a stronger transformer than the one the 0.75 weight
   was swept for.

## Implication for the deferred item #3

Re-sweep the v0.8.1 blend weights **on val** with the meta-aware
transformer: the pick-side weight was chosen when the transformer's test
picks were 12.8 vs GBM 14.8; with 15.4-grade picks the optimal weight may
shift up, and the blend's 17.0 pick top-1 is the number to beat. Ban side
likely stays GBM-only (0.0) given 8.2 vs 15.0. Input-side rate conditioning
remains the open rung for the ban gap itself. Any new production promotion
gets no further test looks — val selection only, per the flag above.
