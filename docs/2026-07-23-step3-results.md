# Step 3 results: soloq→pro embedding transfer — NO-GO

**2026-07-23 ~21:15 ET.** Executes the plan in
`docs/2026-07-22-soloq-data-research.md` §6 per
`docs/2026-07-23-step3-execution-handoff.md`. Script:
`scripts/experiment_v09_soloq_transfer.py`; full per-run output:
`data/processed/soloq_transfer_step3.json` (+ `expcache_step3/` run cache).

## Question

Does initializing the pro draft transformer's `champ_emb` from the rung-0
soloq embedding (purity 0.723, gate passed) beat random init on pro
validation? 5 paired seeds [16, 17, 42, 7, 23], production config d192x4L6H,
cutoff 2026-07-15, 14-day val (54 games / 1,080 decisions), EWC test frozen.
Injection: fixed orthonormal 128→192 projection (numpy seed 1234), rows
rescaled to fresh-init norm 0.02·√192 ≈ 0.2771, 168/168 champions matched.
Torch RNG untouched by injection, so per seed both conditions saw identical
init (except the 168 champion rows), dropout, and batch order.

## Result: NO-GO

| seed | random loss | soloq loss | d_loss | random top1 | soloq top1 | d_top1 |
|-----:|------------:|-----------:|-------:|------------:|-----------:|-------:|
| 16 | 3.5855 | 3.5322 | **+0.0533** | .1389 | .1306 | −0.83 pts |
| 17 | 3.5623 | 3.6332 | −0.0709 | .1222 | .1083 | −1.39 pts |
| 42 | 3.5244 | 3.5126 | **+0.0118** | .1167 | .1278 | +1.11 pts |
| 7 | 3.5654 | 3.6462 | −0.0808 | .1213 | .1130 | −0.83 pts |
| 23 | 3.5766 | 3.6335 | −0.0569 | .1185 | .1120 | −0.65 pts |

- mean d_loss = **−0.0287** (soloq slightly *worse*), 10k-bootstrap 95% CI
  **[−0.0721, +0.0202]** — straddles 0; sign count 2/5 positive.
- mean d_top1 = **−0.52 pts** — nowhere near the +1.5 pt GO bar.
- Sanity anchor held exactly: random seed 16 best val loss 3.5855 ==
  `embedding_evolution_v08_demo.json`.

Neither GO criterion met; deltas flip sign across seeds, mean is mildly
negative. This is a real null(-to-negative) answer, reported straight.

## Observations (not spin)

- The three losing soloq runs early-stopped much sooner (21–24 epochs vs
  32–47 for their random pairs), consistent with the injected geometry
  steering optimization into an early plateau rather than accelerating it.
- The two winning soloq runs (16, 42) show it *can* help, but the effect is
  unstable across seeds — soloq draft-role geometry evidently doesn't
  transfer reliably into a weight-tied head trained on 100k pro decisions;
  the pro data is sufficient to learn its own embedding from scratch.
- The §5.6 frozen-embedding ablation was **not** run: it is reserved for an
  ambiguous headline (CI straddling 0 with all deltas positive). With 2/5
  positive and a negative mean, the headline is not ambiguous.

## Ledger

- Purity gate (Step 2): 0.723 ≥ 0.55 — the soloq embedding is *internally*
  sensible; transfer value to pro draft prediction is what failed.
- Rung 0 verdict: soloq scrape → embedding pipeline works end-to-end, but
  embedding transfer is not a fruitful path for v0.9. Candidate next uses of
  the soloq data, if any, should target features (e.g. champion pick/win
  priors) rather than representation transfer.
