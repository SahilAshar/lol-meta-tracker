# Results: slot-gated pocket injection — GO

**Run 2026-07-28 ~09:50–10:20 ET. Spec (pre-registered before the code was
written): `docs/2026-07-28-pocket-injection-slot-gating-spec.md`. Prior rung:
`docs/2026-07-28-pocket-conditioning-results.md`. Script:
`scripts/experiment_v14_slot_gated_pocket.py`. Raw:
`data/processed/slot_gated_pocket_rung.json`. Val only — the frozen EWC
July-2026 test set was not touched.**

## Verdict: GO on the pre-registered gate

| Gate (evaluated on `ban_only`) | Threshold | Result | Pass |
|---|---|---|---|
| Primary: mean d_ban_top1 | ≥ +1.5 pts | **+5.89 pts** (5/5 seeds positive) | ✅ |
| Guard: mean d_pick_top1 | within ±1.5 pts | **−0.26 pts** | ✅ |

Both required, and both hold. The ban gain is not merely preserved by the
gating — it is **larger** than v13's ungated +5.30, so the pick-slot injection
was never buying the ban result in the first place.

| | mean ban top-1 | mean pick top-1 | mean val loss |
|---|---|---|---|
| baseline (v0.8 lineage) | 7.70 | 17.00 | 3.5628 |
| **`ban_only`** (primary) | **13.59** | **16.74** | **3.4015** |
| `gated_proj` (secondary) | 13.07 | 12.56 | 3.4195 |
| v13 ungated, for reference | 13.00 | 12.85 | 3.4009 |

Harness integrity: all five baselines were reused from `expcache_pocket`, with
the seed-16 canary **3.5855** and the `allclose` prob-identity check against
`expcache_attr` asserted on load. Split invariants held (cutoff 2026-07-15,
train 5,043g/100,836d, val 54g/1,080d, vocab 171), and the feature tensor
rebuilt to train statistics identical to v13's to six decimals. Before
spending compute, three properties were verified directly: base parameters
bit-identical to a same-seed `DraftTransformer` (RNG hygiene), zero-init
forward exactly equal to baseline logits, and pick-slot input context provably
zero under randomized weights.

### Per-seed

| seed | d_ban_top1 | d_pick_top1 | d_loss |
|---|---|---|---|
| 16 | +4.26 | +0.19 | +0.194 |
| 17 | +4.07 | −1.67 | +0.183 |
| 42 | +5.56 | 0.00 | +0.118 |
| 7 | +8.15 | −0.37 | +0.168 |
| 23 | +7.41 | +0.56 | +0.143 |
| **mean** | **+5.89** | **−0.26** | **+0.161**, CI [0.138, 0.185], 5/5 |

Only seed 17 breaches the guard band individually (−1.67), and it is the one
seed where the ban gain is smallest. Four of five pick deltas sit within
±0.6 pts.

## H1 confirmed, H0 rejected — the secondary arm did its job

The spec pre-committed to a reading of each outcome. The one that landed:
*Arm A picks within ±1.5 and bans ≥ +1.5 → H1 confirmed → GO.*

`gated_proj` is what makes this diagnostic rather than merely lucky. Giving
pick slots their **own** zero-init projection reproduced v13's failure almost
exactly (pick top-1 −4.44 vs v13's −4.15, breaching on 5/5 seeds) while
gaining bans just as well (+5.37). So:

- The damage was **not** shared-projection parameters. Unsharing them changes
  nothing.
- The damage was **not** shared encoder/embedding weights being reshaped by
  ban gradients (H0). If it were, `ban_only` — which shares exactly the same
  encoder — would have degraded picks too. It did not.
- The damage **is** the direct injection at pick positions, as predicted from
  v13's NLL decomposition. The pick objective is indifferent to that term
  (Δ NLL ≈ 0), nothing constrains it, and it drifts into a solution that
  perturbs the pick argmax. Removing it is the fix; giving it its own
  parameters is not.

This is the spec's central argument surviving a test that could have killed
it, and it is the reason `gated_proj` was worth the extra 15 minutes.

### An unpredicted bonus: picks get *better* through attention

NLL decomposition (mean over 5 seeds, positive favours the arm):

| | d_NLL bans | d_NLL picks |
|---|---|---|
| v13 (ungated) | +0.326 | −0.002 |
| `gated_proj` | +0.294 | −0.007 |
| **`ban_only`** | +0.234 | **+0.088** |

Pick NLL *improves* by 0.088 nats under `ban_only` — the only arm where it
moves at all. Pick slots receive no feature term, so this can only have
arrived through causal attention over ban positions whose representations are
now better informed. The model reads its own improved ban understanding when
predicting picks. Pick top-1 does not move (−0.26), so the gain is calibration
rather than argmax, but it is a real and unforced effect, and it is evidence
the injection is teaching the model something structural rather than
decorating ban logits.

Note the trade: `ban_only` gains *less* ban NLL than the ungated arms (+0.234
vs +0.326) while gaining *more* ban top-1 (+5.89 vs +5.30). The ungated arms
spend some of their capacity on probability mass the argmax never sees.

## Right-reason check — same shape as v13, phase-1 again

5-seed ensemble means, joined to `ban_attribution_perban.parquet` (540 val
bans, 0 unmatched).

| Cell | n | baseline | `ban_only` | Δ | `gated_proj` |
|---|---|---|---|---|---|
| all bans | 540 | 8.3/36.9 | **13.2**/47.4 | +4.9 | 13.3/48.1 |
| META | 291 | 10.0/37.5 | **15.8**/53.9 | +5.8 | 17.2/54.3 |
| POCKET | 221 | 6.3/34.4 | **9.0**/38.9 | +2.7 | 9.0/41.6 |
| COMP | 28 | 7.1/50.0 | **17.9**/46.4 | +10.7 | 7.1/35.7 |
| B1–3 × META | 209 | 5.3/32.1 | **15.3**/55.0 | **+10.0** | 15.8/54.5 |
| B1–3 × POCKET | 115 | 1.7/24.3 | **7.0**/34.8 | **+5.3** | 9.6/34.8 |
| B4–5 × META | 82 | 21.9/51.2 | 17.1/51.2 | **−4.8** | 20.7/53.7 |
| B4–5 × POCKET | 106 | 11.3/45.3 | 11.3/43.4 | +0.0 | 8.5/49.1 |
| B4–5 × COMP | 28 | 7.1/50.0 | **17.9**/46.4 | +10.7 | 7.1/35.7 |

The v13 reading holds and strengthens on the pocket axis:

1. **Phase-1 is where everything happens.** B1–3 META +10.0, B1–3 POCKET
   +5.3. The B1–3 POCKET cell — the single worst cell in the whole attribution
   study, 1.7 top-1 at baseline against the GBM's 13.9 — is now at 7.0, up
   from v13's 6.1. Roughly 40% of that gap is closed, from a model that
   previously had no roster identity at all.
2. **Phase-2 pays a price this time.** B4–5 META drops 21.9 → 17.1, where v13
   *gained* there (+6.1). At n=82 that is four decisions, so it is directional
   at best — but it points the same way as the ban-ordinal split (B5 was the
   one ban ordinal v13 did not improve), and it is the clearest cost visible in
   this run. Phase-2 bans are where the transformer was already winning
   against the GBM; a rung that trades phase-2 for phase-1 is not obviously
   a good deal for the blend, which is why this belongs in the follow-up
   list rather than being waved through.
3. **Vector: still META-leaning**, as in v13. POCKET overall +2.7 vs META
   +5.8. The `opp_usage` caveat from the v13 write-up stands — it is a
   team-level usage rate, so this rung remains only partly "pocket".
4. COMP moves 7.1 → 17.9, but n=28. Three decisions. Ignore it.

One number that cuts the other way and should be said plainly: on the
**ensemble**, `ban_only` (13.2) is slightly *behind* v13's ungated arm (14.1)
and `gated_proj` (13.3) on all-ban top-1, even though its per-seed mean is
ahead (13.59 vs 13.00). Ensembling recovers some of what the ungated arms lose
to variance. The gate is defined on per-seed paired deltas and `ban_only` wins
that comparison cleanly, but if a future production model is an ensemble, this
gap is worth re-measuring rather than assuming.

## What this does and does not license

**Does:** `ban_only` is a genuine, paired, 5/5 improvement in ban top-1 of
+5.89 pts with picks untouched, concentrated in the regime five sessions of
evidence identified as the transformer's blind spot. It is the first change in
this lineage to move B1–3 at all. It goes on the approvals ledger as a
recommended promotion.

**Does not:** nothing is promoted here. `train_draft_model_v08.py`, the blend
weights, the rung-2/soloq lane and `data/raw/soloq/` are untouched. Promotion
needs Sahil's approval and would fold in the still-pending meta-injection port.

Honesty notes:

- The architecture was chosen after reading v13's val diagnostics. The gate was
  pre-registered before this run, and `gated_proj` was included specifically so
  the choice could be falsified — but this is still one iteration of
  val-informed design on 54 games, and the val selection pressure accumulated
  across this lineage (phase blend split, v13, now this) is real. The test set
  stays frozen.
- 540 bans / 540 picks. n=28 cells are anecdote; n=82 is thin; n=115–291 is
  directional.
- The B4–5 META regression is the loose thread. Before promotion it is worth
  knowing whether it is noise or whether the gating costs phase-2 — a
  ban-ordinal-conditioned look at more val games, or a phase-gated injection
  (inject at B1–3 only), would answer it.

## Follow-ups this suggests, in order

1. **Re-run the phase-aware ban blend** (`blend_phase_sweep.json`, Mission 1 of
   the prior handoff) against `ban_only` rather than the old transformer. That
   sweep concluded the transformer was worth weight only in B4–5; this rung
   changes the phase-1 picture entirely, so the optimal weights are almost
   certainly stale.
2. **Phase-gated injection** — inject at B1–3 slots only, on the same 5 paired
   seeds. Directly tests the B4–5 META regression and costs ~15 minutes.
3. **Player identity, not team usage.** Stage A said the pocket signal is the
   player (0.46) not the team (0.09). B1–3 POCKET is at 7.0 against the GBM's
   13.9 — the remaining gap is the one a per-player representation would
   target, and the soloq scrape feeds it.
