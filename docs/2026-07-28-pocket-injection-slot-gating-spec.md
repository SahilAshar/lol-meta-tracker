# Spec: slot-gated pocket injection (follow-up to the v13 guard breach)

**Pre-registered 2026-07-28 ~09:30 ET, before any code for this rung was
written or run. Prior result: `docs/2026-07-28-pocket-conditioning-results.md`.
Script to be written: `scripts/experiment_v14_slot_gated_pocket.py`. Val only —
the frozen EWC July-2026 test set is not touched, before or after.**

## 1. What the prior rung established

v13 injected `opp_usage` / `player_pool` / `player_wr` through a single
zero-init 3d×d projection **shared by all 20 slots**, plus a slot-type-gated
6-scalar output bias. Result: ban top-1 +5.30 pts (5/5 seeds), val loss +0.162
(CI [0.150, 0.173]), pick top-1 **−4.15 pts** (5/5 seeds) → NO-GO on the guard.

The diagnostics pin the mechanism precisely:

- Val NLL decomposes to **+0.326 nats on bans, −0.002 on picks**. All of the
  objective's appetite for these features lives at ban slots.
- Pick top-1 falls at every pick ordinal (−2.2 to −6.1) — not an
  early-slot or early-stopping artifact.
- At pick slots the model's argmax becomes the decision's max-`player_pool`
  candidate 18.3% of the time (baseline 13.1%, truth 9.6%) — it over-commits
  to the roster favourite, which lifts pick top-5 (44.2 → 45.3) and sinks
  pick top-1.

## 2. Hypothesis

**H1.** The pick-slot degradation is caused by the *direct* feature injection
at pick positions, not by the ban-side learning. The shared projection is fit
almost entirely by ban-slot gradients; applied at pick positions with a
weight-tied output head it pushes the hidden state toward the feature-weighted
champion-embedding centroid, which the pick objective is indifferent to
(Δ NLL ≈ 0) but the pick argmax is not.

**H1 predicts:** remove the injection at pick slots and the ban gain survives
roughly intact while pick top-1 returns to baseline.

**H0 (the alternative worth being able to reject).** The degradation comes
from shared encoder/embedding weights being reshaped by ban-slot gradients. If
so, gating the injection will not rescue picks, because the encoder is shared
no matter what is gated.

## 3. Design — and why this one

Two arms, both against the same cached baselines.

**Arm A (primary): `ban_only`.** Feature information reaches ban slots only.
The input-side context is zeroed at pick positions before the projection, and
the output bias is restricted to the ban slot type (3 scalars instead of 6).
Pick positions receive no feature term at all.

**Arm B (secondary, diagnostic): `gated_proj`.** Two separate zero-init input
projections, one per slot type (`feat_proj[slot_type(s)]`, 2×3d×d), output
bias unchanged from v13 (already gated). Both slot types still get an
injection; only the parameters are unshared.

**Why Arm A is primary rather than Arm B.** The obvious reading of "gate by
slot type" is Arm B, and the diagnostics argue it will not work. A separate
pick-side projection is still fit by gradient descent on a loss that is
*indifferent* to it (−0.002 nats). Nothing in the objective pushes it toward
zero, so it is free to drift to exactly the kind of argmax-perturbing solution
that v13 found — unsharing the parameters removes the constraint that caused
the drift but not the pressure-free drift itself. Arm A instead removes the
term the objective never wanted. It is the design the evidence supports, so it
carries the gate; Arm B is run alongside because it is the only clean way to
tell H1 from "parameter sharing specifically" and costs one extra arm.

Note what Arm A does *not* remove: pick slots can still read ban-slot feature
context through causal attention (a pick at position s attends to ban
positions < s). That is legitimate — it is information the drafter has — and
it means Arm A tests "no direct injection at pick positions", not "picks are
feature-blind".

RNG hygiene, unchanged and non-negotiable: every new parameter is created
**after** the base module's construction, zero-init, consuming no RNG, so the
baseline condition reproduces production bit-for-bit. Arm A's and Arm B's
parameter counts differ from v13's, but since all are created post-hoc and
zero-valued, the baseline arm is untouched in all three experiments.

## 4. Pre-registered gate

Evaluated on Arm A. Same 5 paired seeds (16/17/42/7/23), same config
(d192×4L6H), CPU, val only (54 games / 540 bans / 540 picks).

- **Primary:** mean d_ban_top1 ≥ **+1.5 pts** vs baseline — i.e. the ban gain
  must survive the gating, not merely be non-negative.
- **Guard:** mean d_pick_top1 within **±1.5 pts** of baseline.
- **GO requires both.** Either one failing is a NO-GO. Unlike v13 there is no
  "or loss CI excludes 0" alternative path on the primary: the loss criterion
  is what let a pick-destroying model pass a ban-shaped test, and this rung is
  specifically about the top-1 behaviour.

Reported but not gating: d_loss with 10,000-resample bootstrap 95% CI, the
ban/pick NLL decomposition, and the Stage-B conditional table (join
`ban_attribution_perban.parquet`) for the right-reason check. The v13 result
sets the bar for the right-reason reading: gains concentrated in B1–3, and
whether the POCKET cell moves more than v13's +2.3 pts.

**Arm B is descriptive.** It does not get a GO/NO-GO of its own and cannot
promote; it exists to interpret Arm A. If Arm A passes and Arm B also passes
with a larger ban gain, that is a finding to write down and re-test, not a
result to promote on this run — selecting the better of two arms on 54 val
games is exactly the multiplicity this note is here to prevent.

**Pre-committed interpretations:**

| Arm A picks | Arm A bans | Reading |
|---|---|---|
| within ±1.5 | ≥ +1.5 | H1 confirmed → GO, rung is promotable pending Sahil's approval |
| within ±1.5 | < +1.5 | Gating rescues picks but the ban gain needed the pick-slot pathway → NO-GO, and the ban gain was partly a shared-representation effect |
| outside ±1.5 | any | H1 rejected in favour of H0: the damage is shared-weight reshaping. Next step would be a pick-side stop-gradient or a separate ban head, not more gating |

## 5. Protocol

1. Reuse the cached baseline runs in `data/processed/expcache_pocket/`
   (`baseline_seed<S>.json` + `_probs.npz`, all 5 seeds present). They are the
   same lineage, same config, same split; the script's existing cache path
   loads them by name, so the baseline arm costs zero compute.
2. Hard-assert the seed-16 baseline canary **3.5855** and the `allclose`
   identity of all 5 baseline val prob stacks against `expcache_attr` — from
   cache this is a check that the cache is the run we think it is.
3. Assert split invariants: cutoff 2026-07-15, train 5,043g/100,836d, val
   54g/1,080d, vocab 171.
4. Train Arm A and Arm B, 5 seeds each, ~2.5–3.5 min per run on CPU
   (~30 min total). Under `caffeinate`.
5. Features and standardization identical to v13 (train rows only, same three
   features) — this rung changes the injection geometry and nothing else.
6. Output `data/processed/slot_gated_pocket_rung.json` + per-run caches in
   `data/processed/expcache_slotgate/`, then a results doc.

## 6. Scope

No production change. `train_draft_model_v08.py`, blend weights, the
rung-2/soloq lane and `data/raw/soloq/` are untouched. A GO here is a
recommendation for Sahil's approval ledger, not a promotion.
