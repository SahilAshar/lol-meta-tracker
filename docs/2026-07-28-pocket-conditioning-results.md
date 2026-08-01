# Results: team/player conditioning rung (pocket features, input-side)

**Run executed 2026-07-25 ~23:09–23:31 ET by a prior session and left
unwrapped; written up 2026-07-28 ~09:15 ET. Spec:
`docs/2026-07-25-next-rungs-handoff.md` Mission 2. Script:
`scripts/experiment_v13_pocket_conditioning.py`. Raw:
`data/processed/pocket_conditioning_rung.json`, diagnostics (written today
from the cached probs, no retraining):
`scripts/pocket_conditioning_diagnostics.py` →
`data/processed/pocket_conditioning_diagnostics.json`. Val only — the frozen
EWC July-2026 test set was not touched.**

## Verdict: NO-GO on a guard breach — but the ban side is the largest live lead in the repo

| Gate | Pre-registered threshold | Result | Pass |
|---|---|---|---|
| Primary A: mean d_ban_top1 | ≥ +1.5 pts | **+5.30 pts** (5/5 seeds positive) | ✅ |
| Primary B: d_loss CI excludes 0 | favorably | mean **+0.162**, 95% CI [0.150, 0.173], 5/5 | ✅ |
| Guard: d_pick_top1 | within ±1.5 pts | **−4.15 pts** (5/5 seeds negative) | ❌ |

GO required (A or B) **and** the guard. The guard failed, so the rung does not
promote. Nothing was changed in production; no blend weights, no train script.

Harness integrity: seed-16 baseline best val loss reproduced the canary
**3.5855** exactly, and all five baseline conditions' val probability stacks
were asserted `allclose` against `expcache_attr` — the lineage is unmoved, so
the deltas below are the injection, not drift. Split invariants held (cutoff
2026-07-15, train 5,043g/100,836d, val 54g/1,080d, vocab 171).

## What was run

Three opponent-referenced candidate features — `opp_usage`, `player_pool`,
`player_wr` — standardized on train rows only, built as a
(n_games, 20, vocab, 3) tensor via `SEQ_TO_SLOT` (per (gameid, seq, candidate),
not per game; ban rows reference the opponent's roster, pick rows the own
team's; all windows end the day before the game, so no leakage). Injection,
all new params zero-init after base construction and consuming no RNG:

- **input-side**: `ctx = Σ_v feat[g,s,v,f] · champ_emb[v]`, then
  `x[g,s] += reshape(ctx) @ feat_proj` with `feat_proj` a zero-init 3d×d matrix
  — one projection **shared by all 20 slots**;
- **output-side**: `logits += feat · feat_out[slot_type(s)]` — 6 scalars,
  already gated ban-vs-pick.

5 paired seeds (16/17/42/7/23), d192×4L6H, CPU.

### Per-seed

| seed | val loss (base → feat) | ban top-1 | pick top-1 |
|---|---|---|---|
| 16 | 3.5855 → 3.4101 | 9.44 → 12.04 | 18.33 → 11.30 |
| 17 | 3.5623 → 3.3927 | 7.41 → 13.52 | 17.04 → 13.15 |
| 42 | 3.5244 → 3.3810 | 7.59 → 13.33 | 15.74 → 11.67 |
| 7 | 3.5654 → 3.3940 | 6.85 → 12.59 | 17.41 → 14.44 |
| 23 | 3.5766 → 3.4265 | 7.22 → 13.52 | 16.48 → 13.70 |
| **mean** | **3.5628 → 3.4009** | **7.70 → 13.00** | **17.00 → 12.85** |

Overall top-1 (all 1,080 decisions) moves 12.35 → 12.93, i.e. the ban gain
slightly outweighs the pick loss but the composition is wrong.

## Right-reason check — the phase claim holds, the vector claim does not

Stage-B-style conditional table on the 5-seed ensemble means, joined to
`ban_attribution_perban.parquet` (540 val bans, 0 unmatched). GBM column
carried over from `docs/2026-07-25-ban-attribution-results.md` for scale only —
it is a different refit and is not directly comparable.

| Cell | n | baseline ens | feat ens | Δ top-1 | (GBM, for scale) |
|---|---|---|---|---|---|
| all bans | 540 | 8.3/36.9 | **14.1**/48.7 | +5.7 | 13.3/46.5 |
| META | 291 | 10.0/37.5 | **18.6**/55.7 | **+8.6** | 15.5/50.2 |
| POCKET | 221 | 6.3/34.4 | **8.6**/40.7 | +2.3 | 12.2/45.2 |
| COMP | 28 | 7.1/**50.0** | **10.7**/39.3 | +3.6 | 0.0/17.9 |
| B1–3 × META | 209 | 5.3/32.1 | **14.8**/55.0 | **+9.6** | 17.7/52.1 |
| B1–3 × POCKET | 115 | 1.7/24.3 | **6.1**/34.8 | +4.3 | 13.9/46.1 |
| B4–5 × META | 82 | 22.0/51.2 | **28.1**/57.3 | +6.1 | 9.8/45.1 |
| B4–5 × POCKET | 106 | 11.3/45.3 | 11.3/47.2 | **+0.0** | 10.4/44.3 |
| B4–5 × COMP | 28 | 7.1/**50.0** | **10.7**/39.3 | +3.6 | 0.0/17.9 |

The spec demanded the gain concentrate in **POCKET and B1–3**. Half of that
came true:

1. **Phase: confirmed, emphatically.** The gain lives in B1–3 (META +9.6,
   POCKET +4.3) and in B4–5 META; B4–5 POCKET is flat to the decimal. This is
   exactly the regime Stage B said the transformer was blind in — slots 1–6
   with no draft context to attend over — and the injection is the first thing
   in this lineage to move it. The ensemble's phase-1 META cell (14.8) is now
   within striking distance of the GBM's 17.7, where the baseline sat at 5.3.
2. **Vector: not confirmed.** The gain is META-dominant (+8.6) and only weakly
   POCKET (+2.3). B1–3 POCKET improves 1.7 → 6.1 but the GBM is at 13.9 —
   under half the deficit is closed. The features are roster-derived, yet what
   they mostly bought is meta-flavoured ban accuracy.

That is interpretable rather than contradictory: Stage A already found META and
POCKET "deeply entangled" (banned champs sit at the ~87–94th percentile on
both, multi-motive is 59–71% of bans at ε=10), and `opp_usage` is a
*team-level* trailing usage rate, which is as much a local-meta signal as a
roster signal. The rung supplied a generic "what is hot and who plays it"
prior to a model that had neither, and phase-1 bans of every flavour got
better. The specifically-pocket deficit — one player's signature champion —
is only partly addressed, which is consistent with Stage A's other finding
that the pocket signal is the *player* (0.46–0.47) rather than the team
(0.09). A per-player identity representation, not a per-team usage rate, is
what that cell is still asking for.

One cell to not over-read: COMP (n=28) gains top-1 but loses 10.7 pts of
top-5. At that n a single game moves the number.

## The tension worth investigating — and what it actually is

The flag raised in the handoff was: overall loss improved massively while pick
argmax got worse. The diagnostics resolve it, and the resolution is sharper
than expected. Decomposing val NLL into its ban and pick halves
(`pocket_conditioning_diagnostics.py`, mean over 5 seeds, positive favours
feat):

| | baseline NLL | feat NLL | Δ |
|---|---|---|---|
| ban decisions (540) | 3.685 | 3.359 | **+0.326** |
| pick decisions (540) | 3.441 | 3.443 | **−0.002** |

**The entire 0.162-nat headline gain is bans.** Pick NLL is flat to two
decimal places on every seed. So the tension is not "loss and accuracy
disagree" — it is that at pick slots the injection is *loss-neutral but
argmax-destructive*. It shuffles the top of the pick ranking without changing
how much probability mass sits on the truth.

Two more diagnostics say what the shuffle is:

**The pick damage is uniform, not concentrated.** Pick top-1 by ordinal
(seed mean, baseline → feat): P1 20.7 → 16.3, P2 17.2 → 11.1, P3 16.9 → 13.2,
P4 13.2 → 10.9, P5 17.0 → 12.8. Every ordinal loses 2–6 pts. This rules out
"early context-poor picks only" and rules out an early-stopping artifact
localized to late slots. (Ban gains by ordinal, same table: B1 +13.5, B2 +5.4,
B3 +6.7, B4 +1.3, B5 −0.4 — the phase story again, at ban-ordinal resolution.)

**The injection over-commits picks to the roster favourite.** Share of
decisions whose top-ranked candidate is also that decision's maximum on a
given feature, against the same rate for the champion actually chosen:

| | truth | baseline | feat |
|---|---|---|---|
| picks × `player_pool` | 9.6 | 13.1 | **18.3** |
| picks × `opp_usage` | 8.3 | 12.2 | **16.0** |
| bans × `player_pool` | 12.0 | 7.6 | **11.7** |
| bans × `opp_usage` | 12.2 | 9.8 | **15.1** |

On bans the injection moves the argmax *toward* the truth rate (7.6 → 11.7
against a truth of 12.0). On picks it pushes past a truth rate that was
already lower than where the baseline sat — the model increasingly answers
"the champion this player plays most", which is a good top-5 heuristic (pick
top-5 in fact rises, 44.2 → 45.3) and a bad top-1 one. Pros pick around
their pool; they do not pick the modal champion of their pool.

Mechanically this is one shared `feat_proj` doing two jobs. Ban slots supply
essentially all of the gradient signal that the projection is worth having
(+0.326 nats there, 0.000 at picks), the projection is fit to serve them, and
because it is applied at every position with a weight-tied output head, the
same "push the hidden state toward the feature-weighted embedding centroid"
lands on pick slots, where the objective is indifferent to it and the argmax
is not.

## Assessment

The rung failed its gate and is not promotable as built. But the ban result is
real, paired, 5/5, and larger than anything else currently live: +5.3 pts ban
top-1 with the gain sitting exactly where five sessions of evidence said the
transformer's deficit was. The guard breach has a specific, testable
mechanism, which is the subject of the follow-up rung spec'd in
`docs/2026-07-28-pocket-injection-slot-gating-spec.md`.

Honesty notes to carry forward:

- The follow-up's design was chosen after looking at these val diagnostics.
  Its gate is pre-registered before running, but the architecture is not
  a-priori — it is one iteration of val-informed design on 54 games. Weaker
  evidence than a blind first shot, and the accumulated val selection pressure
  across this lineage (blend phase split, now this) is real. The test set stays
  frozen regardless.
- 540 bans / 540 picks. Cells of n=28 are anecdote; n=115–291 cells are
  directional.
- `opp_usage` being a team-usage rate makes the "pocket" label on this rung
  partly a misnomer. If a future rung wants the player-identity cell
  specifically, it should say so and be measured on B1–3 × POCKET alone.
