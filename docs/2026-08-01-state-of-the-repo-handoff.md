# Handoff: state of the repo after the pocket-conditioning arc (2026-08-01)

**Written 2026-08-01 ~10:15 ET. Successor to
`docs/2026-07-25-next-rungs-handoff.md` — all three of its missions are
executed and committed. Read this first, then the results docs it points
at. The purpose of this file is that a fresh session can reconstruct where
the model stands, what is proven-but-unported, and what is genuinely open.**

## Non-negotiable rails (inherited, updated where marked)

- **The frozen EWC July-2026 test set has had its three looks (v0.8,
  v0.8.1, timesignal blind test). It is never scored again.** Everything
  below is val-only (54 games / 1,080 decisions / 540 bans). Promotions are
  measured prospectively by weekly north-star grading, not by a fourth look.
- Split invariants: cutoff 2026-07-15, train 5,043g/100,836d, val
  54g/1,080d, vocab 171. CPU only (MPS breaks masked_fill(-inf));
  `train_loss=inf` is cosmetic — do not "fix" masked_loss.
- Paired-seed RNG hygiene for any transformer change: new params created
  AFTER base module construction, zero-init, consuming no RNG, so the
  baseline condition reproduces production bit-for-bit.
- **Canaries** (seed 16 best val loss): baseline **3.5855**, meta rung
  **3.4623**, v14 `ban_only` pocket injection **3.3911**. The v14 number is
  the strongest harness check for pocket-lane work — it pins the feature
  tensor and injection code, not just the base lineage
  (`docs/2026-07-28-capacity-sweep-results.md`). If a canary misses, stop.
- **RAIL UPDATE — lightgbm now imports locally.** The rung-2 session
  installed libomp on this Mac (~07-25), so GBM fits no longer require the
  codespace. The cross-platform caveat survives: Linux-refit and
  darwin-refit GBM numbers are not comparable to each other or to stored
  blocks — compare only within one run. The codespace
  (`effective-space-couscous-q57gg77r5rh4pp6`, stopped, synced) is now
  optional; its fate is on the approvals ledger.
- Never push without Sahil's explicit approval (commits are fine — that is
  the standing "offer results as commits" mandate). Never stage
  `docs/ROADMAP.md` or `artifact/*` (other sessions own them). Parquets,
  npz/pt caches, and `logs/` stay untracked. Run `date` before writing any
  date into any file.

## What landed since the 07-25 handoff (chronological, with commits)

| Commit | What | Verdict |
|---|---|---|
| `72df364` / `e8f7c25` | Synergy 1c GO (thin 1-df aggregate) / rung 2 soloq→pro transfer | 1c GO; rung 2 NO-GO both arms |
| `a72b06e` | **Mission 1: phase-aware ban blend sweep** (`docs/2026-07-25-blend-phase-sweep-results.md`) | bans B1–3 w=0.0 (16.4 top-1), B4–5 w=0.75 (18.1 vs 8.8); composite 17.0 vs 13.3 production, **+3.7pts** (post-hoc-split caveat inside) |
| `e27a76e` | **Mission 2: pocket conditioning v13** (`docs/2026-07-28-pocket-conditioning-results.md`) | NO-GO — bans +5.30 but picks −4.15 breached the guard |
| `6e9c5a3` | **Slot-gated pocket v14** (`docs/2026-07-28-slot-gated-pocket-results.md`) | **GO** — bans +5.89 (7.70→13.59, 5/5 seeds), picks −0.26; gain concentrates in B1–3 as Stage B predicted |
| `dc2b39a` | **Capacity sweep v15** (`docs/2026-07-28-capacity-sweep-results.md`) | NO-GO — d128x6L wins loss, misses the +1.0pt top-1 gate; d192x4L6H incumbent stays |
| (viz lane commits) | attempt #6 etc. | separate lane, see its own handoffs |

**Uncommitted in the working tree** (other sessions' lanes, deliberately
left by the pocket-lane close-out): the 07-31 rung-2 density re-run
(`docs/2026-07-31-rung2-rerun-results.md` + modified `rung2_gbm.json`,
`soloq_lift_tables.json`, `experiment_v12_rung2_gbm.py` — NO-GO holds on
1.8× denser tables, cutoff correctly held), the counter-swing analysis
(`scripts/analysis_counter_swing.py` + doc + json), the LoL Meta Weekly
edition-1 skeleton in `drafts/`, `scripts/scrape_health.sh`, and ROADMAP
edits. A fresh session should either commit these with their owners'
framing or leave them alone — do not fold them into unrelated commits.

## Where the model actually stands

Production (`train_draft_model_v08.py`) is still the **v0.8.1 blend: old
(non-meta) transformer, picks w=0.75, bans w=0.0 (GBM-only)**. Nothing
val-proven since has been ported. The proven-but-unported stack, in
dependency order:

1. **Meta injection** (timesignal rung + blind test) — recommended for port
   since 07-24, unstarted.
2. **v14 `ban_only` pocket injection** — GO on val; solo transformer ban
   top-1 13.59 is now at parity with the GBM's 13.3 (codespace scale).
3. **Phase-aware ban blend weights** (Mission 1) — B4–5 w=0.75 swept with
   the *meta* ensemble's cached probs.

**Interaction warnings a fresh session must respect before porting:**

- **Meta and pocket injections have never been trained together.** v13/v14
  compared pocket-vs-plain-baseline; the meta rung compared meta-vs-plain.
  The natural next rung (v16) is the combination, paired against each solo
  condition, with the standard protocol (5 paired seeds 16/17/42/7/23,
  canaries hard-asserted, ban top-1 primary, ±1.5pt pick guard).
- **The Mission 1 blend weights are conditional on which transformer
  ships.** They were swept on `expcache_timesignal_blind` (meta ensemble).
  If v14 (or a v16 combo) is promoted, re-run `blend_phase_sweep.py`
  pointing at the new ensemble's val probs (`expcache_slotgate/` holds
  v14's per-seed prob npz) before fixing weights — the ban-side transformer
  is now a different (much stronger) model than the one the 0.75 was swept
  for.

## Machine-local caches (untracked — regenerate if absent)

| Cache | Contents | Regenerate with |
|---|---|---|
| `expcache_timesignal_blind/seed<S>.npz` | meta ensemble eval probs (val+test, 104g) | `blindtest_ban_timesignal.py` |
| `expcache_attr/{baseline,meta}_seed<S>.npz/.pt` | val probs both conditions + meta weights | `ban_attribution_train.py` (~30 min) |
| `expcache_pocket/` | v13 per-run json + prob npz + feat .pt | `experiment_v13_pocket_conditioning.py` (~30 min) |
| `expcache_slotgate/` | v14 per-run json + prob npz + weights | `experiment_v14_slot_gated_pocket.py` |
| `expcache_capacity/` | v15 sweep runs | `experiment_v15_capacity_sweep.py` (~2.5h) |
| `gbm_val_scores.parquet` | production GBM ensemble val scores, original gameid keys | `gbm_val_scores.py` (now runnable locally; note platform in provenance) |
| `ban_attribution_perban.parquet` | per-ban dominant-vector assignments | `ban_attribution_stageA.py` (minutes) |

## Open approvals ledger (Sahil has green-lit NONE of these)

1. The **production port bundle**: meta injection + v14 `ban_only` pocket
   injection + phase-aware blend weights into `train_draft_model_v08.py` —
   pending the v16 combination rung and the blend re-sweep above.
2. Committing the other sessions' uncommitted work (rung-2 rerun,
   counter-swing, newsletter skeleton).
3. Pushing anything. Local commits exist through `dc2b39a` + this doc.
4. Deleting or keeping the codespace (less needed now that lightgbm loads
   locally).

## Sharpest open questions, in leverage order

1. **v16: meta + pocket combined** — the port bundle is blocked on knowing
   the effects compose. Protocol and scaffolding are all in place
   (`experiment_v14_slot_gated_pocket.py` is the template).
2. **Blend re-sweep under the promoted transformer** — 30 minutes, local,
   after v16 picks a winner.
3. **B4–5 POCKET is still flat** (11.3 top-1, unmoved by v14) — the one
   Stage-B cell no rung has touched. Deferred idea: team-identity
   embeddings (roster churn made feature injection the right first shot).
4. **Rung 2 re-run when soloq spans a full split** — calendar-gated, not
   compute-gated; scrape must resume next window first
   (`docs/2026-07-31-rung2-rerun-results.md` follow-ups).
5. Pick-side conditioning (v13's own-roster features helped loss but hurt
   pick argmax — the "tension worth investigating" section of
   `docs/2026-07-28-pocket-conditioning-results.md` is the starting point
   for understanding why).
