# Handoff: post-attribution rungs — phase blend, pocket conditioning, syn-flavored 1b

**Written 2026-07-25 ~14:45 ET, after the denial-vector attribution study
landed (commit 2a56438). Read first:
`docs/2026-07-25-ban-attribution-results.md` (the evidence driving all
three missions), `docs/2026-07-25-ban-attribution-spec.md` (proxy
definitions), `docs/2026-07-24-blend-resweep-results.md` (blend mechanics +
cross-platform caveat), `docs/2026-07-23-ban-timesignal-results.md` +
`-blindtest.md` (meta injection lineage).**

## Non-negotiable rails (inherited, verified through five sessions)

- **The frozen EWC July-2026 test set has been looked at 3 times (v0.8,
  v0.8.1, timesignal blind test). It is never looked at again.** All
  selection below is val-only (54 games / 1080 decisions / 540 bans).
- Paired-seed RNG hygiene for any transformer change: new params created
  AFTER base module construction, zero-init, consuming no RNG, so the
  baseline condition reproduces production bit-for-bit. Canaries: baseline
  seed-16 best val loss **3.5855**, meta seed-16 **3.4623**. If a canary
  misses, stop and review — do not proceed on a drifted harness.
- Split invariants: cutoff 2026-07-15, train 5,043g/100,836d, val 54g/1,080d,
  vocab 171. CPU only (MPS breaks masked_fill(-inf)); `train_loss=inf` is
  cosmetic — do not "fix" masked_loss.
- lightgbm cannot import locally (libomp). GBM fits run on a codespace
  (`.devcontainer/` exists; `gh codespace cp` is broken — ship files with
  `tar cf - ... | gh codespace ssh -c $CS -- 'tar xf - -C ...'`; use
  `set -o pipefail`; python output is block-buffered over ssh, check the
  process with `pgrep`, and beware `pgrep -f` matching your own probe).
  Linux-refit GBM numbers differ from stored local blocks — compare only
  within one run.
- Never push without Sahil's explicit approval. Never stage
  `docs/ROADMAP.md` or `artifact/*` (parallel sessions own them). Parquets
  and npz caches stay untracked. Run `date` before writing any date.

## Machine-local caches (untracked — regenerate if you're not on the Mac)

| Cache | Contents | Regenerate with |
|---|---|---|
| `data/processed/expcache_timesignal_blind/seed<S>.npz` | meta ensemble eval probs (val+test, 104g) | `scripts/blindtest_ban_timesignal.py` (deterministic; DOES touch no new test info — probs only recomputed) |
| `data/processed/expcache_attr/{baseline,meta}_seed<S>.npz` + `meta_seed<S>.pt` | val-only probs both conditions + meta weights | `scripts/ban_attribution_train.py` (~30 min local CPU) |
| `data/processed/gbm_val_scores.parquet` | production 10-model GBM ensemble val scores, original gameid keys | `scripts/gbm_val_scores.py` on codespace, tar back |
| `data/processed/ban_attribution_perban.parquet` | per-ban dominant-vector assignments | `scripts/ban_attribution_stageA.py` (minutes, local) |

## Mission 1 — phase-aware ban blend sweep (do first; ~1 hour, LOCAL)

**Why:** Stage B showed the transformer's ban deficit is entirely B1–B3
(GBM 17.7 vs 6.2 top-1 on META, 13.9 vs 2.6 on POCKET) while in B4–B5 the
transformer wins or ties (META 24.4 vs 9.8; COMP 10.7 vs 0.0). Production
blends bans at w=0.0 (GBM-only), discarding the phase-2 advantage. Naive
cell arithmetic suggests ~+2.4 val ban top-1 points.

**Task:** extend the v0.8.1 per-type sweep (`blend_resweep_timesignal.py`
is the template) to per-type-×-phase for bans only: one weight for ban
decisions with `seq` in 1–6 (B1–3), one for `seq` in 13–16 (B4–5). Picks
keep their single weight (0.75 stands; re-verifying it in the same sweep is
fine and free). Grid w ∈ {0, .25, .5, .75, 1}, rank-average of pct-ranks
exactly as the template does, select on val top-1 per cell, ties → lower w.

**This now runs locally**: the GBM side is cached in
`gbm_val_scores.parquet` (that was the point of caching it) and the
transformer side in `expcache_timesignal_blind` (val slice, first 54 games
of the eval ordering — reconstruct ordering as the template does). No
codespace needed unless the parquet is missing.

**Honesty caveat to carry into the results doc:** the phase split was
chosen after seeing Stage B's cells on these same 540 bans. It has clean
a-priori motivation ("phase 2 has context to attend over") and only adds
1 df, but the sweep result is weaker evidence than a pre-registered split —
say so. Output: `data/processed/blend_phase_sweep.json` + a short results
doc. Any promotion into `train_draft_model_v08.py` needs Sahil's approval
and folds in the still-pending meta-injection port
(`docs/2026-07-24-blend-resweep-results.md` §Recommendation).

## Mission 2 — team/player conditioning rung (the structural fix)

**Why:** POCKET is 41% of all bans and the transformer's worst vector
(12.2 vs 5.4 overall; 13.9 vs 2.6 in B1–3). The transformer has zero
roster identity — pocket denial is structurally invisible to it. This is
the largest coherent deficit left.

**Design (v0 recommendation):** inject the candidate table's
opponent-referenced features — `opp_usage`, `player_pool`, `player_wr` —
the same causal, train-standardized way the meta rates went in, with one
critical difference: **input-side or deeper, not 4 output scalars.** Stage
B finding 3: the output-layer bias improved phase-1 top-5 (32→42) but not
top-1 — it re-ranks the shortlist but can't sharpen the argmax. Give the
model the features where attention can use them (e.g. project a per-slot
feature vector into d_model and add to slot embeddings, or a small MLP
bias head over the feature channels — zero-init the last layer for RNG
hygiene and exact-baseline start).

**Data subtlety the meta rung didn't have:** these features are per
(gameid, seq, candidate), NOT constant per game — ban rows reference the
opponent's roster, pick rows the own team's, and role-open probabilities
evolve through the draft (`draft_dataset.py` docstring, lines 10–39).
Build a (n_games, 20, vocab, F) tensor from the candidate table via the
(gameid, seq)→slot mapping (`SEQ_TO_SLOT`); positions with no candidate row
stay 0 (they're masked to -inf anyway); standardize on train games only.
Memory check: 5097×20×171×3 float32 ≈ 210MB — fine.

**Protocol:** `experiment_v11_ban_timesignal.py` is the scaffolding. 5
paired seeds (16/17/42/7/23), baseline canary 3.5855 hard-asserted,
primary metric ban val top-1, guard pick top-1 within ±1.5pts, GO = mean
d_ban ≥ +1.5pts or loss CI excluding 0. Additionally report the
Stage-B-style conditional table (join `ban_attribution_perban.parquet`):
the rung succeeds *for the right reason* if the gain concentrates in
POCKET and B1–3 cells. Deferred alternative: team-identity embeddings
(roster churn / cold-start makes feature injection the better first shot).

## Mission 3 — syn-flavored comp features (coordinate, don't duplicate)

Stage A: `pair_ctr` ("counters our comp") is nearly uninformative about
what gets banned (+0.5–2 lift) while `pair_syn` ("fits their comp")
carries signal (+4.6–6). Whatever comp/synergy features future rungs add
should be built opponent-comp-fit-first.

**A parallel session is actively working this lane** — commit `72df364`
(synergy rung 1c: 1-df duo-synergy scalar GO) landed while the attribution
study ran, and untracked rung-2/soloq work (`experiment_v12_rung2_*.py`,
`soloq_lift_tables.py`, `docs/2026-07-25-rung2-transfer-results.md`) is in
flight in the working tree. Before touching this mission: `git log` +
read the newest docs, and fold the syn-over-ctr finding into that lane
rather than starting a separate one. Rung 1b
(`experiment_v10_synergy_rung1b.py`, spec'd in 0a04506) remains gated on
the soloq scrape finishing.

## Open approvals ledger (things Sahil has NOT yet green-lit)

1. Porting the meta injection into production `train_draft_model_v08.py`
   (recommended since the blend re-sweep; unstarted).
2. Any blend-weight change from Mission 1.
3. Deleting or reusing the codespace
   (`effective-space-couscous-q57gg77r5rh4pp6`, stopped, has data parquets
   + venv already synced — cheapest path for any future GBM work).
