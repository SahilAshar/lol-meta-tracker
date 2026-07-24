# Execution handoff: ban time-signal rung — trailing meta-rates into the transformer

**Written 2026-07-23 ~22:05 ET by the Fable session that ran Step 3. Audience:
the agent that will EXECUTE this rung in a fresh session.** Data anchors below
were verified against the repo tonight. Read
`docs/2026-07-23-step3-execution-handoff.md` first — its corrections (init
scale, CPU-only, inf train loss, lightgbm import trap, seed set, repo
etiquette) all still apply, and `scripts/experiment_v09_soloq_transfer.py` is
your scaffolding: same data build, same split asserts, same inlined
train_model/topk_accuracy pattern, same incremental `expcache_*` caching.

**Mission:** the transformer's ban top-1 is 7.6 vs the GBM's 15.0 (blind
test), attributed by ROADMAP to date-blindness — it cannot see the current
meta. The GBM's cheapest meta features already exist and are causal: 28-day
trailing global pick/ban rates. Inject them at the transformer's output layer
and measure, over 5 paired seeds on **val only**, whether ban prediction
improves. This is the highest-expected-value accuracy rung (ROADMAP open
loop #2) and it feeds the v0.8.1 production blend the north-star column
grades weekly.

---

## 1. Verified anchors

- Features: `draft_decisions_multi.parquet` columns `pick_rate`, `ban_rate`
  (and `presence` = their sum) are **28-day trailing global rates**
  (`draft_dataset.py:64`, `RollingRates.global_rates`, windows end the day
  before the game — causal by construction; precedent for trusting causal
  trailing features across the split: `experiment_v09_outcome_baseline.py`).
- **Constant per (gameid, candidate) across slots** (verified on live data),
  and every game's candidate table covers all **168** candidates. So each
  game yields one (vocab_size, 2) meta matrix: read columns
  `[gameid, seq, candidate, pick_rate, ban_rate]`, take first per
  (gameid, candidate), scatter into vocab positions via `vocab.id_of`.
  Special tokens (PAD/START/MISSED) get zeros.
- Split/config/data identical to Step 3: cutoff 2026-07-15, train 5,043
  games / 100,836 decisions, val 54 / 1,080 (assert all four), vocab 171,
  `Config(d_model=192, n_layers=4, n_heads=6)`, seeds **[16, 17, 42, 7, 23]**,
  CPU only, ~3–4 min/run → ~35–45 min for 10 runs.
- Baseline anchor: random seed-16 best val loss **3.5855** (reproduced
  exactly by the Step 3 scaffolding — if your baseline condition doesn't hit
  it, your harness drifted).
- Ban-side scale: v0.8.1 val sweep had bans at top-1 **14.3 (pure GBM)**;
  the transformer's val ban top-1 will fall out of your baseline runs —
  report it, it's the number this rung tries to move.

## 2. Design (recommended minimal version — resist scope creep)

Add an optional per-game meta tensor to the batch (`to_tensors` extension or
a parallel dict entry, shape (n_games, vocab, 2), float32; standardize each
feature to zero mean / unit std **computed on train games only**). Model
change, gated so `meta=None` reproduces the current model bit-for-bit:

    # in forward(), after the weight-tied head:
    logits = h @ self.champ_emb.weight.T + self.out_bias
    if meta is not None:
        # slot-type-gated linear read of the meta features:
        # bans and picks get separate weights (2 slot types x 2 features)
        w = self.meta_w[SLOT_TYPE]            # (20, 2) after indexing
        logits = logits + (meta[:, None] * w[None, :, :, None]... )

(Exact einsum is your call — the contract is: logits[g, slot, champ] +=
sum_f w[slot_type(slot), f] * meta[g, champ, f]. That is **4 learnable
scalars**. Initialize to zero so training starts from the exact baseline
model.) Do NOT start with a patch embedding — it can't extrapolate past the
training patches and adds params; it's the ablation if rates alone
disappoint, not the first experiment.

**RNG hygiene, same trick as Step 3:** construct the model, then create the
meta parameters deterministically (zeros — no RNG consumed), so per seed the
baseline and meta conditions see identical init/dropout/shuffle streams and
the comparison is strictly paired. If you add parameters before
`torch.manual_seed`'s stream is consumed by the base modules, verify the
baseline still reproduces 3.5855 for seed 16 — that's your canary.

## 3. Protocol

1. One data build reused across all runs (Step 3 scaffolding + the meta
   matrix build; slim column reads keep memory trivial).
2. 5 seeds × {baseline, meta}: train to early-stop, record best val loss,
   val top-1/3/5 **split by picks/bans** via probs_for → attach_scores →
   inlined topk_accuracy (val_rows = 1,080 decisions, assert). Cache each
   run to `data/processed/expcache_timesignal/` incrementally.
3. Paired deltas per seed, primary metric **ban val top-1**; secondary: val
   loss, pick top-1 (watch for regression — the meta bias must not hurt the
   picks specialist).
4. **Decision:** GO if mean ban val top-1 delta ≥ **+1.5 points** (the
   documented seed-noise band) or the 10k-bootstrap 95% CI of paired val
   loss delta excludes 0 favorably, **with pick top-1 not degrading by more
   than the noise band**. NO-GO if within noise — which would be a real
   finding: output-layer rate injection is the cheapest possible time
   signal; its failure points to input-side conditioning next, not more
   output tricks.
5. Output `data/processed/timesignal_rung.json` (runs, deltas, CI, verdict,
   provenance) + results note offered as a commit. **Test set untouched** —
   promotion to a blind-test look only happens after a GO and only with
   Sahil's explicit approval (it would be the third look at EWC).

## 4. Traps & etiquette (inherited, plus new ones)

- lightgbm import trap, CPU-only, `train_loss=inf` cosmetic — all per the
  Step 3 handoff.
- **Standardization leakage:** compute meta feature mean/std on train games
  only; val games use train statistics.
- Val-window rates legitimately include train-era games in their trailing
  window (causal, not leakage — same as the GBM features).
- `docs/ROADMAP.md` carries BOTH a parallel viz session's north-star edit
  AND this session's open-loop updates, uncommitted — coordinate with Sahil
  before staging it; `artifact/*` stays untouched.
- Run `date` before writing dates; ET; commit/push only on approval.
