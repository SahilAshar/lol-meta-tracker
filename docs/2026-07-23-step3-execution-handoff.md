# Execution handoff: Step 3 — soloq→pro embedding transfer test

**Written 2026-07-23 ~20:25 ET by a Fable review session. Audience: the Opus 4.8
agent that will EXECUTE Step 3.** This supersedes-and-extends
`docs/2026-07-23-step3-handoff.md` (still worth reading for background). Every
claim below was re-verified against the repo tonight, a full pipeline pilot was
run (data → tensors → injection → 2 training epochs → decision-level top-k),
and several corrections to the original handoff were found. Follow THIS doc
where they disagree.

**Mission:** does initializing the pro draft transformer's `champ_emb` from the
soloq embedding (`data/processed/soloq_champ_embeddings.npz`) beat random init
on pro **validation**, over 5 paired seeds? Spec:
`docs/2026-07-22-soloq-data-research.md` §6. Steps 1–2 are done (purity 0.723,
gate PASSED). You only run Step 3.

---

## 1. Corrections to the previous handoff (important)

1. **Init-scale target is std=0.02, NOT N(0,1).** The old handoff said fresh
   `nn.Embedding` rows have norm ≈ √192 ≈ 13.9. Wrong for this model:
   `DraftTransformer.__init__` re-initializes every embedding with
   `nn.init.normal_(std=0.02)` (`scripts/draft_transformer.py:158-160`),
   precisely because the output head is **weight-tied** to `champ_emb`
   (`h @ champ_emb.weight.T`) and N(0,1) rows blow initial logits to ±50.
   Correct target row norm: **0.02·√192 ≈ 0.2771** (measured fresh-init mean:
   0.2764). Following the old handoff literally would inject vectors ~50×
   too large and invalidate the whole comparison.
2. **No Data Dragon fetch needed — the full name mapping is below (§3),
   verified against the actual data: 168/168 pro vocab champions covered.**
   Zero random-init fallback; nothing to dilute or report as unmatched.
3. **Don't write a new masked-argmax top-1.** The repo already has the exact
   production decision-level metric: `probs_for` + `attach_scores`
   (`scripts/draft_transformer.py`) + `topk_accuracy`
   (`scripts/train_draft_model.py:54-79`). Use that path so your numbers are
   directly comparable to ROADMAP's. **Import trap:** `train_draft_model` and
   `experiment_v08` both import lightgbm at module top, and lightgbm is broken
   on this machine (libomp missing) — **copy the ~25-line `topk_accuracy` into
   your script** (precedent: `embedding_evolution_v08.py` inlines `VAL_DAYS=14`
   for the same reason).
4. **Seeds: use the production set `[16, 17, 42, 7, 23]`** (`SEEDS` in
   `train_draft_model.py:49`), not "16–20". Same 5 for both conditions.
5. **Expect `train_loss=inf` — it is cosmetic, do not "fix" it.** Exactly 1 of
   100,836 train decisions has its target champion outside the availability
   mask; `masked_fill(-inf)` makes that slot's loss an infinite *constant*
   (no gradient flows through masked entries), so the displayed epoch mean is
   inf while gradients and training are unaffected. Val (1,080 decisions) has
   zero such slots — val loss, early stopping, and all reported metrics are
   clean. If you want a readable train curve, report the finite-slot mean, but
   do not change `masked_loss` semantics.
6. **Cost is small: ~4 s/epoch on this machine (CPU).** The committed demo run
   went 47 epochs to early-stop → expect ~3–4 min/run, **~35–45 min for all
   10 runs**. No smoke run needed for timing — the pilot already measured it.

## 2. Verified anchors (recompute nothing, use these to sanity-check)

- Split (rule in `embedding_evolution_v08.py:64-68`, same as
  `experiment_v08.split_dates`): cutoff **2026-07-15**; train **5,043 games /
  100,836 decisions**; val **54 games / 1,080 decisions** — matches ROADMAP
  exactly. EWC July-2026 test games dropped entirely, never evaluated.
- Vocab: **171** tokens = 3 specials + **168 candidates**, and (with the
  multi-year data) **zero "extra" ban-only champs** — every champion token is a
  candidate row. So "inject candidate rows" ≡ "inject all champion rows."
  PAD/START/MISSED stay at their default init; PAD row stays zeroed.
- Random-init baseline, seed 16, this exact config/split: **best val loss
  3.5855** (`embedding_evolution_v08_demo.json`). Your random-init seed-16 run
  should land near this (snapshotting overhead removed, so not identical-run
  but same setup).
- soloq npz: `emb` (173×128, all rows unit-norm), `champs` (173), `primary`.
  Purity 0.7225 (`soloq_embedding_rung0.json`).
- **CPU only** (`dev = torch.device("cpu")`) — MPS produces inf losses with
  `masked_fill(-inf)` + cross-entropy. Confirmed convention in both training
  scripts.

## 3. Champion-name bridge (verified complete — paste as-is)

21 pro names differ from soloq internal names; everything else matches exactly.
5 soloq champs (Briar, Katarina, Locke, MasterYi, Teemo) never appear in pro
data — unused, fine.

```python
PRO_TO_SOLOQ = {
    "Aurelion Sol": "AurelionSol", "Bel'Veth": "Belveth", "Cho'Gath": "Chogath",
    "Dr. Mundo": "DrMundo", "Fiddlesticks": "FiddleSticks",
    "Jarvan IV": "JarvanIV", "K'Sante": "KSante", "Kai'Sa": "Kaisa",
    "Kha'Zix": "Khazix", "Kog'Maw": "KogMaw", "LeBlanc": "Leblanc",
    "Lee Sin": "LeeSin", "Miss Fortune": "MissFortune",
    "Nunu & Willump": "Nunu", "Rek'Sai": "RekSai", "Renata Glasc": "Renata",
    "Tahm Kench": "TahmKench", "Twisted Fate": "TwistedFate",
    "Vel'Koz": "Velkoz", "Wukong": "MonkeyKing", "Xin Zhao": "XinZhao",
}
# lookup: soloq_name = PRO_TO_SOLOQ.get(pro_name, pro_name)
```

Still assert coverage at runtime (`matched == len(vocab.id_of)`) so a future
data refresh can't silently break the bridge.

## 4. Injection recipe (piloted tonight, works)

```python
rng = np.random.RandomState(1234)              # fixed; NOT torch's global RNG
Q, _ = np.linalg.qr(rng.randn(128, 192).T)     # 192x128 orthonormal columns
P = Q.T                                        # 128->192, norm-preserving (JL)
target_norm = 0.02 * np.sqrt(192)              # ≈ 0.2771, matches fresh init
# per champion: v = soloq_emb[row] @ P; v *= target_norm / ||v||
# after model construction, under torch.no_grad():
#   model.champ_emb.weight[cid] = torch.from_numpy(v).float()
```

- Build `P` **once** with numpy, before any training, and reuse it for all
  seeds. **RNG hygiene:** injection must not consume torch's global RNG —
  `train_model` calls `torch.manual_seed(seed)` before constructing the model,
  and if injection leaves torch RNG untouched, both conditions see *identical*
  dropout/shuffle streams. That makes the comparison paired in the strongest
  sense: the ONLY difference is the 168 embedding rows.
- Pilot verification numbers: fresh-init champ-row norm mean 0.2764; post-
  injection 0.2771; 168/168 matched; first val loss after injection is sane
  (~4.2 at epoch 0, vs ~4.0 random — fine, that's pre-convergence noise).

## 5. Protocol

1. **One data build, reused across all 10 runs.** Load
   `draft_sequences_multi.parquet` + `draft_decisions_multi.parquet` (for the
   candidate table read ONLY columns
   `[gameid, date, league, seq, is_ban, candidate, label]` — the full table is
   15.8M rows and this keeps memory trivial). Split per §2. Build vocab from
   full `cand.candidate.unique()` + `seq.champion.unique()` (post-test-drop
   seq), `build_games`, `to_tensors` once. Data load + build_games ≈ 3 s total.
2. **Training loop:** reuse `draft_transformer.train_model` — either add an
   optional `champ_init: dict[int, np.ndarray] | None = None` parameter
   (applied under `no_grad` right after model construction; default None keeps
   every existing caller identical), or copy the ~45-line function into your
   script. The parameter is cleaner; your call. Config:
   `Config(d_model=192, n_layers=4, n_heads=6, seed=S)` — lr/dropout/patience
   defaults are already production.
3. **For each seed in [16, 17, 42, 7, 23] × condition in {random, soloq}:**
   train to early-stop (train_model already restores the best-val state),
   record best val loss, then compute val top-1/3/5 via
   `probs_for(model, t_val)` → `attach_scores(val_rows, probs, val_pos, vocab)`
   → inlined `topk_accuracy`, where `val_rows` is the candidate-table slice for
   the val window (1,080 decisions — assert that count). Log epochs + wall time
   per run. **Cache each run's results to disk incrementally** (precedent:
   `expcache_v08/`) so a restart never refits.
4. **Decision (from §6, val only):**
   - Paired deltas per seed: `d_loss_i = loss_random_i − loss_soloq_i`,
     `d_top1_i = top1_soloq_i − top1_random_i`.
   - **GO** if the 10k-resample bootstrap 95% CI of mean `d_loss` excludes 0
     (with n=5 that effectively requires all 5 deltas same-sign — report the
     sign count too), **OR** mean `d_top1` ≥ **+1.5 points**. (Purity ≥0.55
     already satisfied at 0.723.)
   - **NO-GO** if within noise on both. A null result is a real, publishable
     answer — report it straight, don't torture the stats.
5. **Output:** `data/processed/soloq_transfer_step3.json` — per-run
   {condition, seed, best_val_loss, top1/3/5, epochs, wall_s}, the paired
   deltas, bootstrap CI, verdict, and provenance (projection seed 1234,
   target_norm, mapping size, split cutoff). Then a short results note
   (new doc or ROADMAP entry) **offered** as a commit — do not commit/push
   until Sahil approves.
6. **Only if the headline is ambiguous** (CI straddles 0 but every delta is
   positive, say): the cheap ablation is soloq-init with `champ_emb` frozen
   (`requires_grad_(False)`) — does the geometry help when it can't be
   overwritten? Don't run ablations preemptively.

## 6. Repo etiquette (unchanged, still true tonight)

- `docs/ROADMAP.md` (modified) and `artifact/*` (untracked) belong to a
  parallel viz session — **never stage them**.
- `data/raw/soloq/soloq.db` is gitignored (122 MB) — never commit it. You
  don't need it for Step 3; the npz is your only soloq input.
- CPU only; run `date` before writing dates anywhere; times in ET; commit
  trailers per repo convention; commit/push only when Sahil asks.
- The EWC July-2026 test set stays frozen. Nothing in Step 3 touches it.
