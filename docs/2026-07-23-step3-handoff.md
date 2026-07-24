# Handoff: soloq embedding rung-0 — Step 3 (pro-transformer transfer test)

**Written 2026-07-23 20:09 ET by the scrape-day session (Opus 4.8). Audience:
a fresh Fable agent tasked with planning and executing Step 3 correctly and
at maximum leverage.** Steps 1–2 (data + purity gate) are DONE and the gate
PASSED. Your job is Step 3: does soloq-initialized `champ_emb` measurably help
the pro draft transformer on validation, over a 5-seed protocol? Read
`docs/2026-07-22-soloq-data-research.md` §6 first — it is the spec. This doc
makes it executable and flags the traps.

---

## 1. What is already done (do not redo)

### The scrape
- **78,171 high-elo soloq matches** collected 2026-07-23 (KR 26,448 / EUW
  25,120 / NA 26,603), Chall/GM/Master ladders, RANKED_SOLO, last ~45 days.
- Stored at `data/raw/soloq/soloq.db` (128 MB, **gitignored** — exceeds
  GitHub's 100 MB limit; never commit it). Schema/states in the scraper
  docstring. `integrity_check: ok`.
- Payload per match: `participants[{champ, team, pos, win}]` (champ =
  Data-Dragon `championName`, pos ∈ TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY) and
  `bans[{champId(numeric), turn, team}]`. Picks carry names; bans carry only
  numeric ids (Data Dragon `champion.json` maps id→name if you ever need bans;
  Step 3 does NOT need bans).
- Committed fix worth knowing: the scraper now uses `isolation_level=None`
  (autocommit) — an implicit write txn spanning rate-limited HTTP calls was
  deadlocking region threads ("database is locked"). Commit 6f2e17f.

### The rung-0 gate (Steps 1–2) — PASSED
- `scripts/soloq_embedding_rung0.py`: PPMI + truncated SVD (d=128) over the
  unordered 10-champion pick bags → champion embeddings; 5-NN role purity via
  the **exact** metric in `chart_embeddings_v08.py`
  (`NearestNeighbors(n_neighbors=6)`, mean over champs of neighbor role match).
- **Result: purity 0.723** vs pro-trained baseline 0.704 and the 0.55 GO gate
  (chance 0.20). Per-role: JNG 0.859, SUP 0.756, TOP 0.697, BOT 0.694, MID
  0.587. Soloq geometry encodes role at least as well as the pro embedding.
- Artifacts (committed): `data/processed/soloq_embedding_rung0.json` (metrics
  + provenance) and `data/processed/soloq_champ_embeddings.npz` with arrays
  `emb` (173×128, L2-normalized rows), `champs` (173 names), `primary` (173
  soloq primary roles). **This npz is your Step-3 input.**

---

## 2. Is the data sufficient? (analysis — answer: yes, don't scrape more first)

- **Coverage is complete and the tail is thick.** 173 distinct champions (the
  full live roster). NO champion has <500 games; 140 champs have ≥2,000, the
  rarest ~699. Co-occurrence marginals are well-conditioned for every champion
  — which is why purity landed above pro parity.
- **§6 wanted "~100k games is plenty to start"; 78k games = 781k pick
  instances is the same ballpark and already cleared the gate decisively.**
- **The Step-3 bottleneck is the PRO side, not soloq scale.** The comparison
  is measured on the pro **validation** split (~1,080 decisions per ROADMAP).
  More soloq games sharpen the *init* slightly but cannot enlarge the pro
  signal you're measuring against. More soloq data has low marginal value for
  THIS experiment.
- **If you later want more soloq data anyway, the real constraints are:**
  1. Dev key dies ~24 h after issue — sustained scraping needs a production
     key or daily refresh.
  2. Rate limit (20/s, 100/2 min **per key**) is a hard ceiling — ~2,600
     matches/hr/region, and more instances/agents just 429 each other (this
     is why the run is ONE process; do not "parallelize" it).
  3. Patch drift: more days = wider patch mix (current db is 16.13 40k /
     16.14 21k / 16.12 15k / 16.11 1.4k). For a clean single-meta embedding
     you'd filter to one patch, which fights volume.
  4. Re-scraping the same ladders yields diminishing NEW matches (dedup by
     match_id); genuinely more games means lower elo or more regions, which
     shifts the population and may hurt transfer.
- **Verdict: run Step 3 on what we have. Only consider a bigger, single-patch
  scrape with a production key IF Step 3 is promising and you want a v2 init.**

---

## 3. Step 3 — the experiment (from §6, made concrete)

**Claim under test:** initializing the pro transformer's `champ_emb` from the
soloq embedding beats random init on pro validation, over 5 seeds.

**GO** (either metric) — measured **validation only**:
- 5-seed mean **val-loss** improvement whose bootstrap CI excludes 0, **OR**
- **val top-1 up ≥ 1.5 points** (clears the documented ±1.5-pt seed-noise band).
- AND soloq role purity ≥ 0.55 — already satisfied (0.723).

**NO-GO / cheaply falsified:** soloq-init within noise of random-init on both
val loss and val top-1. (Purity <0.40 would also kill it — not our case.) A
null result is publishable: it means the pro data already learns everything
the embedding can carry.

### The working pro training loop to adapt
`scripts/embedding_evolution_v08.py` already trains ONE production seed on CPU
with the correct split and config. Clone its setup; strip the per-epoch
snapshotting; wrap it in a 5-seed × 2-condition loop. Key facts from it:
- **Config: `d_model=192, n_layers=4, n_heads=6`, lr 3e-4, dropout 0.1,
  patience 8** (`Config(d_model=192, n_layers=4, n_heads=6, seed=S)`).
- **Split (NEVER deviate):** test = EWC & July 2026 → dropped entirely
  (frozen, spent — **never evaluate against it**). cutoff = min test date;
  val = last 14 days before cutoff; train = before val_start.
- **CPU only** — MPS makes `masked_fill(-inf)` + cross-entropy produce inf
  losses. `dev = torch.device("cpu")`.
- `Vocab(list(cand.candidate.unique()), list(seq.champion.unique()))`;
  `model.champ_emb = nn.Embedding(vocab.size, 192, padding_idx=PAD)`;
  candidate rows are `vocab.candidate_ids`. Early-stop on val loss.
- Data: `draft_sequences_multi.parquet` (+ `draft_decisions_multi.parquet`)
  if present, else the 2026-only files.

### The three parts you must BUILD (these are the leverage points — get them
### right or the comparison is meaningless)

1. **Champion-name alignment: soloq `championName` ↔ pro Vocab names.** These
   namespaces differ (e.g. soloq `MonkeyKing`/`Belveth`/`KSante` vs pro
   `Wukong`/`Bel'Veth`/`K'Sante`). Build the bridge from Data Dragon
   `champion.json` (`data[*].id` = internal name used in soloq, `data[*].name`
   = display name close to pro CSVs); reconcile the handful that still differ
   by hand. **Log unmatched champions in BOTH directions** — every pro
   candidate champ that gets no soloq vector falls back to random init, and
   that dilution must be reported, not hidden. Expect near-full overlap.

2. **Project soloq d=128 → pro d=192, at the right scale.** Two independent
   issues:
   - *Dimensionality:* SVD rank ≤ 170 so you cannot SVD straight to 192.
     Options: a fixed random orthonormal 128→192 projection (Johnson–
     Lindenstrauss, preserves geometry — recommended default), or zero-pad
     64 dims. Pick one, justify it, keep it fixed across seeds/conditions.
   - *Scale (easy to get wrong, silently ruins the test):* soloq `emb` rows
     are L2-normalized (unit norm). `nn.Embedding` default init is ~N(0,1)
     per element (row norm ≈ √192). Injecting unit-norm rows next to a model
     that expects ~√192-scale rows changes effective LR/gradient flow and
     confounds the comparison. **Rescale the injected vectors to match the
     std/'norm distribution of a fresh random init** before writing them into
     `champ_emb.weight`. State exactly what you did.

3. **Add val top-1/3/5 accuracy** (the loop only tracks val loss). Compute
   masked argmax over legal candidates on the val tensors; report top-1 (the
   gate metric) and top-3/5 for color. Reuse the model's candidate masking so
   illegal champs are excluded, same as loss.

### Protocol
- Seeds: use the same 5 seeds for both conditions (e.g. 16,17,18,19,20) so the
  only difference is the init. For each (condition, seed): build model, (if
  soloq) overwrite candidate `champ_emb` rows with aligned+projected+rescaled
  soloq vectors, train to early-stop, record best val loss + val top-1/3/5 at
  the best-val checkpoint.
- Compare: paired by seed. Report mean±CI of (random − soloq) val loss via
  bootstrap over seeds (CI excludes 0 ⇒ GO on loss); report mean val-top-1
  delta vs the 1.5-pt band. Decide GO/NO-GO per §6. Write results to
  `data/processed/soloq_transfer_step3.json` and a short ROADMAP note offered
  as a commit.
- **Cost:** 10 CPU training runs (5 seeds × 2 conditions). Time-box and log
  per-seed wall time; the evolution script prints per-epoch timing you can use
  to estimate before launching the full sweep. Consider a 1-seed smoke run of
  each condition first to confirm the pipeline and get a time estimate.

### High-leverage guidance
- The single most important thing is a **fair** comparison: same seeds, same
  data, same schedule, init-scale matched — so any val delta is attributable
  to the soloq geometry alone. A sloppy scale or vocab mismatch will produce a
  confident wrong answer.
- If the headline is null, an **ablation** sharpens it cheaply: freeze
  `champ_emb` for the first N steps (probe whether the init helps early then
  gets overwritten), or evaluate a soloq-init model with `champ_emb` frozen
  entirely (does the geometry survive fine-tuning?). Only bother if the plain
  test is ambiguous.
- Keep the EWC test set frozen — the whole point of rung 0 is to decide
  cheaply on train/val before spending any test-set credibility.

---

## 4. Loose ends / repo etiquette
- **Uncommitted, NOT mine — leave alone:** `docs/ROADMAP.md` (modified) and
  `artifact/*` belong to a parallel viz session. Don't stage them. Sahil
  approved committing only my scrape+embedding work + strategy docs (done:
  commit 6f2e17f, pushed to main).
- The scraper (`riot_soloq_scrape.py`) supports `--snapshot` (WAL-safe backup)
  and `--export path.jsonl.gz` (done matches → jsonl) if you want a portable
  dump instead of hitting SQLite directly.
- Two idle Codespaces (`congenial-succotash`, `reimagined-engine`, Jul 19–20)
  linger on Sahil's account — not from this mission.
- Times ET; run `date` before writing any date. Commit trailers per repo
  convention; commit/push only when Sahil asks.
