# Explainer artifact — version ladder

Every attempt at the embedding-evolution panel is kept as **its own builder
script** plus **its own output HTML** plus **a git tag**. Nothing is edited in
place and nothing is deleted, so any version can be rebuilt, re-published, or
diffed against any other.

**Live artifact URL (always republish IN PLACE, favicon 🌲, never mint a new one):**
https://claude.ai/code/artifact/36f462bc-ef98-42dd-af89-6c1a309c8877

## The ladder

| Attempt | Builder | Output | Tag | Panel idiom | Verdict |
|---|---|---|---|---|---|
| base | — | `artifact-v2.html` | — | Explainer with **no** evolution panel. Every builder splices into this pristine file. Never edit it. | n/a — this is the substrate |
| #1 | *(not retained)* | — | — | Flat PCA projection of embeddings over training | Failed — projection was meaningless |
| #2 | `build_evo_panel2.py` | `artifact-v4.html` *(file not retained)* | — | Per-frame t-SNE, warm-started + Procrustes-aligned, plus a raw-numbers exhibit | Failed — per-frame t-SNE animates layout noise, not learning ("amorphous drift") |
| #3 | `build_evo_panel3.py` | `artifact-v5.html` | — | Role-block-sorted 168×168 cosine-similarity heatmap scrubbed over ~18 PNG keyframes, plus a champs×dims raw-value heatmap toggle | Rejected — not visually beautiful, wrong ending, change hard to perceive, and **no step-by-step path for a non-League reader** |
| #4 | `build_evo_panel4.py` | `artifact-v6.html` | `viz-attempt4` | Three-beat guided story: (1) one champion as 192 numbers, (2) two similarity curves, (3) animated fly-in map ending on a true t-SNE | **Partial.** Beat 3's animation reads decently; final frame needs polish. **Beats 1–2 lose non-player readers** — the on-ramp is the weak link |

## Feedback log (what the reader actually said)

- **On #3 (verbatim):** *"I don't see step by step how things change — if I'm
  publishing this as an interactive visual on a blog post, how do I expect
  readers to step through this to understand champ similarity, especially if
  they don't play league?"* → the north star is a **guided step-through for a
  non-League, non-ML blog audience**.
- **On #4 (2026-07-25):** beats 1 and 2 are the biggest problems and where most
  readers get lost. Beat 3 makes sense but "is a bit of a reach"; the animation
  looks decent and can be improved. Final frame: close, needs polish. On the
  paced-not-mapped honesty compromise: wants **a mix of keeping the motion and
  the annotated comic-strip idiom** — real keyframes, more annotation.

## How to move between versions

Rebuild any attempt from source (each builder reads the pristine base and writes
its own output — builders never overwrite each other):

```sh
cd ~/Documents/repos/lol-meta-tracker
.venv/bin/python artifact/build_evo_panel4.py    # -> artifact-v6.html + evo_panel4_test.html
```

Recover an attempt's exact state as shipped:

```sh
git show viz-attempt4:artifact/artifact-v6.html > /tmp/attempt4.html
git checkout viz-attempt4 -- artifact/build_evo_panel4.py   # restore just the builder
```

Compare two attempts' builders:

```sh
git diff viz-attempt4 -- artifact/build_evo_panel4.py
```

**To publish a different version:** point the Artifact tool at that attempt's
`artifact-vN.html` and publish to the SAME URL above. Reverting is therefore a
one-call operation — no rebuild required, because every attempt's full output
HTML is committed.

## Rules for the next attempt

1. **New attempt = new builder** (`build_evo_panel5.py`) + **new output**
   (`artifact-v7.html`) + **new tag** (`viz-attempt5`). Never mutate a shipped
   builder — that is what makes rollback trivial.
2. Add a row to the table above and a line to the feedback log **before**
   publishing, not after.
3. Tag immediately after the commit that produces a publishable version.
4. Every number on the page must come from the saved snapshots. Anything
   illustrative gets said plainly, in the figure where possible.

## Verified exhibit numbers (do not re-derive from memory)

Centered cosine, from `data/processed/embedding_evolution_v08_snapshots.npz`:

- Rakan ↔ Alistar: −0.04 → +0.56
- Rakan ↔ Orianna: +0.04 → +0.09
- 5-NN role purity: **0.219** → **0.704** (chance = 0.20). An early handoff said
  0.195 for frame 0; that was wrong. Trust the data.
- Snapshots: 56 frames × 168 champions × 192 dims. Demo run: seed 16, config
  d192x4L6H, 5043 training games.
