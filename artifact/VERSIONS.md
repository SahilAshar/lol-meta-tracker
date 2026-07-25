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
| #5 | `build_evo_panel5.py` | `artifact-v7.html` | `viz-attempt5` | Four beats: (1) whole vector as a colour strip at 3 moments, (2) the mechanism *demonstrated* from the draft record — substitutes are never teammates — plus sorted strips and the similarity curves, (3) the map with the false global claim replaced, (4) new: the exceptions, exposure quartiles + flex picks | in review |

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

## What changed in #5, and why (so a future session does not re-litigate it)

| Beat | Was (#4) | Is (#5) | Reason |
|---|---|---|---|
| 1 | Line chart of 4 dimensions labelled `slot 44/116/161/167`, then a caption saying no single number means anything | All 192 numbers as one colour strip at start / midway / final; the "you don't need to know what a slot means" line moved *before* the visual; raw digits moved into a `<details>` | The old beat asked the reader to study something and then told them it did not matter. Slot indices implied meaning that is not there. Pattern taken from Jay Alammar's Illustrated Word2vec |
| 2 | One asserted sentence: "pro teams pick Rakan in the same drafting situations as Alistar" | The team-comp evidence: Rakan+Alistar together in **0** of 10,294 line-ups (chance 93.9); Rakan+Orianna **64** vs 64.4 expected. Then 3 strips sharing one column order, then the curves | The load-bearing claim of the whole piece was unverifiable by a non-player. Now it is shown from the real draft record |
| 3 | "Five clusters form anyway" + a 9-line honesty caption | "Every champion ends up surrounded by others who play its role", pre-empts the blob count, caption cut to 2 sentences | The old claim is false — see `scripts/viz_claim_checks.py` C1. Caption length follows Distill's convention |
| 4 | did not exist | Exposure quartiles (0.443 / 0.624 / 0.914 / 0.833) and flex picks (0.453 vs 0.736), Corki as the counter-example to Rakan | The map's messiness needed explaining, and the explanation is measurable and more interesting than the claim it replaced |

Two things #5 also fixed that were invisible from the code:

- **A colour defect.** The beat-4 bars first used gold for Q1–Q3 and orange for the
  Q4 dip. That pair fails the dataviz validator hard: deutan ΔE **1.9**, and
  normal-vision ΔE **6.5**, below the 15 floor — full-colour readers cannot
  separate them either. It was also wrong in principle: purity-by-quartile is a
  single series, so a second hue encoded emphasis as identity. Now one hue, with
  the dip labelled. Dark mode uses `#AE8C34` rather than the page's `--gold`,
  because `--gold` sits at L 0.768 in dark — outside the 0.48–0.67 band for a
  large fill. Bars are drawn **opaque**: alpha-compositing washed the chroma below
  the floor (reads gray).
- **A build-time claim guard.** `check_claims()` in the builder asserts every
  structural claim against the data — the blob count, the co-occurrence contrast,
  the quartile ordering, the flex gap, and Corki-vs-Rakan. Nothing previously tied
  the prose to a measurement, which is how "five clusters" shipped.

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
