# Adversarial self-grade — evolution panel, attempt #4 (live)

Graded 2026-07-25 against the rubric in `artifact/HANDOFF-viz-v6.md`. Scored from
rendered pixels (9 headless configs: light/dark/390px × beats 1–3), not from
reading the code. Tag: `viz-attempt4`.

**Score each 1–5. Fix the weakest link first. Do not polish an 8/10 while a 3/10
sits next to it.**

| Dimension | Weight | Score | Why |
|---|---|---|---|
| Reader comprehension (non-player) | highest | **2** | The on-ramp spends the reader's attention and then refunds it. See below. |
| Honesty / no-overclaim | high | **3** | Every *number* is real and sourced. But beat 3's headline claim is not what the picture shows — an unflagged text↔image mismatch, which is worse than a missing caveat. |
| "Step by step" legibility | high | **3** | Tabs + Next work and are keyboard-reachable, but each beat is one static chart. Nothing shows *what changed* between beats, and there is no stepping *within* a beat. |
| Aesthetic bar vs. the static chart | medium | **3.5** | Type, palette, and direct line-labeling are good. Beat 1's chart is visually inert; the purity strip is cramped and unlabeled; the final frame has no title or cluster-level annotation. |
| Robustness | medium | **4.5** | Zero console errors across all 9 configs. Light + dark + 390px all lay out correctly. Reduced-motion steps discretely. CSP-clean, no external anything. ~82KB added. |

**Weakest link: reader comprehension, concentrated in beats 1–2.** This matches
Sahil's unprompted read exactly ("beat 1 and 2 are the biggest problems... that's
where most folks get lost").

## The three specific defects

### 1. Beat 1 asks for attention, then refunds it (comprehension)

The lead promises "the four of his numbers that moved the most." The reader then
studies four curves labeled `slot 44 / 116 / 161 / 167` — three of which nearly
overlap — drifting between 0.0 and +0.2. The very next caption says: *"No single
number means anything on its own."*

So the reader is asked to study something and is then told it was meaningless.
The slot numbers are noise labels: `slot 167` carries no information a lay reader
can use, and implying it might is a small dishonesty. The raw 12-slot table has
the same problem — it is real data that supports no inference.

### 2. Beat 2 asserts the load-bearing mechanism instead of showing it

The entire conceptual pivot of the piece is: *things that appear in similar
contexts end up with similar vectors.* Beat 2 delivers it as one asserted
sentence — "pro teams pick Rakan in the same drafting situations as Alistar." A
non-player cannot evaluate that, and nothing on screen demonstrates it. The
reader is asked to take the causal claim on faith and then read a cosine curve
that only makes sense if they already believed it.

The similarity definition then stacks three ML concepts into one sentence: sign
convention (−1/0/+1), direction-agreement across 192 dims, and mean-centering
("after removing the shared average direction").

### 3. Beat 3's headline claim is not what the map shows (honesty)

Beat 3 says: **"Five clusters form anyway."** Measured on the exact final t-SNE
layout the reader is looking at (`scripts` reproduction, same params: perplexity
18, `random_state=42`, `init="pca"`, cosine):

| Measure | Value | What it means |
|---|---|---|
| DBSCAN blobs per role (eps 0.22, min_samples 4) | top **3**, jng **2**, mid **2**, bot **2**, sup **1** → **10 total** | The map shows ~10 groups, not 5 |
| KMeans(k=5) on the 2D layout vs. true roles | purity **0.595**, ARI **+0.288** | The five visible blobs are *not* the five roles |
| 5-NN role purity in the 2D layout | **0.781** | But each dot's neighbors really do share its role |
| 5-NN role purity in raw 192-dim space (shipped metric) | **0.704** | The honest, measured, global claim |

The reader counts blobs, gets ~10, and quietly discounts the narration. **The
true claim is local, and the panel makes a global one.** Correct framing is
"every champion ends up surrounded by others who play its role" — which the
picture supports strongly (0.781) — not "five clusters form," which it does not.

This is very likely what Sahil meant by beat 3 being "a bit of a reach."

Note: the shipped `0.704` is *correctly* labeled. It uses Euclidean 5-NN on raw
embeddings over champions with a known primary role. A centered-cosine
re-derivation over all 168 gives 0.739. Different method, same conclusion — the
panel's number is sound and should not be changed.

## What is already good (do not regress it)

- Honest provenance habit: seed, config, training-game count, and the
  paced-not-mapped disclosure are all present and specific.
- Rakan as a single traced thread across all three beats, gold-ringed in the wide
  view. This is the right pattern and the research confirms it.
- Direct labels at line ends instead of a legend. Role palette validated on both
  surfaces. Reduced-motion path is real, not decorative.
- Robustness is genuinely strong and was verified, not assumed.

## Harness gotcha (cost ~20 minutes; do not repeat)

Element screenshots of a `<canvas>` come back **blank** from the Playwright
headless shell at `deviceScaleFactor: 2`. Beat 1's chart appeared to be a
rendering bug; it was not — `getImageData` confirmed 12,719 painted pixels.
**Keep `deviceScaleFactor: 1` in the screenshot harness.** Verify a suspected
blank canvas with `getImageData` before believing the pixels.
