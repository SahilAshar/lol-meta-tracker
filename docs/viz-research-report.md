# Visualizing embedding evolution for lol-meta-tracker — research + design proposals

## TL;DR

The animated-scatter approach (warm-started t-SNE + Procrustes per frame) is fighting the tool. t-SNE's own design guarantees that cluster *size* and inter-cluster *distance* carry no reliable meaning, and re-running it per frame — even with alignment tricks — reproduces exactly the "amorphous drift" problem the field has been documenting since 2016. The papers that specifically tackle *training-dynamics* visualization (dynamic t-SNE, Aligned-UMAP/NeuroMapper) exist to patch this, but even they still ship a point cloud, which is the wrong tool for 168 items with known ground-truth roles. With only 168 champions and 5 known roles, a **reordered similarity-matrix heatmap** (champs × champs, block-sorted by role, scrubbed over epochs) tells the "structure emerges from gradient nudges" story more directly, more cheaply, and without any of t-SNE's interpretive traps. It is also the closest established technique to your "2D slicing" instinct — just applied to *pairwise similarity* rather than *raw dimension values*, which turns out to matter a lot (see verdict below).

---

## What I found

### 1. Dynamic/temporal dimensionality reduction (the "improve the current approach" path)

- **Dynamic t-SNE (Rauber et al., 2016)** — adds a quadratic penalty between consecutive frames' embeddings to trade off per-frame fidelity against temporal smoothness. This is the principled version of what you're already doing by hand (warm-start + Procrustes). [paper](https://dl.acm.org/doi/10.5555/3058878.3058894) · [reference implementation](https://github.com/paulorauber/thesne)
- **Aligned-UMAP** — optimizes all frames' embeddings jointly with a cross-frame alignment regularizer, rather than aligning post-hoc. [UMAP docs](https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html)
- **NeuroMapper (Li et al., 2022)** — the closest published analog to what you're building: an in-browser, real-time visualizer of a model's internal embeddings across training epochs, built on Aligned-UMAP specifically to fix the "naive animated scatter drifts incoherently" problem. Notably, even this state-of-the-art tool still renders a *point cloud* — it just aligns it properly. [arXiv](https://arxiv.org/abs/2210.12492) · [project page](https://www.kevinyli.com/papers/neuromapper)
- **TimeCluster (Ali et al., 2019)** — general pattern for visualizing temporal high-dim data as a 2D trajectory/connected scatter; same family, general-purpose rather than training-specific. [paper](https://cs.swan.ac.uk/~csmark/publications/2019_TimeCluster.html)

**Takeaway:** doing this properly (real dynamic t-SNE or Aligned-UMAP, jointly optimized across all 56 frames) would likely fix a lot of your drift complaints — but it's a heavier Python precompute, still inherits t-SNE/UMAP's distance-and-size distortions, and produces the same "watch dots swim around" viewing experience you already found unsatisfying. It's an upgrade to the same idiom, not a different idiom.

### 2. Why t-SNE animation specifically underwhelms

Distill's canonical **"How to Use t-SNE Effectively"** (Wattenberg, Viégas, Johnson, 2016) documents exactly the failure modes you're hitting: cluster *size* in a t-SNE plot means nothing (the algorithm equalizes density), inter-cluster *distance* is often meaningless, and results can differ meaningfully run-to-run even at fixed hyperparameters. [distill.pub/2016/misread-tsne](https://distill.pub/2016/misread-tsne/) — every one of these gets *worse*, not better, when you additionally animate across 56 re-optimized layouts, because now the viewer is asked to track spatial relationships that were never guaranteed to be stable in the first place.

### 3. How respected ML-viz work handles training dynamics (non-scatter approaches)

- **TensorBoard Embedding Projector** — the default tool for this exact use case, and notably it does *not* animate; it lets you step between discrete checkpoints. [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) — the implicit design lesson: discrete, inspectable snapshots beat continuous motion for this task.
- **The Grand Tour (Li, Wattenberg, et al., Distill 2020)** — uses a classic *linear* projection (rotate high-dim data, project to 2D) instead of nonlinear t-SNE/UMAP specifically because linear projections preserve real distances ("data-visual correspondence"), so motion in the animation is actually meaningful. [distill.pub/2020/grand-tour](https://distill.pub/2020/grand-tour/) — a legitimate alternative philosophy, but it needs continuous rotation control to work, and its selling point (many possible honest 2D views) is more valuable for large point clouds without ground-truth labels than for your 168 role-labeled champions.
- **DeepTracker (Liu et al., 2018)** — CNN training-dynamics visual analytics system; explicitly chose **hierarchical small multiples** over animation to let experts compare training states, precisely because small multiples support side-by-side comparison better than motion. [arXiv](https://arxiv.org/pdf/1808.08531)
- **NeuroMapper**, again — even the most modern, best-funded version of "animate the embeddings" still had to solve alignment as a first-class research problem to make the motion legible. That's a strong signal this idiom is expensive to get right.
- Reordered **similarity/distance-matrix heatmaps** are the standard genomics/bioinformatics answer to "show emergent block structure in pairwise relationships" (clustergrams, seriated heatmaps) — cheap, robust, and immune to t-SNE's distortions because you're plotting actual similarity values, not a nonlinear projection of them.

### 4. Nomic Atlas / embedding explainers for lay audiences

Nomic Atlas (the modern "explore your embeddings" product) is still fundamentally a static/interactive point-cloud explorer for a single snapshot, not a training-evolution tool — confirms there isn't a well-known lay-audience example of *animated* embedding training that people consider a gold standard. The gold-standard lay explainers (Grand Tour, TensorBoard Projector) both lean on **discrete steppable views**, not continuous animation, which is the strongest single signal from this research.

Sources: [Nomic Atlas visualization guide](https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/how-to-visualize-embeddings) · [Nomic blog](https://www.nomic.ai/blog/posts/improve-ai-model-performance-with-embedding-visualization)

---

## Verdict on the "2D slicing" idea

Your instinct is right, but pointed at the wrong matrix. The closest established technique is a **clustered/reordered heatmap** ("clustergram," common in genomics: rows/columns sorted so that structurally-related items sit adjacent, revealing block patterns as color regions). Applied literally to your raw **champs × dims** embedding matrix (168 × 192), this technique will likely disappoint for the same reason the scatter did: individual embedding dimensions in a learned representation are not axis-aligned with human concepts and have no natural column order — a raw dimension heatmap will look like shifting static/noise even as real structure is forming underneath it, unless you first reorder the *columns* by hierarchical clustering (group correlated dimensions together) and reorder rows by role. Even then, it shows *within-champion* value patterns, not *between-champion* relationships, which is the part of the story ("Jungle champions become distinguishable from Support champions") that people actually want to see.

**Recommendation:** apply the "2D slicing" / clustered-heatmap idea to the **role × role cosine-similarity matrix** instead of raw dimension values. That's design #1 below, and it's the one I'd build first. Keep the raw-dimension heatmap as a secondary/supplementary panel (design #2) for a reader who wants to go one level deeper into "what are the numbers actually doing."

---

## Design proposals (ranked)

### 1. Role-block similarity-matrix evolution (recommended primary visual)

**What the reader sees:** A 168×168 grid, champions sorted into 5 contiguous role blocks (Top/Jungle/Mid/ADC/Support) with thin gridlines and role labels on the axes. Each cell's color encodes cosine similarity between that pair of champions' embeddings at the current training frame (diverging colormap, e.g. dark = dissimilar, bright = similar, centered near 0). A scrubber/play control steps through training; a small synced line chart underneath shows val_loss and 5-NN role-purity with a playhead marker tracking the scrubber. At epoch 0 the grid is visual noise (random init). By epoch ~10–15 the five diagonal blocks visibly brighten relative to the off-diagonal regions — the reader watches role clusters *crystallize as literal glowing squares*, which is a much more legible "structure emerging" signal than points drifting into loose clouds.

**Why it beats the animated scatter:** every cell is a real, stable number (a cosine similarity), not the output of a re-optimized nonlinear projection — so there's no run-to-run instability, no meaningless cluster-size inflation, and no "did that gap widen because of training or because t-SNE repacked the layout" ambiguity. It directly visualizes the exact quantity (role separation) that your 5-NN purity metric already measures, so the qualitative picture and the quantitative curve reinforce each other instead of needing separate interpretation.

**Implementation sketch:**
- *Python precompute:* pick keyframes (don't need all 56 — select frames concentrated where `val_loss` curvature is highest, e.g. dense early on, sparse late, ~16–20 frames total). For each keyframe: L2-normalize the 168×192 embedding snapshot, compute the 168×168 cosine-similarity matrix, reorder rows/cols by role block. Render each matrix as a small PNG (e.g. 168×168 or 2–3x upscaled for crispness) using a diverging colormap, base64-encode it. Store per-frame: `{epoch, val_loss, purity, image_b64}` plus a static `role_boundaries` array (index ranges per role, for the JS gridline/label overlay) and the champion order.
- *Why PNG, not raw JSON floats:* a 168×168 float matrix per frame is huge (~113KB raw per frame before base64), but the matrix has heavy spatial redundancy (adjacent champions in the same role are near-identical), so PNG's built-in compression crushes it — expect low single-digit KB per frame, comfortably inside the 150–300KB total budget even with ~20 frames. This is standards-compliant "no external libraries": `<img>`/canvas `drawImage` decode PNG data URIs natively in every browser.
- *JS:* scrubber drives which keyframe's `<img>` is drawn into a `<canvas>` with `imageSmoothingEnabled = false` for crisp cells; crossfade opacity between adjacent keyframes on scrub for perceived smoothness without needing true interpolation; role gridlines/labels drawn as a canvas overlay from `role_boundaries` (crisp at any zoom, theme-aware color); the loss/purity line chart is trivial (56 numbers each, drawn as simple polylines) with a vertical playhead synced to the scrubber.

**Honest limitations:** collapsing 192 dimensions to one cosine-similarity scalar per pair discards *which* dimensions are doing the work — this view answers "are roles separating" but not "how," which is exactly what design #2 is for. Keyframe sampling (rather than all 56 frames) means the animation is a fast crossfade between snapshots, not a true continuous morph — acceptable given the research above suggests discrete stepping is *preferred* for legibility anyway, not just a fallback. Role ground truth doing the sorting means this view can't discover a *new* clustering the model learns that doesn't match the 5 canonical roles (e.g., a "poke mid" sub-cluster) — a scatter plot would show that; this heatmap structurally can't unless you add a secondary "reorder by k-means on the final-epoch embedding" toggle.

### 2. Reordered raw-dimension heatmap (secondary/supplementary panel)

**What the reader sees:** champs (rows, role-sorted) × dimensions (columns, reordered by hierarchical clustering on the *final*-epoch correlation structure so correlated dimensions sit adjacent) — same scrubber-over-epochs interaction as design #1, toggled as an alternate view ("show what the numbers themselves are doing"). As training progresses, the reader should see faint vertical "bands" emerge — groups of dimensions that all light up similarly for one role block — which is the most literal answer to "how do the numbers actually get tuned."

**Why/limitations:** this is the literal version of your "2D slicing" idea, so it directly answers the question that's nagging at your reader — but it's higher-risk: without the column reordering it will look like noise the whole time (raw learned dimensions have no natural order or human-legible meaning), and even with reordering the bands may be subtle compared to the crisp block structure in design #1. Recommend building it *after* design #1 is working, as a "go deeper" toggle rather than the hero visual — pair it with a one-line caption explaining that columns were reordered by similarity, or a naive reader will assume the ordering is meaningful in itself (e.g., "dimension 40" means something), which it doesn't.

**Implementation sketch:** same PNG-baking trick as design #1 (168×192 per frame, diverging colormap, base64 PNG); expect worse compression than the similarity matrix (less spatial redundancy pre-reordering) so budget for ~12 keyframes instead of ~20, or reduce to top-variance dims (e.g., top 96 of 192 by cross-frame variance) to shrink both payload and visual clutter.

### 3. Static small-multiples "comic strip" (cheap fallback / MVP)

**What the reader sees:** 6–10 of the design-#1 similarity-matrix thumbnails laid out side by side (e.g., epoch 0, 2, 5, 10, 20, 30, 46), each stamped with its purity score — no scrubber at all, just Tufte-style small multiples the eye can compare directly. Click a thumbnail to load it full-size.

**Why:** this is the lowest-effort, most robust option — it reuses the exact same precomputed PNGs from design #1 with zero scrubbing/crossfade logic, and small multiples are a well-established, boringly reliable technique (this is literally what DeepTracker chose over animation, and what TensorBoard Projector's discrete-checkpoint model implies). Good as a fallback if the interactive scrubber in #1 turns out to be more engineering than it's worth, or as a "print view"/static export of #1.

**Limitations:** loses the play/scrub "watch it happen" engagement your current tool has, which was presumably part of the appeal; best paired with #1 rather than as a total replacement.

---

## Bottom line recommendation

Build **design #1** (role-block similarity-matrix heatmap, PNG-baked keyframes, scrubber + synced purity/loss chart) as the primary replacement for the animated t-SNE scatter. It directly fixes the specific complaint (amorphous drift, unclear "clusters forming" story) by plotting real, stable numbers instead of a re-optimized nonlinear projection, fits comfortably in the payload budget via PNG compression, and is a substantially smaller build than doing dynamic t-SNE/Aligned-UMAP properly. Add **design #2** as a "go deeper" toggle once #1 is working, and keep **design #3** in your back pocket as a zero-risk fallback since it's nearly free once #1's precompute exists.

---

# v6 addendum (2026-07-25) — fixing the ON-RAMP, not the animation

The v1 research above asked "which idiom shows embeddings evolving." That was the
wrong question. Attempt #4 shipped a working animation and still failed, because
readers get lost in beats 1–2 — before the animation. This addendum is about the
on-ramp, and it is grounded in what the best explainers actually do.

Reader feedback that triggered this (2026-07-25): *"beat 1 and 2 are the biggest
problems to me, I think that's where most folks get lost. 3 makes sense in an
abstract way, but it's a bit of a reach."*

## The single most transferable finding

**Jay Alammar, "The Illustrated Word2vec"** — https://jalammar.github.io/illustrated-word2vec/

He *pre-empts* the meaninglessness of individual dimensions instead of refunding
the reader's attention afterward. He shows a vector as a colored heatmap strip
and writes, verbatim:

> "I've hidden which traits we're plotting just so you get used to not knowing."

Our beat 1 does the exact inverse: it labels four curves `slot 44 / 116 / 161 /
167` — implying the slot index is informative — lets the reader study them, then
says "no single number means anything on its own." Identical honesty, opposite
reader experience. One sets up the next idea; ours cancels the previous one.

### His full on-ramp sequence, in order

1. **A familiar scoring analogy before any ML.** Big Five personality traits.
   Score 38/100 on introversion. Plot it. Add a second trait, call the pair "a
   vector from the origin to that point." The reader already trusts personality
   tests, so "a person as a list of numbers" costs nothing.
2. **Motivate similarity with a need, not a definition.** He asks: if you were
   hit by a bus, which of two candidates could replace you? The reader *wants* a
   similarity comparison before the word "cosine" appears.
3. **Encode the vector as a colored heatmap strip, and stack the strips.** ~50
   values per word, red/white/blue diverging. Readers spot that `king`, `queen`,
   and `girl` share column patterns *without knowing what any column means*. A
   line chart of four dimensions structurally cannot do this — it shows four
   numbers where the point is the pattern across all of them.
4. **Demonstrate co-occurrence with a sliding window over real text.** A real
   Dune sentence, the window sliding three times, each step showing the actual
   (features, label) pair it generates. He shows the training data itself, not a
   description of it.

## Caveat conventions

**Distill, "How to Use t-SNE Effectively"** — https://distill.pub/2016/misread-tsne/

- Caveats live in the **running prose at the point of use**, 1–2 sentences,
  declarative. Verbatim: *"The bottom line, however, is that you cannot see
  relative sizes of clusters in a t-SNE plot."* / *"...distances between
  well-separated clusters in a t-SNE plot may mean nothing."*
- **Every section heading IS its takeaway claim**: "Cluster sizes in a t-SNE plot
  mean nothing", "Distances between clusters might not mean anything", "Random
  noise doesn't always look random."
- Interactive sliders let the reader *discover* a pitfall rather than be told.
- Notably they use **no** visual "this element is unreliable" marking. The
  standard practice is short declarative prose at the point of use — not a
  hedging paragraph, and not a novel visual convention. Our beat-3 caption is a
  9-line wall; it should be 2 sentences where the reader needs them.

## Translation to concrete changes (proposed 2026-07-25)

1. **Beat 1: heatmap strip, not four labelled curves.** Show all 192 of Rakan's
   numbers as one strip, start vs final. Drop the slot indices — they imply
   meaning that isn't there. Add the pre-emptive Alammar line so "you don't need
   to know what the slots mean" arrives as a setup, not a retraction.
2. **Beat 2: stack the strips before defining anything.** Rakan / Alistar /
   Orianna as three strips. The reader sees Rakan and Alistar sharing column
   patterns and Orianna not, before "similarity" is defined or a curve appears.
3. **Beat 2: demonstrate the mechanism with real drafts.** Show actual draft
   sequences from `draft_sequences_multi.parquet` where Rakan and Alistar occupy
   the same slot. This is the sliding-window move: show what the model read. It
   replaces the one asserted sentence that currently carries the whole thesis.
4. **Beat 3: fix the claim.** "Five clusters form anyway" is false (see
   `scripts/viz_claim_checks.py`, and `docs/viz-selfgrade-2026-07-25.md`). Use the
   local claim the picture supports, and cut the caption wall to 2 sentences.
5. **Beat 4 (new): the exceptions are the payoff.** Purity by exposure quartile
   (0.481 / 0.714 / 0.952 / 0.810) plus flex picks (0.442 vs 0.777, p=7.9e-06).
   Two measured reasons a dot sits in the "wrong" place. Corki becomes a real
   second traced example against Rakan.

## Idiom decision

Keep **button-stepping**. Distill's own inline figures and Alammar's article are
both read-at-your-own-pace with local controls; scrollytelling would fight the
host article, which we do not control, and adds mobile/reduced-motion/scroll-
jacking risk for no comprehension gain here. Adopt Distill's structural move
instead: **make each beat's heading its own claim**, so a reader who only reads
the four headings still gets the argument.
