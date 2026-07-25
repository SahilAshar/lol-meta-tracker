# Handoff: iterate on the "watch the embeddings learn" visualization (v6)

You are picking up a session with Sahil about a data-viz panel embedded in a
League-of-Legends draft-model explainer artifact. **Attempt #4 is now live and
he has NOT yet reacted to it.** Your job this session: get his reaction, do the
competitive research that should have preceded the build, self-grade honestly,
and iterate — without rebuilding blindly.

## Cardinal rules (learned the hard way over 3 prior failures)

1. **Do not rebuild first.** Two earlier attempts were built-and-published
   before checking against his mental image; both flopped. Interrogate, mock
   (static matplotlib PNGs are fine), confirm, *then* touch panel code.
2. **Every number on the page must be real** — pulled from the saved snapshots,
   never invented. If something is illustrative (e.g. interpolated positions),
   say so in the caption. The whole artifact's credibility rests on this; every
   panel discloses its own limitations in plain language.
3. **Plain language, blog-reader default.** Assume the reader does not play
   League and does not know ML. Short sentences, active voice, no unexplained
   jargon. Run `/technical-writing-cleanup` on any prose you add.

## What's live right now (attempt #4)

Published IN PLACE at (republish to this exact URL, favicon 🌲, never mint a new one):
**https://claude.ai/code/artifact/36f462bc-ef98-42dd-af89-6c1a309c8877**
(scroll to "Step 9 · in motion" → panel "How role clusters form, in three beats")

It replaced the previous role-block similarity matrix. It is a 3-beat guided story:
1. **One champion** — Rakan as 192 numbers. Line chart of his 4 most-changed
   slots + a raw first-12-slots table (start / midway / final).
2. **Friends** — two centered-cosine similarity curves: Rakan↔Alistar climbs
   −0.04→+0.56, Rakan↔Orianna stays flat ~+0.09.
3. **All 168** — animated "fly-in" scatter map. Dots race from a seeded random
   cloud into 5 role clusters; trails; purity strip 0.22→0.70 under a
   play/scrub control; named champions fade in on the final frame; Rakan stays
   gold-ringed as the thread. Final frame = a TRUE t-SNE of the last snapshot.

**The honesty compromise in beat 3 (understand this before defending it):** the
final layout is a real t-SNE, and purity is measured in raw 192-dim space at
every snapshot — those are true. But the in-between motion is *paced, not
mapped*: each champ interpolates from random start → final spot, and its
progress along that path is its measured cosine-similarity to its own final
state at that snapshot. So *when* a champ settles is real; *where* it passes
through is interpolation. This is deliberate — per-frame t-SNE animates layout
noise (that was failed attempt #2). The caption says all of this. **Open
question for Sahil: is this compromise acceptable to him, or does he want the
principled version (aligned-UMAP / dynamic t-SNE with joint cross-frame
optimization — see docs/viz-research-report.md, NeuroMapper / Rauber et al.)?**

## His stated preferences (from the AskUserQuestion this session)

- Final frame: "same feel, new form" — does NOT have to be the exact polished
  t-SNE layout, just something equally polished showing the final structure.
- Gripes with the OLD matrix that drove this redesign: not visually beautiful;
  wrong ending; change hard to perceive; and the big one, verbatim:
  *"I don't see step by step how things change — if I'm publishing this as an
  interactive visual on a blog post, how do I expect readers to step through
  this to understand champ similarity, especially if they don't play league?"*
  → the "guided step-through for a non-League blog audience" is the north star.
- He picked "B + A combo" (guided story + fly-in map) and "replace the matrix
  entirely." Both delivered.

## FIRST actions this session (in order)

1. **Get his reaction to the live panel.** Ask concrete either/or questions, not
   "do you like it?". e.g.: Does the 3-beat step-through read for a non-player?
   Is beat 3's paced-not-mapped compromise OK or a dealbreaker? Is the final
   frame "polished enough"? Does the animation let you *feel* the accumulation?
2. **Deep competitive research (do this early, it should have come first).**
   Use the `deep-research` skill and/or WebFetch. Find how the best explorable
   / scrollytelling ML explainers handle "watch a model learn":
   - Canonical references: distill.pub (esp. t-SNE / UMAP / "How to Use t-SNE
     Effectively", momentum, embeddings articles); Google PAIR Embedding
     Projector & "Understanding UMAP"; TensorFlow Playground; Nicky Case /
     Bartosz Ciechanowski / Amit Patel (Red Blob) explorables; Observable
     notebooks on embedding evolution; nn-svg / BertViz style attention viz;
     3blue1brown framing for intuition.
   - Extract concrete, transferable patterns: how they pace a reader through
     steps, how they annotate honestly, scrollytelling vs. button-stepping,
     how they handle "the layout is not literally meaningful" disclaimers, use
     of a single traced example before the wide view, motion/easing choices,
     legend/labeling under color-vision constraints.
   - Write findings to `docs/viz-research-report.md` (append a v6 section) so
     it's durable, and translate them into 3–5 specific, actionable changes to
     propose to Sahil — not a book report.

## How to self-grade in this session (be adversarial with your own work)

Score the CURRENT live panel against these before proposing changes. Write the
rubric + scores into the session log so the next session can see movement.

- **Reader comprehension (non-player):** Could someone who's never played
  League read beats 1→2→3 and correctly explain "the model grouped champions by
  role without being told roles"? Where would they get lost? (This is his #1
  concern — weight it highest.)
- **Honesty / no-overclaim:** Is every number real? Is every illustrative
  element labeled as such? Does the beat-3 caption actually land, or is it a
  wall of hedging the reader skips?
- **"Step by step" legibility:** Can the reader control the pace and see the
  change between steps, or does it blur past? Is button-stepping the right idiom
  vs. scrollytelling for a blog embed?
- **Aesthetic bar:** Does it hit the polish of the static t-SNE chart he loves
  (`charts/champion_embeddings_tsne_v08.png`)? Composition, type, motion.
- **Robustness:** light + dark themes, mobile (390px), reduced-motion,
  keyboard/focus, CSP (no external anything), payload budget.
- **Grade each 1–5, name the single weakest link, and fix that first.** Do not
  polish what's already an 8/10 while a 3/10 sits next to it.

## What "good delivery" means here (the bar for shipping a change)

A change is not done until ALL of these hold — verify, don't assume:

- **Rendered and eyeballed in both themes + mobile + the final animation
  frame** via headless screenshots, with **zero console errors**. (Harness for
  this already exists — see below. Look at the pixels; the validator checks
  color, not layout.)
- **Palette re-validated** if any colors changed, via the dataviz skill's
  `scripts/validate_palette.js` on BOTH light (#FFFFFF) and dark (#1D2634)
  surfaces. Role palette currently passes on both.
- **Prose cleaned** (`/technical-writing-cleanup`) and **honesty captions
  intact** — no silent overclaim introduced.
- **Republished IN PLACE** to the URL above (favicon 🌲), and the WebFetch
  round-trip confirms your content is live (the publish flow requires reading
  the current version first — expect a 409 if another session touched it).
- **Load `artifact-design` + `dataviz` skills BEFORE editing** the page.
- Only THEN tell Sahil it's live, and say plainly what you verified and what you
  didn't.

## Assets & how to build (all durable paths)

- Repo: `~/Documents/repos/lol-meta-tracker` (venv `.venv/bin/python`).
- **Builder for the live panel:** `artifact/build_evo_panel4.py` — splices the
  fragment into pristine `artifact/artifact-v2.html` → writes
  `artifact/artifact-v6.html` (the published file) AND
  `artifact/evo_panel4_test.html` (standalone panel for screenshots; supports
  `#b0/#b1/#b2` to select a beat and `#b2end` to jump to the final frame).
  Run: `cd ~/Documents/repos/lol-meta-tracker && .venv/bin/python artifact/build_evo_panel4.py`
- Prior builders kept for reference: `build_evo_panel3.py` (matrix, now
  superseded), `build_evo_panel2.py` (old animated t-SNE).
- Snapshots: `data/processed/embedding_evolution_v08_snapshots.npz` — 56 frames
  × 168 champs × 192 dims, plus role, labels, val_loss, purity. Metadata:
  `data/processed/embedding_evolution_v08_demo.json` (n_train_games=5043,
  config d192x4L6H, seed 16). Regenerate snapshots with
  `scripts/embedding_evolution_v08.py` (CPU only — MPS gives inf losses, ~5min).
- Quality bar chart he loves: `charts/champion_embeddings_tsne_v08.png`
  (+ `_dark`). Role colors defined in `scripts/chart_embeddings_v08.py`
  (light: top #2a78d6 / jng #008300 / mid #e87ba4 / bot #eda100 / sup #4a3aa7;
  dark: #3987e5 / #008300 / #d55181 / #c98500 / #9085e9). Panel reuses these.
- Research survey with citations: `docs/viz-research-report.md`.

### Screenshot harness (already set up, reuse it)

No Chrome/Chromium app installed on this Mac. Use the Playwright-cached headless
shell binary directly:
`~/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell`
A Playwright-core driver script lives in the scratchpad (`shoot.js`) that loads
`evo_panel4_test.html`, drives each beat, forces light/dark + mobile widths +
mid/end animation frames, and reports console errors. `playwright-core` is
installed in the scratchpad `node_modules`. For a fresh session, either recreate
a similar script or `npm i playwright-core` in a scratch dir and point it at the
same binary. Dark theme: run against a copy of the test page with
`<html data-theme="dark">` (media-query dark alone won't stamp the attribute).

## Constraints (unchanged, non-negotiable)

- Artifact CSP: no external libs/fonts/network. Vanilla JS + canvas; precompute
  in Python; assets as data URIs. Current added payload ≈ 82KB (positions are a
  base64 uint16 array); stay well under ~600KB total added.
- Both themes via the page's existing CSS tokens (`--model`, `--meta`, `--gold`,
  `--ink`, `--muted`, `--card`, `--hairline`, etc.); respect
  prefers-reduced-motion (beat 3 already steps discretely under it).
- Verified exhibit numbers (centered cosine): Rakan↔Alistar −0.04 → +0.56;
  Rakan↔Orianna +0.04 → +0.09; 5-NN role purity 0.219 → 0.704 (note: the very
  first handoff said 0.195 for frame 0 — the true measured value is **0.219**;
  trust the data, not older notes).

## Directions still worth considering (if his reaction points there)

- **Scrollytelling instead of button-stepping** — beats advance on scroll, which
  is the dominant idiom in the explainers you'll research. Bigger lift; only if
  he wants it.
- **Principled dynamic projection** for beat 3 (aligned-UMAP / joint t-SNE) to
  make the in-between motion honest-by-construction rather than
  honest-by-caption. See research report.
- **Small-multiples "comic strip"** fallback of keyframes — zero-risk, static,
  print-safe; assets already exist.
- One traced champion (Rakan) is already the through-line; consider whether a
  second contrasting example (a flex pick that lands *between* clusters) makes
  the "no labels" point land harder.

## Suggested skills

- `deep-research` — for the competitive artifact research (do early).
- `artifact-design` + `dataviz` — load BEFORE editing the page.
- `/technical-writing-cleanup` — for any prose.
- `cos-session-log` — write a session log at the end with the self-grade rubric.
- `/handoff` (or write to this file) — leave the next handoff at session end.

## Not yet done (offer these)

- Repo changes (`build_evo_panel4.py`, `artifact-v6.html`, this handoff) are
  UNCOMMITTED. Offer to commit once Sahil is happy.
