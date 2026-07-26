# Handoff: publish attempt #5, then finish the cleanup (v7)

You are picking up a session with Sahil about a data-viz panel embedded in a
League-of-Legends draft-model explainer artifact. **Attempt #5 is built, verified,
and committed — but NOT published. Sahil has never seen it rendered.** He asked
for exactly two things this session:

1. **Let him view the work** — which means publishing #5 to the existing artifact.
2. **Clean up what needs cleaning, and be satisfied with the work.**

Do them in that order. He is authorising the publish; do not re-litigate it.

## Cardinal rules (learned over five attempts)

1. **Do not rebuild before confirming.** Interrogate → mock → confirm → build.
   Three attempts died from building first. #5 is already built, so this now means:
   do not *restructure* anything before he reacts to what is live.
2. **Every number on the page must be real**, pulled from the saved snapshots or
   the draft record. Anything illustrative says so, in the figure where possible.
3. **Plain language, blog-reader default.** Assume the reader does not play League
   and does not know ML. Short sentences, active voice.
4. **A visual the reader cannot verify is worse than no visual.** This is the
   hardest-won lesson of #5 — see the naive-reader quote in `VERSIONS.md`.

## STEP 1 — Publish #5 in place, then hand him the link

Publish `artifact/artifact-v7.html` to the SAME URL, favicon 🌲. **Never mint a
new URL.**

**https://claude.ai/code/artifact/36f462bc-ef98-42dd-af89-6c1a309c8877**

- Load the `artifact-design` skill before touching the page.
- The publish flow requires reading the current version first; expect a 409 if
  another session touched it. On 409, re-read, merge, publish again — do not force.
- Confirm with a WebFetch round-trip that your content is actually live.
- Then tell him plainly: it is live, scroll to **"Step 9 · in motion"** → the panel
  **"How role clusters form, in four beats"**, and say what changed since #4 (the
  four bullets under "What he has not seen" below).
- Rollback is one call: `viz-attempt4` is tagged and `artifact-v6.html` is
  committed in full. Publish that file to the same URL to revert.

## STEP 2 — Get his reaction, with concrete either/or questions

He answered #4's questions well when they were specific. Ask about:

- **Beat 2's team-comp framing.** This was *his* idea ("i am building a team comp
  for a pro team?") and it is now the load-bearing beat. Does it land?
- **Beat 1's on-ramp.** It now opens with the model's one job before showing any
  numbers. Is the entry point right, or still too abstract?
- **Beat 4.** Does making the exceptions the payoff work, or does it feel like an
  appendix?
- **The idiom question he has not answered** (see "Open judgment call" below).

## What he has not seen (the #4 → #5 diff, in his terms)

1. **Beat 1 now says what the model does first.** Was: "Rakan is a list of 192
   numbers" with four curves labelled `slot 44/116/161/167`, then a caption
   admitting no single number means anything — it spent the reader's attention and
   refunded it. Now: the model's one job, then all 192 numbers as signed bars at
   three moments, with "you don't need to know what a slot means" moved *before*
   the visual as a setup. Raw digits live in a `<details>`.
2. **Beat 2 demonstrates instead of asserting.** The whole thesis used to be one
   sentence a non-player could not check. Now it is the draft record: a team fields
   one champion per role, so two supports compete for the same seat — **Rakan and
   Alistar appear together in 0 of 10,294 line-ups** (chance would give 93.9),
   while Rakan and Orianna, who do different jobs, appear together 64 times against
   64.4 expected.
3. **Beat 3's claim is scoped to what is measured.** "Five clusters form anyway"
   was false (the layout holds 8–13 blobs). It now reads "Most champions end up next
   to others who play their role — 0.704 of the time, against 0.20 by chance", says
   plainly that parts of the map are a jumble, and hands that to beat 4. The
   nine-line honesty caption is two sentences, placed *before* the play button.
4. **Beat 4 is new: the exceptions are the payoff.** A dot lands in the wrong place
   for exactly two measured reasons — the model hardly saw that champion (purity by
   exposure quartile 0.443 / 0.624 / 0.914 / 0.833) or it genuinely plays two roles
   (flex 0.453 vs non-flex 0.736). Corki is the counter-example to Rakan: comparable
   exposure, purity 0.20.

## STEP 3 — The cleanup queue, in priority order

Everything here is a known, named defect. Fix the weakest link first; do not
polish an 8/10 next to a 3/10.

### Must-fix before he shares this anywhere

1. **Run `/technical-writing-cleanup` on the panel prose.** Never done — the prose
   has been rewritten twice from reviewer feedback and deserves one clean pass.
   **Keep every honesty caption intact and do not weaken a hedge into a claim.**
2. **Rebuild beat 3's purity sparkline.** Currently ~45px tall with no y-axis, its
   label (`role purity ↑ (never optimized)`) sitting *inside* the plot area over
   the line's own baseline, a bare terminal tick that reads as a clipping artifact,
   and `chance 0.20` below legible contrast. Target: 90–110px, label above the
   plot, two y-ticks (0.20 and 0.70), no terminal tick, 32px clearance below.
3. **Get dark + mobile independently reviewed.** The robustness subagent never
   returned a report. Dark and 390px are verified only by my own eyes plus
   zero-console-error runs across 12 configs. This is the one unverified surface.

### Should-fix

4. **Beat 2 is doing two beats' work.** The panel jumps 611px → 1350px on one
   click of Next, and beat 2 carries three separate figures with three label
   treatments. Either split it or set a min-height across beats.
5. **Unify the stat-card template in beat 4.** The four cards use three different
   internal structures; nested boxes are redundant chrome. One template, no inner
   boxes, equal border weights.
6. **One spacing scale.** Figure→prose gaps currently vary 17/46/33/27px and
   intro→figure 22–40px. Pick 32px and hold it.
7. **Tab row.** Wraps 3+1 leaving a large gutter with `Next →` stranded in it.
   Make it a 4-up grid (or 2×2 on mobile) with Next on its own right-aligned line.
8. **Monospace overuse.** It does not earn its place on tab labels or on prose
   captions like `champions played in ONE role (149)`. Reserve it for numerals and
   axis values.
9. **Measure is ~80 characters.** Tighten to ~68ch.

### Worth considering (from the research, not yet acted on)

- **Text-to-visual linking** — the highest-evidence mechanic in the pacing
  research (Zhi et al., CGF 2019, n=180: measured engagement win *and* recall
  improvement). Hovering a champion's name in the prose highlights its dot on the
  map.
- **Encode arrival in the marks** on beat 3: in-transit dots pale/desaturated,
  settled dots crisp and saturated. Converts the paced-not-mapped disclosure from
  prose into something the reader sees. Also: tick the real snapshots on the
  scrubber so the interpolated spans are visible.
- **A "show raw per-frame t-SNE" toggle** — turns the longest disclaimer into a
  button and lets the reader discover *why* the motion is paced. This is Distill's
  own move: demonstrate rather than hedge.

## Open judgment call — his to make, do not decide it alone

The pacing research argues for **restructuring the four gated beats into annotated
static small multiples** (a "comic strip"), with the animation demoted to an
optional play/scrub on the final panel. The evidence is unusually aligned: panel
layouts beat vertical scroll on comprehension (Zhi et al., n=180); scroll buys no
engagement advantage over stepping (McKenna et al., via Distill); discrete content
wants discrete interaction (Kosara); and a button gate "makes a promise it hasn't
earned" whereas small multiples pay off *before* any click. Full scrollytelling is
ruled out — the panel sits inside a larger post and cannot own the page scroll.

Note this also matches what he said about #4's honesty compromise: he wanted "a mix
of keeping the motion and the annotated comic-strip idiom." **That request is only
half-delivered.** The motion is kept and the claims are annotated, but the
keyframe/comic-strip structure was never built. Raise this explicitly.

It is a redesign, not a tweak. Do not start it without his word.

## The bar for shipping any change

A change is not done until ALL of these hold — verify, do not assume:

- **Rendered and eyeballed** in both themes, at 390px, and on the final animation
  frame, with **zero console errors**. Look at the pixels; the validator checks
  colour, not layout.
- **Palette re-validated** if any colour changed, on both light `#FFFFFF` and dark
  `#1D2634`, via the `dataviz` skill's `scripts/validate_palette.js`. Load the
  `dataviz` skill before editing chart code.
- **`check_claims()` passes** — it runs automatically in the builder and will fail
  the build if a structural claim drifts.
- **Prose cleaned**, honesty captions intact, no new overclaim.
- **Republished in place** with a WebFetch round-trip confirming it is live.
- Then say plainly what you verified and what you did not.

## Assets and how to build

- Repo: `~/Documents/repos/lol-meta-tracker` (venv `.venv/bin/python`).
- **Builder:** `.venv/bin/python artifact/build_evo_panel5.py` — splices into
  pristine `artifact/artifact-v2.html` and writes `artifact/artifact-v7.html`
  (the publishable file), plus `evo_panel5_test.html` and
  `evo_panel5_test_dark.html` for screenshots. The test pages accept
  `#b0`…`#b3` to select a beat and `#b2end` to jump to the final frame.
  **The builder emits its own dark test page** — no hand-editing needed any more.
- **Claim checks:** `.venv/bin/python scripts/viz_claim_checks.py` verifies every
  structural claim and regression-checks the shipped exhibit numbers.
- Snapshots: `data/processed/embedding_evolution_v08_snapshots.npz` (56 frames ×
  168 champs × 192 dims, plus `role`, `flex`, `labels`, `val_loss`, `purity`).
- Draft record: `data/processed/draft_sequences_multi.parquet` (5,147 games,
  2024-01-13 → 2026-07-19). This is the source for exposure and co-occurrence.
  **Do not use `pick_rates.csv`** — it covers only some leagues and left 41 of 168
  champions unmatched, which silently corrupts any exposure analysis.
- Quality bar he likes: `charts/champion_embeddings_tsne_v08.png` (+ `_dark`).
- Role colours (light): top `#2a78d6` / jng `#008300` / mid `#e87ba4` /
  bot `#eda100` / sup `#4a3aa7`. Dark: `#3987e5` / `#008300` / `#d55181` /
  `#c98500` / `#9085e9`. Defined in `scripts/chart_embeddings_v08.py`.
- Research + rationale: `docs/viz-research-report.md` (v6 addendum has the
  on-ramp findings), `docs/viz-selfgrade-2026-07-25.md` (rubric + scores),
  `artifact/VERSIONS.md` (version ladder, feedback log, per-beat rationale).

### Screenshot harness

No Chrome app on this Mac. Use the Playwright headless shell directly:

```
~/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell
```

`npm i playwright-core` in a scratch dir, then drive it with `chromium.launch({
executablePath: EXE })`. Two gotchas that cost real time:

- **Keep `deviceScaleFactor: 1`.** At 2, `<canvas>` element screenshots come back
  **blank** and you will "find" rendering bugs that do not exist. Verify any
  suspected blank canvas with `getImageData` before believing it.
- Use `evo_panel5_test_dark.html` for dark. A media-query dark preference alone
  does not stamp `data-theme`, and the colour tokens key off the attribute.

## Constraints (non-negotiable)

- Artifact CSP: no external libs, fonts, or network. Vanilla JS + canvas,
  precompute in Python, assets as data URIs. Added payload ≈58KB; stay well under
  ~600KB.
- Both themes via the page's existing CSS tokens (`--model`, `--meta`, `--gold`,
  `--ink`, `--muted`, `--card`, `--hairline`, `--inset`, `--faint`). Respect
  `prefers-reduced-motion` — beat 3 steps discretely under it.
- **Blue and orange are reserved for nothing categorical.** They were reused as a
  diverging value scale while also meaning Top lane and Bot lane on the map. Sign
  is now encoded by direction in neutral ink. Do not reintroduce a hue that
  already carries a role meaning.
- Beat-4 bars are drawn **opaque**: alpha-compositing pushed chroma below the
  validator floor (`#827140` reads gray). Dark mode uses `#AE8C34`, not the page's
  `--gold`, which sits at L 0.768 in dark — outside the band for a large fill.

## Versioning rules (he asked for this explicitly)

**New attempt = new builder + new output + new tag.** Never mutate a shipped
builder; that is what makes rollback trivial. Add the `VERSIONS.md` row and the
feedback-log line *before* publishing, not after. Tag right after the commit that
produces a publishable version. Because every attempt's full output HTML is
committed, reverting never needs a rebuild.

Current tags: `viz-attempt4` (live now), `viz-attempt5` (to be published).

## Verified numbers — do not re-derive from memory

From the snapshots and the draft record; all regression-checked by
`scripts/viz_claim_checks.py`:

- Rakan ↔ Alistar centred cosine: **−0.04 → +0.56**. Rakan ↔ Orianna: **+0.04 → +0.09**.
- 5-NN role purity: **0.219 → 0.704** (chance 0.20). The shipped metric is
  Euclidean 5-NN over all 168 champions; it reproduces exactly.
- Layout blobs: **8–13** across every DBSCAN setting, never 5. KMeans(k=5) on the
  2D layout recovers roles at only **0.595** purity / **+0.288** ARI.
- Team comps: **10,294** (one per side per game). Rakan+Alistar together **0**
  (expected 93.9); Rakan+Orianna **64** (expected 64.4). Across the 40 most-picked
  champions, same-role pairs are teammates at **0.074×** chance, different-role at
  **0.936×**.
- Purity by exposure quartile: **0.443 / 0.624 / 0.914 / 0.833**. Flex **0.453**
  vs non-flex **0.736** (p = 7.9e-06). Corki: 1,808 appearances, purity 0.20.
- Rakan ↔ Alistar distance in the final layout: **0.0398** — practically stacked,
  which is why Alistar is labelled deliberately rather than by the auto-labeller.
- Raw (uncentered) cosine used by the beat-2 bars: Rakan-Alistar **+0.513**,
  Rakan-Orianna **−0.063**. The shared mean vector is small (0.198 vs 0.797 mean
  champion norm), so the bars are honest.

## Suggested skills

`artifact-design` + `dataviz` before editing the page ·
`/technical-writing-cleanup` for prose · `cos-session-log` at the end ·
`/handoff` to leave the next handoff.

## State of the repo

All viz work is committed on `main`. The only uncommitted things in the repo are
an unrelated ban-attribution workstream (`scripts/ban_attribution_*.py`,
`data/processed/expcache_*`, `docs/2026-07-25-ban-attribution-results.md`) and a
modified `docs/ROADMAP.md`. **Leave those alone** — they are a different piece of
work and were deliberately not touched.
