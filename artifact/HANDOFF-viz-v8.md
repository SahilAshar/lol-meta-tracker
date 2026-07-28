# Handoff: attempt #6 is live — the open question is the idiom (v8)

You are picking up a session with Sahil about a data-viz panel embedded in a
League-of-Legends draft-model explainer. **Attempt #5 was published 2026-07-27 —
the first time he had seen any of this rendered — and attempt #6, a cleanup pass
on top of it, was published 2026-07-28.** Both went to the same URL.

The structural work is done and endorsed. The one thing genuinely open is whether
to rebuild the panel in the **comic-strip idiom**. A prototype exists and is
waiting on his verdict. Do not start that redesign without his word.

## Cardinal rules (learned over six attempts)

1. **Do not rebuild before confirming.** Interrogate → mock → confirm → build.
   Three attempts died from building first.
2. **Every number on the page must be real**, pulled from the saved snapshots or
   the draft record. Anything illustrative says so, in the figure where possible.
   This now extends to *annotations*: the prototype's own claims carry asserts.
3. **Plain language, blog-reader default.** Assume the reader does not play League
   and does not know ML. Short sentences, active voice.
4. **A visual the reader cannot verify is worse than no visual.**
5. **New — a harness that lies is worse than no harness.** See the `#b2end` bug
   below. Before trusting any screenshot check, prove it is looking at what it
   claims to be looking at.

## Where things stand

**Live artifact (always republish IN PLACE, favicon 🌲, never mint a new URL):**
https://claude.ai/code/artifact/36f462bc-ef98-42dd-af89-6c1a309c8877

| | |
|---|---|
| Live now | attempt #6 — `artifact/artifact-v8.html`, tag `viz-attempt6` |
| Rollback one step | publish `artifact/artifact-v7.html` (tag `viz-attempt5`) |
| Rollback two steps | publish `artifact/artifact-v6.html` (tag `viz-attempt4`) |

Rollback never needs a rebuild — every attempt's full output HTML is committed.

## What Sahil said when he finally saw it (2026-07-28)

- **Beat 2 lands.** *"It lands a lot better, let's keep it as the spine."* The
  team-comp framing — a team fields one champion per role, so two supports compete
  for one seat, 0 of 10,294 line-ups — was his own idea and is now load-bearing.
  **Do not restructure beat 2.**
- **Beat 1's on-ramp is right.** "Right entry point." Settled; leave it.
- **Beat 4: right content, weak landing.** Fixed in #6 — it now closes on the
  payoff (the jumble names the champions the pros themselves will not file under
  one role) instead of on a definition of purity.
- **The idiom: prototype first, then decide.** See below. This is the live question.
- **New ask, parked:** *"I really want to show a visualizer of how the model
  predicts champ to champ, i.e. can we step through a draft together? maybe out of
  scope for this specific visual artifact."* He flagged the scope himself. It is
  **not** started. It is a genuinely different exhibit — a draft-state stepper
  reading `draft_sequences_multi.parquet` and the shipped model's actual
  next-pick distribution — and probably its own panel or its own page. Raise it;
  do not silently fold it into this panel.

## THE OPEN DECISION — the comic-strip idiom

After #4 he asked for "a mix of keeping the motion and the annotated comic-strip
idiom." The motion was kept and the claims were annotated, but the
keyframe/comic-strip **structure** was never built. He chose *prototype it first,
then decide from the pixels*.

**The prototype exists:** `artifact/proto_comic_beat3.py` →
`proto_comic_beat3.html` / `_dark.html`. It rebuilds **beat 3 only** as four
annotated static keyframes with the animation demoted to a `<details>`. It is
wired into nothing and publishes nowhere.

- Keyframes are chosen **from the data**, not by eye: first snapshot, first to
  clear 0.30 purity, first to clear 0.50, and the last → frames 0 / 17 / 26 / 55,
  purity 0.219 / 0.310 / 0.511 / 0.704.
- Its annotations carry their own asserts. "Bot and Support separate before Top
  and Mid" is checked (at frame 17: Bot 0.650, Support 0.622 vs Top 0.380, Mid
  0.412). "Top sorts last" is checked at frames 26 and 55.
- Those per-role numbers are measured in the **2-D layout**, not the raw 192-dim
  space, so they are deliberately **not printed on the page** — the page only ever
  shows `D.purity`, the raw score. Two different purity numbers on one figure
  would be worse than none. Keep that separation if you extend it.

**The case for it** (from `docs/viz-research-report.md`): panel layouts beat
vertical scroll on comprehension (Zhi et al., CGF 2019, n=180); scrolling buys no
engagement advantage over stepping (McKenna et al., via Distill); discrete content
wants discrete interaction (Kosara); a button gate "makes a promise it hasn't
earned" whereas small multiples pay off *before* any click. Observed in the
prototype: the whole beat-3 argument fits in ~550px versus ~1400px gated.

**The case against:** it is a redesign of a panel he has now twice endorsed the
structure of, and full scrollytelling is ruled out anyway — the panel sits inside
a larger post and cannot own the page scroll.

If he says go: that is **attempt #7** — new builder `build_evo_panel7.py`, new
output `artifact-v9.html`, new tag `viz-attempt7`, VERSIONS row first.

## What #6 actually changed

Cleanup only; the four beats are structurally identical to #5.

1. **Beat-3 sparkline rebuilt.** Was 56px with its title printed *inside* the plot
   box over the line's own baseline, a playhead tick hanging past the axis that
   read as a clipping artifact, and tick text in `--faint` — **2.9:1** on the light
   card, genuinely illegible. Now 96px, title above the box (wrapping to two lines
   under ~500px so the honesty note survives at 390px), two labelled y-ticks at
   0.20 and 0.70, `chance` named at the dashed line's right end and held 8px clear
   of the playhead, all text in `--muted`.
2. **Beat 4 re-landed**, per his note.
3. **One stat-card template** for all four beat-4 boxes (heading / number /
   caption). Identity via heading colour, never border weight. No inner boxes.
4. **One spacing scale** — 16px between prose blocks, 32px around figures.
5. **Tab row** is a 4-up grid (2×2 under 640px), `Next →` on its own right-aligned
   row and out of the `role="tablist"` it was never a valid child of.
6. **Monospace reserved** for numerals and axis values.
7. **Prose pass** over all four beats. Every honesty caption and hedge intact.

### Four bugs found while verifying — none previously known

- **`#b2end` never reached the final frame.** Selecting beat 3 starts autoplay;
  setting the scrub to max while the loop runs makes the next `tick()` overrun the
  end, finish, and reset to frame 0. The harness screenshotted **frame 0** while
  labelling it the last frame — so *"verified on the final animation frame"* in
  `HANDOFF-viz-v7.md` was **not true**. Fixed by pausing before scrubbing. The
  animation itself was never broken (it reaches 0.70 and shows "Replay").
- **Beat-4 card colours were baked at load.** An IIFE built them once, so champion
  headings kept the *light* role palette after a theme toggle. Now rebuilt per
  render.
- **Line-chart direct labels wore the series colour.** At `--meta` pink that is
  **2.6:1** as text on the light card — passes as a mark, fails as text. Now an ink
  token, with the line's end dot carrying identity.
- **Beat-4 bar chart reserved 54px** for range/count labels it drops below 560px,
  leaving a visible hole at 390px. Height is now width-dependent.

## STILL UNVERIFIED — carried over unchanged from the v7 handoff

**Dark mode and 390px have never had an independent review.** This was must-fix #3
last session and it is must-fix #3 again. In v7 the robustness subagent never
returned a report; this session two reviewers (dark, mobile) were spawned against
the 20 rendered frames and also never reported, across four follow-ups.

What this surface *does* have: my own eyes on all 20 configs (4 beats × 2 themes ×
2 widths + the real final frame), zero console errors in every one, a
`getImageData` ink check proving the canvases are not blank, and the palette
validator passing on both surfaces. That found four real bugs — but it is one pair
of eyes, and the same pair that wrote the code.

If you do nothing else from this handoff, get this reviewed. The screenshots
regenerate in about a minute (see the harness section).

## Remaining queue (nothing here is disqualifying)

**Should-fix**

1. **Beat 2 is doing two beats' work** — three figures with three label treatments,
   and the panel still jumps height on one click of Next. Either split it or set a
   min-height across beats. *He was asked and chose to keep beat 2 as the spine, so
   splitting it is a bigger call than it looks — ask before splitting.*
2. **`dips` label at 390px** sits close to the `0.91` value label. Not overlapping
   in any rendered config, but it is the tightest spot on the page.

**Worth considering (from the research, still not acted on)**

- **Text-to-visual linking** — highest-evidence mechanic in the pacing research
  (Zhi et al., n=180: measured engagement win *and* recall improvement). Hovering a
  champion's name in the prose highlights its dot on the map.
- **Encode arrival in the marks** on beat 3: in-transit dots pale, settled dots
  crisp. Turns the paced-not-mapped disclosure from prose into something visible.
  Also: tick the real snapshots on the scrubber so interpolated spans show.
- **A "show raw per-frame t-SNE" toggle** — turns the longest disclaimer into a
  button. Distill's own move: demonstrate rather than hedge.

## Known and accepted, do not "fix" silently

- **Light-mode role palette WARNs on contrast.** Mid `#e87ba4` (2.62:1) and Bot
  `#eda100` (2.11:1) sit below 3:1 against the light card. All other checks PASS in
  both modes. This is the project's established role palette
  (`scripts/chart_embeddings_v08.py`), shared with the PNG charts, so changing it
  here would desync the whole project. Relief in place: a text legend, four direct
  labels, and a hover tooltip naming champion and role. Dark mode passes contrast
  outright. If you ever do change it, re-validate both surfaces and update the PNGs
  in the same change.
- **Alistar sits 0.0398 from Rakan** in the final layout — practically stacked,
  which is *the payoff of beat 2*, so he is labelled deliberately rather than by
  the auto-labeller. The auto-labeller drops him; that is expected.
- **Beat-4 bars are drawn opaque.** Alpha-compositing pushed chroma below the
  validator floor (`#827140` reads gray). Dark mode uses `#AE8C34`, not the page's
  `--gold`, which sits at L 0.768 in dark — outside the band for a large fill.
- **Blue and orange carry role meanings** (Top, Bot) on the map. Do not reintroduce
  either as a categorical hue anywhere in the panel. Sign is encoded by direction
  in neutral ink.

## The bar for shipping any change

- **Rendered and eyeballed** in both themes, at 390px, and on the final animation
  frame, with **zero console errors**. Prove the final-frame shot is actually the
  final frame — check the scrub value and the readout, not the filename.
- **Palette re-validated** if any colour changed, on light `#FFFFFF` and dark
  `#1D2634`, via the `dataviz` skill's `scripts/validate_palette.js`.
- **`check_claims()` passes** — it runs in the builder and fails the build on drift.
- **Prose cleaned**, honesty captions intact, no new overclaim.
- **Republished in place** with a WebFetch round-trip confirming it is live.
- Then say plainly what you verified and what you did not.

## Assets and how to build

- Repo: `~/Documents/repos/lol-meta-tracker` (venv `.venv/bin/python`).
- **Builder:** `.venv/bin/python artifact/build_evo_panel6.py` — splices into
  pristine `artifact/artifact-v2.html`, writes `artifact/artifact-v8.html` (the
  publishable file) plus `evo_panel6_test.html` and `evo_panel6_test_dark.html`.
  Test pages accept `#b0`…`#b3` and `#b2end`.
- **Prototype:** `.venv/bin/python artifact/proto_comic_beat3.py` (imports the
  builder, so it rebuilds #6 as a side effect — idempotent).
- **Claim checks:** `.venv/bin/python scripts/viz_claim_checks.py`.
- Snapshots: `data/processed/embedding_evolution_v08_snapshots.npz` (56 × 168 × 192,
  plus `role`, `flex`, `labels`, `val_loss`, `purity`).
- Draft record: `data/processed/draft_sequences_multi.parquet` (5,147 games,
  2024-01-13 → 2026-07-19). Source for exposure and co-occurrence.
  **Do not use `pick_rates.csv`** — it covers only some leagues and leaves 41 of
  168 champions unmatched, silently corrupting any exposure analysis.
- Role colours (light): top `#2a78d6` / jng `#008300` / mid `#e87ba4` /
  bot `#eda100` / sup `#4a3aa7`. Dark: `#3987e5` / `#008300` / `#d55181` /
  `#c98500` / `#9085e9`. Defined in `scripts/chart_embeddings_v08.py`.
- Research + rationale: `docs/viz-research-report.md`,
  `docs/viz-selfgrade-2026-07-25.md`, `artifact/VERSIONS.md`.

### Screenshot harness

No Chrome app on this Mac. Use the Playwright headless shell directly:

```
~/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell
```

`npm i playwright-core` in a scratch dir, then `chromium.launch({ executablePath: EXE })`.
Three gotchas that have each cost real time:

- **Keep `deviceScaleFactor: 1`.** At 2, `<canvas>` element screenshots come back
  **blank** and you will "find" rendering bugs that do not exist. Verify any
  suspected blank canvas with `getImageData` before believing it.
- Use `evo_panel6_test_dark.html` for dark. A media-query dark preference alone
  does not stamp `data-theme`, and the colour tokens key off the attribute.
- **Assert the state you think you screenshotted.** `#b2end` silently gave frame 0
  for a whole attempt. Read back `scrub.value` and the frame readout.

## Constraints (non-negotiable)

- Artifact CSP: no external libs, fonts, or network. Vanilla JS + canvas,
  precompute in Python, assets as data URIs. Added payload ≈58KB; stay well under
  ~600KB.
- Both themes via the page's existing CSS tokens (`--model`, `--meta`, `--gold`,
  `--ink`, `--muted`, `--card`, `--hairline`, `--inset`, `--faint`). Respect
  `prefers-reduced-motion` — beat 3 steps discretely under it.
- **Watch CSS specificity.** The page's own `.panel .sub` is (0,2,0) and silently
  beat the panel's `.evo4-beat > *` spacing rules; that is why the spacing block is
  prefixed `#evo4-panel`. Check the cascade before assuming a rule applied.

## Versioning rules (he asked for this explicitly)

**New attempt = new builder + new output + new tag.** Never mutate a shipped
builder; that is what makes rollback trivial. This held for #6 even though it was a
pure cleanup pass, because #5 was already published and must stay reproducible. Add
the `VERSIONS.md` row and the feedback-log line *before* publishing. Tag right after
the commit that produces a publishable version.

Current tags: `viz-attempt4`, `viz-attempt5`, `viz-attempt6` (live now).

## Verified numbers — do not re-derive from memory

All regression-checked by `scripts/viz_claim_checks.py` and the builder's
`check_claims()`:

- Rakan ↔ Alistar centred cosine: **−0.04 → +0.56**. Rakan ↔ Orianna: **+0.04 → +0.09**.
- 5-NN role purity: **0.219 → 0.704** (chance 0.20), Euclidean over all 168.
- Layout blobs: **8–13** across every DBSCAN setting, never 5. KMeans(k=5) on the
  2D layout recovers roles at only **0.595** purity / **+0.288** ARI.
- Team comps: **10,294**. Rakan+Alistar together **0** (expected 93.9);
  Rakan+Orianna **64** (expected 64.4). Across the 40 most-picked champions,
  same-role pairs are teammates at **0.074×** chance, different-role at **0.936×**.
- Purity by exposure quartile: **0.443 / 0.624 / 0.914 / 0.833**. Flex **0.453** vs
  non-flex **0.736** (p = 7.9e-06). Corki: 1,808 appearances, purity 0.20. Rakan:
  1,658 appearances, purity 0.80.
- Rakan ↔ Alistar distance in the final layout: **0.0398**.
- Raw (uncentered) cosine for the beat-2 bars: Rakan-Alistar **+0.513**,
  Rakan-Orianna **−0.063**. Shared mean vector is small (0.198 vs 0.797 mean
  champion norm), so the bars are honest.
- **Prototype only, 2-D layout, never shown on the page:** per-role purity at
  frame 17 — Top 0.380 / Jungle 0.552 / Mid 0.412 / Bot 0.650 / Support 0.622; at
  frame 26 — Top 0.610 / Jungle 0.643 / Mid 0.747 / Bot 0.892 / Support 0.844.

## Suggested skills

`artifact-design` + `dataviz` before editing the page ·
`/technical-writing-cleanup` for prose · `cos-session-log` at the end ·
`/handoff` to leave the next handoff.

## State of the repo

All viz work is committed on `main` and tagged. Untouched and unrelated: the
ban-attribution and pocket-conditioning workstreams
(`scripts/ban_attribution_*.py`, `scripts/experiment_v13/v14/v15_*.py`,
`scripts/analysis_counter_swing.py`, `data/processed/expcache_*`,
`docs/2026-07-28-*.md`) and a modified `docs/ROADMAP.md`. **Leave those alone** —
different piece of work, deliberately not touched.
