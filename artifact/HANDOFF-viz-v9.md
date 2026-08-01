# Handoff: attempt #7 is live — the idiom question is settled (v9)

You are picking up a session with Sahil about a data-viz panel inside a
League-of-Legends draft-model explainer. **Attempt #7 was published 2026-08-01.**
The comic-strip question that dominated three handoffs is **closed**. What is open
is narrower and mostly editorial.

## Cardinal rules (seven attempts of scar tissue)

1. **Do not rebuild before confirming.** Interrogate → mock → confirm → build.
   Three attempts died from building first. This still holds.
2. **Every number on the page must be real.** Anything illustrative says so.
3. **AUDIENCE CHANGED IN #7 — the old rule is dead.** It used to be "assume the
   reader does not play League and does not know ML." It is now: **knows League,
   watches pro play, does NOT know ML.** The job is to bridge what a viewer
   already reads in a televised draft to what the model encodes. Applied so far to
   **beat 3 and the new beat-2 material only** — beats 1 and 4 still carry the old
   register, pending his review of the new voice. *That is the top open item.*
4. **A visual the reader cannot verify is worse than no visual.** This killed two
   things in #7: a caption naming Alistar when Alistar was not drawn, and a
   greyed-out field that hid the very role separation the captions asked the
   reader to check. Watch for it.
5. **A harness that lies is worse than no harness.** Prove the state you
   screenshotted by reading it back.
6. **Check his suggestions against the data before building them.** Two of his
   ideas in #7 were false and were rejected. He was fine with it. See below.

## Where things stand

**Live artifact (always republish IN PLACE, favicon 🌲, never mint a new URL):**
https://claude.ai/code/artifact/36f462bc-ef98-42dd-af89-6c1a309c8877

| | |
|---|---|
| Live now | attempt #7 — `artifact/artifact-v9.html`, tag `viz-attempt7` |
| Rollback one step | publish `artifact/artifact-v8.html` (tag `viz-attempt6`) |
| Rollback two steps | publish `artifact/artifact-v7.html` (tag `viz-attempt5`) |

Rollback never needs a rebuild — every attempt's output HTML is committed.

**Publishing gotcha:** the Artifact tool refuses to publish if this session has not
seen the live version. WebFetch the URL first. The summarizer truncates ~40% in and
will not reach the panel, but the tool saves the **full** HTML to a local path in
its result header — grep that file to verify a publish actually landed.

## What Sahil decided on 2026-08-01

- **The comic strip is in.** *"I really like the comic strip… the captions below
  make a huge difference."* Captions are load-bearing — do not trim them.
- **2×2, not 1×4**, with the animation **below** the grid, not hidden.
- **Highlight a full team comp**, not one dot: *"you can pick a game maybe from
  MSI and be like, this is how these five champ comps came together."*
- **Merge the comp exhibit into beat 2** — chosen over keeping beat 3 self-contained.
  This lifted the long-standing "do not restructure beat 2" constraint. It was
  lifted **by him**, not overridden.
- **Retune beat 3 only, for now**, so he can judge the new voice on one beat.

## The exhibit, and why it is what it is

The bridge is a **flip a pro viewer already half-knows**:

| pair | drafted together | expected | map distance |
|---|---|---|---|
| Rakan + **Xayah** — lane partners | **321** | 45.2 | **0.238** |
| Rakan + **Alistar** — same seat | **0** | 93.9 | **0.026** |
| Rakan + Orianna — unrelated | 64 | 64.4 | 0.421 |

The champions you would *never* draft together are the ones the model **stacks**;
Xayah, Rakan's actual lane partner, sits **9× further away**. Nobody told the model
either fact. That is beat 2's new closing.

The comp is **Hanwha Life Esports, game 3 of the MSI 2026 final**
(`LOLTMNT01_424505`, 2026-07-12): Olaf, Xin Zhao, Ryze, Xayah, Rakan — one per role.
Chosen because it contains Rakan (the panel's existing thread) *and* Xayah.

### Two of his ideas were TESTED AND REJECTED — do not reinstate by accident

Both are guarded by `check_claims()` **C8**, which fails the build if they ever
become true without the copy being rewritten deliberately:

- ❌ *"the five champions land in five different role regions."* Only **3 of 5** do.
  The best comp in the entire MSI final manages 4 of 5.
- ❌ *"the comp fans out as it trains."* Mean pairwise distance among the five goes
  **0.509 → 0.429** — slightly *closer together*, not further apart.

He was told plainly and accepted it. Do not quietly re-add either.

## Record correction carried out in #7

`0.0398` for the Rakan–Alistar layout distance appeared **once in the entire repo**,
as a JS code comment, and was listed under "verified numbers" in earlier handoffs.
It was **never asserted, never printed, and does not reproduce** — it came from
`P_final`, whose two axes are normalised independently. The drawn value is
**0.0262**. There is now a `P_unit` array (the base64 positions ÷ 1000, i.e. exactly
what the browser draws) and **every distance quoted on the page is measured there**.
Nothing shipped was ever wrong; the record was.

## What #7 actually changed

- **Beat 3 rebuilt**: 2×2 grid of four keyframes chosen *by score* (first snapshot,
  first ≥0.30, first ≥0.50, last → frames 0/17/26/55, purity 0.219/0.310/0.511/0.704),
  each with a checked caption, comp highlighted, animation below the grid, purity
  sparkline retained. Prose retuned for the new reader.
- **Beat 2 restructured**: opens on the real line-up, keeps its endorsed pair figure
  and strips untouched in the middle, then closes on a new payoff figure (final
  layout, only Rakan/Alistar/Xayah/Orianna lit, distances annotated) + counted table
  + the flip paragraph.
- **Beats 1 and 4**: untouched.
- **Label placer rewritten**: two passes. Pass 1 avoids labels, comp dots *and* all
  163 other dots; pass 2 relaxes to labels + comp dots only, leaning on the halo
  stroke. It drops a label only if both fail. This replaced "drop rather than
  collide", which would have cost the comp labels at 390px.
- **Reduced motion is now strictly better**: no autoplay, and all four keyframes +
  captions render statically, so the whole argument survives without motion.

### Three defects found *while verifying* #7, all fixed before publish

- A caption still read *"Grey dots are the other 163 champions"* after the design
  changed to keep every dot role-coloured. Greying the field had hidden the role
  separation the captions ask the reader to verify.
- The frame readout inherited `text-align: right` from a stale **ID** rule after
  moving onto its own line — the ID beat the new class silently. (The cascade bites
  this panel repeatedly. Check specificity.)
- Map labels could be drawn on top of dots — the first mock literally printed
  "Xayaḥ".

## Verification actually performed on #7

- **20 renders**: 4 beats × 2 themes × 2 widths, plus proven final frames.
- **Zero console errors, zero blank canvases** across all of them.
- **Final frame proven** — `scrub 55/55`, readout `epoch 46 · purity 0.70`, on a
  re-read *after settle*, in both themes at both widths. The `#b2end` lie did not
  recur.
- **No horizontal overflow**: `scrollWidth == clientWidth == 390` on every beat.
  Two flagged elements were false positives (the digits table has its own
  `overflow-x:auto`; the responsive `<thead>` is clipped and absolutely positioned).
- **`check_claims()` passes**, now including C6–C9 (the comp is a real line-up on
  that date; the flip holds in both directions; the two rejected claims stay
  rejected; every keyframe caption is checked in the 2-D layout).
- **No colour changed**, so palette re-validation was not triggered.
- ⚠️ **No independent review was commissioned.** That check is **outstanding, not
  passed.** Two 390px blockers reached production in #6 exactly this way. If you
  can get a review, get one.

## Open queue

**Top item — the audience retune is half-done.** Beats 1 and 4 still explain things
to a reader who does not play League. Beat 1 defines what a draft is and what a
champion is; beat 4's wording predates the change. Ask him whether the new voice in
beat 3 lands before propagating it.

**Also open**

1. **Beat 4's bar chart drops its range/count sub-labels below 560px.** Carried over
   from #6, still unfixed. Hover restores them; mobile has no hover, so it is real
   content loss. Options: smaller font, rotated labels, or move the ranges into
   prose. *(The `.evo4-ctbl` stacking pattern added in #7 is a proven fix for
   exactly this shape of problem — reuse it.)*
2. **Beat 2 is long now.** It was already doing two beats' work; the comp intro and
   payoff made it longer (3103px at 390px, the tallest beat). Splitting it is a
   bigger call than it looks — he called it the spine. Ask.
3. **The parked ask, still not started:** *"I really want to show a visualizer of how
   the model predicts champ to champ, i.e. can we step through a draft together?"*
   He flagged it as probably out of scope for this panel. It is a genuinely
   different exhibit — a draft-state stepper over `draft_sequences_multi.parquet`
   and the shipped model's next-pick distribution. Raise it; do not fold it in.
4. **Text-to-visual linking** — still the highest-evidence unbuilt mechanic (Zhi et
   al., n=180). Hovering a champion's name in the prose highlights its dot. The comp
   highlight in #7 is a static cousin of this; the hover version is the real thing.

## Known and accepted — do not "fix" silently

- **Light-mode role palette WARNs on contrast.** Mid `#e87ba4` (2.62:1) and Bot
  `#eda100` (2.11:1) sit below 3:1 on the light card. This is the project's shared
  role palette (`scripts/chart_embeddings_v08.py`), also used by the PNG charts, so
  changing it desyncs the project. Relief: text legend, direct labels, hover
  tooltip. Dark passes outright.
- **Rakan and Alistar are practically stacked (0.026).** That is the payoff, not a
  bug. The auto-labeller drops Alistar on the main map; beat 2's payoff figure
  labels him deliberately.
- **Beat-4 bars are drawn opaque** — alpha pushed chroma below the validator floor.
- **Blue and orange carry role meanings** (Top, Bot). Never reintroduce either as a
  categorical hue anywhere in the panel.
- **Per-role purity measured in the 2-D layout is NEVER printed.** The page shows
  exactly one purity number, the raw 192-dim score. `_layout_role_purity()` exists
  only to assert captions. Two different purity numbers on one figure would be
  worse than none.

## The bar for shipping any change

- Rendered and eyeballed in **both themes, at 390px and 860px, on a proven final
  animation frame**, zero console errors. Prove the frame by reading back
  `scrub.value` **and** the readout, after a settle delay.
- Palette re-validated **if any colour changed**, on light `#FFFFFF` and dark
  `#1D2634`, via the `dataviz` skill's `scripts/validate_palette.js`.
- `check_claims()` passes — it runs in the builder and fails the build on drift.
- **VERSIONS.md row + feedback-log line BEFORE publishing.**
- Republished in place, with a round-trip that greps the saved full HTML.
- Then say plainly what you verified **and what you did not**.

## Assets and how to build

- Repo: `~/Documents/repos/lol-meta-tracker` (venv `.venv/bin/python`).
- **Builder:** `.venv/bin/python artifact/build_evo_panel7.py` — splices into
  pristine `artifact/artifact-v2.html`, writes `artifact/artifact-v9.html` plus
  `evo_panel7_test.html` / `evo_panel7_test_dark.html`. Test pages accept
  `#b0`…`#b3` and `#b2end`.
- **Prototypes (mocks, wired to nothing):** `proto_comic_beat3.py` (the 1×4 that
  lost) and `proto_comic_v2.py` (the 2×2 + comp mock that won). Both import the
  #6 builder, so they rebuild #6 as a side effect.
- Snapshots: `data/processed/embedding_evolution_v08_snapshots.npz`
  (56 × 168 × 192, plus `role`, `flex`, `labels`, `val_loss`, `purity`).
- Draft record: `data/processed/draft_sequences_multi.parquet` (5,147 games,
  2024-01-13 → 2026-07-19; leagues LPL/LCK/LEC/LCS/EWC/MSI/FST).
  **Do not use `pick_rates.csv`** — it covers only some leagues and leaves 41 of
  168 champions unmatched.
- ⚠️ **`appears()` ≠ `exposure`.** `appears("Rakan")` = 989 (team line-ups
  containing him); the beat-4 card's `exposure` = 1658 (draft appearances incl.
  bans). Both are correct and they are different quantities. Do not reconcile them.

### Screenshot harness

No Chrome app on this Mac. Use the Playwright headless shell directly:

```
~/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell
```

`npm i playwright-core` in a scratch dir, then `chromium.launch({ executablePath: EXE })`.

- **Keep `deviceScaleFactor: 1`.** At 2, `<canvas>` element screenshots come back
  blank and you will "find" rendering bugs that do not exist.
- Use `evo_panel7_test_dark.html` for dark — a media-query preference alone does not
  stamp `data-theme`, and the tokens key off the attribute.
- **Assert the state you think you screenshotted**, after a settle delay.
- For overflow, trust `documentElement.scrollWidth` vs `clientWidth`. Per-element
  bounding-box checks produce false positives on clipped and `overflow:auto` content.

## Constraints (non-negotiable)

- Artifact CSP: no external libs, fonts, or network. Vanilla JS + canvas, precompute
  in Python, assets as data URIs. Added payload ≈59KB; stay under ~600KB.
- Both themes via the page's existing tokens (`--model`, `--meta`, `--gold`, `--ink`,
  `--muted`, `--card`, `--hairline`, `--inset`, `--faint`). Respect
  `prefers-reduced-motion`.
- **Watch CSS specificity.** The page's own `.panel .sub` is (0,2,0) and silently
  beat `.evo4-beat > *`; that is why the spacing block is prefixed `#evo4-panel`.
  In #7 an **ID** rule silently beat a new class rule. Check the cascade.

## Versioning rules (he asked for this explicitly)

**New attempt = new builder + new output + new tag.** Never mutate a shipped
builder. VERSIONS.md row and feedback-log line go in **before** publishing. Tag right
after the commit that produces a publishable version.

Current tags: `viz-attempt4`, `viz-attempt5`, `viz-attempt6`, `viz-attempt7` (live).

## Verified numbers — do not re-derive from memory

Regression-checked by the builder's `check_claims()`:

- Rakan ↔ Alistar centred cosine **−0.04 → +0.56**; Rakan ↔ Orianna **+0.04 → +0.09**.
- 5-NN role purity **0.219 → 0.704** (chance 0.20), Euclidean over all 168.
- Layout blobs **8–13** across every DBSCAN setting, never 5. KMeans(k=5) on the 2-D
  layout recovers roles at only **0.595** purity / **+0.288** ARI.
- Team comps **10,294**. Rakan+Alistar **0** (expected 93.9); Rakan+Orianna **64**
  (expected 64.4); **Rakan+Xayah 321 (expected 45.2, 7.1× chance)**.
- **Layout distances in `P_unit`:** Rakan–Alistar **0.026**, Rakan–Xayah **0.238**
  (9× further), Rakan–Orianna **0.421**.
- Purity by exposure quartile **0.443 / 0.624 / 0.914 / 0.833**. Flex **0.453** vs
  non-flex **0.736** (p = 7.9e-06). Corki 1,808 appearances, purity 0.20; Rakan
  1,658, purity 0.80.
- Comp raw purity: Olaf 0.80, Xin Zhao 1.00, Ryze 1.00, Xayah 1.00, Rakan 0.80 —
  mean **0.920** vs all-168 **0.704**.
- **Never printed (2-D layout only, for asserting captions):** per-role purity at
  frame 17 — Top 0.380 / Jungle 0.552 / Mid 0.412 / Bot 0.650 / Support 0.622; at
  frame 26 — Top 0.610 / Jungle 0.643 / Mid 0.747 / Bot 0.892 / Support 0.844.

## State of the repo

All viz work is committed on `main` and tagged. Untouched and unrelated: the
ban-attribution and pocket-conditioning workstreams (`scripts/ban_attribution_*.py`,
`scripts/experiment_v13/v14/v15_*.py`, `scripts/analysis_counter_swing.py`,
`data/processed/expcache_*`, `docs/2026-07-28-*.md`, `docs/2026-07-31-*.md`) and a
modified `docs/ROADMAP.md`. **Leave those alone.**
