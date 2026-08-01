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
| #5 | `build_evo_panel5.py` | `artifact-v7.html` | `viz-attempt5` | Four beats: (1) the model's one job + whole vector as signed bars at 3 moments, (2) the mechanism *demonstrated* from the draft record — substitutes are never teammates — plus sorted bars with trend lines and the similarity curves, (3) the map with the claim scoped to the measured number, (4) new: the exceptions, exposure quartiles + flex picks | **Published 2026-07-27** and seen. Verdict: beat 2's team-comp spine lands, beat 1's on-ramp is the right entry point, beat 4 is the right content with a weak landing |
| #6 | `build_evo_panel6.py` | `artifact-v8.html` | `viz-attempt6` | Same four beats as #5 — **no structural change**. A cleanup pass: prose cleaned, beat 3's sparkline rebuilt, beat 4 re-landed, one stat-card template, one spacing scale, 4-up tab grid, monospace reserved for numerals | **Published 2026-07-28.** Independently reviewed: dark 860px clean; 390px returned **two blockers** (clipped pair-card text hiding the 93.9/64.4 figures, and Corki/Alistar labels fusing on the final frame), both fixed and republished the same day |
| #7 | `build_evo_panel7.py` | `artifact-v9.html` | `viz-attempt7` | **The comic-strip idiom, adopted.** Beat 3 rebuilt as a **2×2** grid of four data-chosen keyframes with per-frame captions and the animation kept **below** the grid rather than hidden. A real line-up — Hanwha Life Esports, game 3 of the MSI 2026 final — is highlighted across every frame and the map. Beat 2 **restructured** (constraint lifted by Sahil): opens on that line-up, closes on a payoff figure + counted table contrasting partners, rivals and unrelated pairs. Beat 3's prose retuned for a **League-literate, ML-naive** reader; beats 1 and 4 left alone | **Published 2026-08-01.** Self-verified over 20 renders (4 beats × 2 themes × 2 widths + proven final frames): zero console errors, zero blank canvases, no horizontal overflow. **No independent review was commissioned** — that check is outstanding, not passed |

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
- **On #5, from a naive-reader subagent (no League, no ML), 2026-07-25:** the
  panel *does* land the intended takeaway — their unprompted restatement was
  correct. But the beat-2 strips were unverifiable: *"Alistar's row is, to my eye,
  indistinguishable from Orianna's… You asked me to verify a claim visually and my
  eyes contradicted you. That is worse than showing no visual at all — it made me
  distrust the rest of the panel, including the numbers."* Fixed by encoding sign
  as direction plus a running-average trend line. They also falsified the
  replacement beat-3 headline ("Every champion…") against the map's mixed right
  side, and pointed out beat 1 never said what the model does. Both fixed.
- **Design subagent, 2026-07-25:** MUST-FIX colour collision — blue/orange meant
  "above/below zero" in beats 1–2 and "Top lane / Bot lane" on the beat-3 map.
  Fixed by the same change. Their remaining ranked items are the cleanup queue in
  `HANDOFF-viz-v7.md`.
- **On #5, from Sahil, 2026-07-28 (first time he saw it rendered):** beat 2's
  team-comp framing "lands a lot better — let's keep it as the spine". Beat 1 is
  the right entry point. Beat 4 is the right content but a **weak landing** →
  fixed in #6 with a closing paragraph that ties the exceptions back to beat 3's
  jumble, and the purity definition demoted out of the last line. On the idiom
  question he chose **prototype first, then decide** — see
  `artifact/proto_comic_beat3.py`. New request, parked as out of scope for this
  panel: *"I really want to show a visualizer of how the model predicts champ to
  champ, i.e. can we step through a draft together?"*
- **On the comic-strip prototype, from Sahil, 2026-08-01:** *"I really like the
  comic strip… the captions below make a huge difference."* Direction: **2×2 grid,
  not 1×4**, with the animation **below** it. And **highlight a full team comp**,
  not one dot — *"you can pick a game maybe from MSI and be like, this is how these
  five champ comps came together."*
- **Audience redefined, 2026-08-01 — this supersedes the old cardinal rule.**
  Previously: *assume the reader does not play League and does not know ML.* Now:
  *"we are making this for an audience of people that know League of Legends, have
  watched Pro League… but may not understand ML. When I watch a match, I
  intuitively understand how draft works. How do I tie what the model sees to what
  I see on TV?"* Applied to beat 3 and the new beat-2 material only; beats 1 and 4
  still carry the old register pending his review of the new voice.
- **Two of his suggestions were tested against the data and rejected**, and are
  guarded by `check_claims()` C8 so they cannot creep back in:
  *"the five champions land in five different role regions"* — only **3 of 5** do,
  and the best comp in the entire MSI final manages 4 of 5; and *"the comp fans out
  as it trains"* — mean pairwise distance among the five goes **0.509 → 0.429**,
  slightly closer together, not further apart.

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

## What changed in #6, and why

Cleanup only — the four-beat structure is untouched, because Sahil had just seen
it and endorsed the spine. Named defects from the #5 cleanup queue:

| Item | Was (#5) | Is (#6) |
|---|---|---|
| Beat-3 sparkline | 56px tall, its title printed *inside* the plot box over the line's own baseline, a bare playhead tick hanging past the axis that read as a clipping artifact, tick text in `--faint` (**2.9:1** on the light card — genuinely illegible) | 96px, title above the box (wrapping to two lines under ~500px so the honesty note survives at 390px), two labelled y-ticks at 0.20 and 0.70, `chance` named at the line's right end and held clear of the playhead, text in `--muted` |
| Beat-4 landing | Ended on a definition of purity | Ends on the payoff: the jumble names the champions the pros themselves will not file under one role. Definition demoted to a provenance line |
| Beat-4 cards | Four boxes, three different internal structures | One `statCard` template — heading / number / caption — same padding, same 1px hairline. Identity carried by heading colour, not border weight |
| Spacing | Figure→prose gaps 17/46/33/27px, intro→figure 22–40px | One scale: 16px between prose blocks, 32px around figures. Selectors prefixed `#evo4-panel` on purpose — the page's own `.panel .sub` is (0,2,0) and silently beat a bare `.evo4-beat > *` |
| Tab row | Flex wrap 3+1, `Next →` stranded in the gutter, and a non-tab child inside `role="tablist"` | 4-up grid (2×2 under 640px), `Next →` on its own right-aligned row outside the tablist |
| Monospace | On tab labels, legend, prose captions, champion names | Numerals and axis values only |

Two bugs found while verifying, neither previously known:

- **Dark-mode card colours were baked at load.** `buildBeat4Static` ran once as an
  IIFE, so the champion headings kept the *light* role palette after a theme
  toggle. Now rebuilt on every beat-4 render.
- **`#b2end` in the test harness never reached the final frame.** Selecting beat 3
  starts autoplay; setting the scrub to max while the loop is running makes the
  next `tick()` overrun the end, finish, and reset to frame 0. The harness then
  screenshotted **frame 0** while claiming to be the last frame — so "verified on
  the final animation frame" in `HANDOFF-viz-v7.md` was not true. Fixed by pausing
  before scrubbing. The animation itself was never broken.

## Rules for the next attempt

1. **New attempt = new builder** (`build_evo_panel7.py`) + **new output**
   (`artifact-v9.html`) + **new tag** (`viz-attempt7`). Never mutate a shipped
   builder — that is what makes rollback trivial. This holds for a pure cleanup
   pass too: #6 changed no structure and still got its own builder, because #5
   was already published and must stay reproducible.
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

Layout distances, measured in the **unit square the browser actually draws**
(`P_unit` in `build_evo_panel7.py`, i.e. the base64 positions ÷ 1000) — not in
`P_final`, whose two axes are normalised independently and whose distances do
**not** survive the round-trip:

- Rakan ↔ Alistar: **0.026**. ⚠️ Earlier handoffs and a code comment in
  `build_evo_panel6.py` carried **0.0398** and listed it under "verified numbers".
  It was never asserted anywhere, is not printed on any page, and does not
  reproduce in any weighting. Corrected 2026-08-01 and now regression-checked.
- Rakan ↔ Xayah: **0.238** — the lane partner sits **9×** further off than the
  rival. Rakan ↔ Orianna: **0.421**.

Draft-record co-occurrence across **10,294** team line-ups:

- Rakan + Xayah: **321** together vs **45.2** expected — 7.1× chance.
- Rakan + Alistar: **0** vs **93.9**. Rakan + Orianna: **64** vs **64.4**.
- The highlighted comp: Hanwha Life Esports, **game 3 of the MSI 2026 final**
  (`LOLTMNT01_424505`, 2026-07-12) — Olaf, Xin Zhao, Ryze, Xayah, Rakan.
