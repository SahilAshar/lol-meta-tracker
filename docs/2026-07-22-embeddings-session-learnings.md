# What I Learned About My Draft Model

**Date: 2026-07-22**

This is my own reference note. Tonight the pieces finally clicked: what my
v0.8 draft-prediction model actually is, how its numbers get tuned, and why
it works the way it does. I wrote it down so the understanding sticks. The
audience is me — a software engineer, not a machine-learning person. Every
term gets explained the first time it shows up.

## My model is a tiny language model

The v0.8 model is a small transformer. It is the same design family as GPT
and Claude — the "Attention Is All You Need" architecture. It is a *causal*
(decoder-style) transformer, which means it reads left to right and predicts
what comes next.

Here is the mapping that made it click. In a language model:

- Words are **tokens**. Each token is one unit the model reads.
- A sentence is a sequence of tokens.
- The training job is **next-token prediction**: given the tokens so far,
  guess the next one.

In my draft model, the exact same thing happens, but with champions instead
of words:

- Each **champion is a token**. My vocabulary is about 153 to 168 champions.
  A big language model has around 100,000 words. Mine is tiny by comparison.
- A **draft is a 20-token "sentence"** — the 20 picks and bans in order.
- The training job is still next-token prediction: given the draft so far,
  guess the next pick or ban.

There is even a standard language-model trick inside it called **weight
tying**. The same table of numbers that reads a champion *in* is reused to
score champions *out*. One table, two jobs. This is a common technique in
real language models, and my model uses it too.

So what makes mine different from a chatbot? Two things:

1. **Scale.** Mine is thousands of times smaller (more on this below).
2. **Esports extras.** I feed in things a chatbot has no concept of: which
   draft slot we are on, which side (blue or red), which league, whether the
   series is "fearless" (champions used earlier can't be reused), and which
   game in the series it is. I also apply an **availability mask** — a filter
   that zeroes out champions that are already picked or banned, so the model
   can only predict a legal champion.

## It has about 1.2 million tuned numbers

The exact count is **1,225,113 parameters**. A parameter is just one number
the model learns. I got this by building the model in code (config
`d192x4L6H`, vocabulary 153) and counting.

Where they live:

- **Encoder: 1,188,096 numbers (97%).** The encoder is the stack of
  transformer layers that does the actual "thinking" about the draft.
- **Champion embeddings: 29,376 numbers.** That is 153 champions × 192
  numbers each (the embeddings are explained next).
- **Everything else: about 7,600 numbers.**

The encoder has 6 layers. Each layer holds about 297,000 numbers: four
192×192 attention matrices (about 148,000) plus a small feed-forward network
that goes 192 → 384 → 192 (another 148,000 or so).

For perspective: my model is roughly **100,000 times smaller than a frontier
language model**. And role structure *still* emerges from it. That is the
surprising part.

## What an "embedding" actually is

Each champion gets **192 numbers**. That list of 192 numbers is its
**embedding** — think of it as the champion's coordinates in a 192-dimensional
space.

Two things took me a while to separate:

**All champions share one coordinate system.** Slot 7 in the list means the
same axis for Rakan as it does for Orianna. This is required, not a choice.
The model compares two champions by multiplying their numbers slot for slot,
so the slots have to line up.

**But no single slot has a nameable meaning.** Slot 7 is *not* "tankiness."
No axis is "damage" or "range." Here is the proof: you could rotate the whole
192-dimensional space, and the model's predictions would not change at all.
The meaning does not live in any one axis. It lives only in **where champions
sit relative to each other** — which champions are close, which are far.

## How the numbers actually get tuned

This is the part I kept asking about. It is **not** trial and error, and it
is not someone hand-setting values.

The whole model is **one big formula**. You put a draft in one end, numbers
flow through, and out the other end comes a single **error score** — how
wrong the guess was. The formula is *differentiable*, which means for every
number in it, you can compute a slope.

That slope is the key. **Backpropagation** is the method that computes, for
all 1.2 million numbers at once, this exact fact: "if this number went up a
tiny bit, the error would change by exactly *this* much, in *this* direction."
Every number gets its own slope. Then every number takes a tiny step in the
direction that lowers the error. Repeat thousands of times.

The analogy that finally landed for me: imagine **1.2 million thermostat
dials**. Each dial has a little gauge next to it showing which way to turn it
to reduce the error. Training reads every gauge and nudges every dial a hair,
over and over. No guessing. Each step is calculated.

### Why roles emerge on their own

Nobody tells the model "Rakan is a support." So why do supports end up
clustered together?

The answer is **substitutability**. If two champions get used in
interchangeable spots in drafts, the tuning process keeps giving them similar
nudges. Similar nudges pull them toward the same coordinates. Over thousands
of steps, champions that play the same role drift together — purely because
they show up in the same kinds of draft situations.

I have evidence from tonight. I re-ran training with instruments attached
(single seed 16, the production config, the same 2024–2026 data and the same
split as production, 5,043 training games, with the held-out test set left
untouched). I measured **role purity**: take each champion, find its 5
nearest neighbors in the space, and check how many share its role.

- At the start (random numbers), role purity was **0.195**. Pure chance is
  about 0.20. So at the start, the space knew nothing.
- At the end, role purity was **0.704**.

And the training never once mentioned roles. It only ever tried to predict
the next pick. Roles fell out for free.

Concrete pairs, using cosine similarity (a closeness score from -1 to +1)
after removing a shared drift direction:

- **Rakan and Alistar** (both supports): went from **-0.04** (basically
  strangers) to **+0.56** (clearly related).
- **Rakan and Orianna** (support vs. mid): stayed near zero (**+0.04** to
  **+0.09**). Different roles, so they never pulled together.

## Why 192 dimensions and not 768

Real language models use wide embeddings, often 768 numbers or more. I use
192. Three reasons this is the right call:

1. **The experiment said so.** I swept several configs and picked `d192x4`
   because it scored best on validation data. This was measured, not
   assumed.
2. **I don't have enough data for wider.** The encoder's size grows with the
   *square* of the width. At 768 wide, the model would balloon to about 19
   million numbers, learning from only 100,836 draft decisions — roughly 190
   numbers per example. It would **memorize** the training data instead of
   learning general patterns. Big language models get away with wide
   embeddings because they train on *trillions* of tokens. I have thousands
   of games.
3. **Size was never my bottleneck anyway.** My transformer loses the
   ban-prediction contest to the gradient-boosted model (the GBM, a different
   model type I also trained). But it loses because it is **date-blind** — it
   cannot see today's meta — not because it is too small. Making it wider
   would not fix that.

## The real problem I spotted: mimicry vs. advantage

Here is the thing that bugs me. My model predicts what pros **will** pick. It
does not predict what pros **should** pick to win. Win rate appears **nowhere**
in the training. The model is a very good mimic of pro behavior, and nothing
more.

A tool that actually helps a coach needs to model *outcomes*, not just copy
what everyone already does.

My rough sketch for v0.9:

- Add a **second output** that predicts P(win | draft so far) — the chance of
  winning given the draft up to this point.
- Rank candidate picks by predicted win probability, not by "what's popular."
- Let the mimic side propose realistic candidates, and let the outcome side
  rank them. Realistic *and* good.

Two hard problems stand in the way:

1. **Team strength drowns out the draft.** Strong teams win with lots of
   drafts. I would need a team-strength baseline to subtract that out first.
2. **The draft signal might be tiny.** It may be that the draft barely moves
   the win rate at all.

So the honest first test is small and falsifiable: **does knowing the draft
improve held-out win prediction over a baseline that only knows the side and
the team strength?** If yes, keep building. If no, that is a real,
publishable negative result — still worth writing up.

## Visualization: what worked and what didn't

I tried three times tonight to visualize the embedding space, plus a research
pass. The full write-up is in `docs/viz-research-report.md`. The short
version:

- **Attempt 1 — flat PCA projection: failed.** PCA flattens 192 dimensions
  down to 2. But the clusters I care about curve through those 192
  dimensions, so flattening hid them.
- **Attempt 2 — animated t-SNE: failed, for known reasons.** t-SNE is a
  popular way to draw high-dimensional data in 2D. But its cluster sizes and
  the gaps between clusters carry **no reliable meaning** — this is
  documented in Distill's "How to Use t-SNE Effectively." Re-drawing the
  layout every frame just animates that noise. Also, discrete snapshots beat
  continuous motion for showing training over time (per TensorBoard Projector
  and DeepTracker).
- **What worked — a similarity matrix.** I made a grid: every champion
  against every champion, colored by cosine similarity, sorted so same-role
  champions sit next to each other. Every cell is a real, stable number.
  Roles show up as literal **bright squares along the diagonal**. Nothing is
  hand-wavy.
- **One necessary trick.** All champions drift in one shared direction as
  their numbers grow during training. I subtracted that shared direction out
  first. That exposed the real contrasts between champions instead of the
  drift they all share.
- **A bonus finding.** There is a bright off-diagonal square between Top and
  Jungle. That is real signal, not an error: those two roles share a pool of
  bruiser champions.

## Engineering gotchas (so I don't relearn them)

- **Train on CPU, not MPS.** PyTorch's Apple-GPU (MPS) backend produces
  infinite losses when you combine `masked_fill(-inf)` with `cross_entropy`.
  Production trains on CPU for this reason.
- **lightgbm needs libomp on this Mac.** Run `brew install libomp`. Just
  importing `train_draft_model.py` pulls lightgbm in, so this bites early.
- **A few rows log scary-but-harmless infinite losses.** In the multi-year
  data, a handful of rows have their true pick marked "unavailable" by the
  fearless series logic (a known LPL 2025 overlap quirk). These produce
  infinite batch losses, which gradient clipping quietly turns into no-op
  batches. Harmless, but alarming in the logs — now I know why.
- **The full dataset builds locally.** All 102,916 decisions build via
  `scripts/draft_dataset.py --years 2024 2025 2026`.

## Where this is heading

The next ambition is an article. Working title: **"Attention Is All You Need
(to Win Draft)."** The story is how I stumbled into building a transformer by
climbing honest rungs — baselines first, then the GBM, then the transformer,
then blending them. The audience is technical. Noah's team at Anthropic asked
for something publicly visible, and this is the candidate.

## What's still open

- **The v0.9 outcome head.** Decide whether to build the win-probability
  output. The go/no-go is the falsifiable test above.
- **The north-star conversation.** Settle what this tool is really *for* —
  mimic pros, or advise a coach. The two point in different directions.
- **The visualization is not final.** The similarity matrix works, but I
  haven't landed the finished version.
- **The article.** Still to be written.
