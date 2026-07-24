# The Model Ladder, from First Principles (v0 → v0.9)

*Saved verbatim from a 2026-07-22 session. A teaching walkthrough of every
rung of the draft model, written to answer: "teach me from first principles
how we got here." Companion to `2026-07-22-embeddings-session-learnings.md`
(which covers the embedding/training mechanics in depth) and `ROADMAP.md`
(the source of truth for current state).*

---

Here's the whole ladder, taught as a story, because each version exists to answer a question the previous one raised.

## The problem, stated from zero

A pro draft is 20 decisions in a fixed order: bans and picks, alternating between two teams. Your model's job: given everything that has happened so far in the draft, predict the next champion chosen. You score it with top-1/3/5 accuracy — "was the real answer your first guess? in your top 3? top 5?" — on games the model has never seen (the EWC July main event, 50 games, 1,000 decisions).

Everything that follows is a sequence of answers to one repeated question: **"how do we describe the state of the draft to a machine?"** The ladder is really three eras of answering it.

---

## Era 1: You describe the world (v0 → v0.7)

### The baselines — the floor you must beat

Before any model, two dumb strategies:
- **Meta baseline:** always guess whatever's been picked/banned most in the last 28 days. Gets ~9.9% top-1.
- **Team-habit baseline:** guess what this team historically plays. Worse, ~6%.

These matter more than they look. A model that beats them is learning something; a model that doesn't is an expensive random number generator. Every rung was measured against this floor.

### v0 — the reframe that makes it a machine-learning problem at all

You can't ask a classical model "what comes next in this draft?" directly. So the reframe: for each decision, list every **legal candidate** champion, describe each candidate with a row of numbers, and train a model to score rows. Highest score = your prediction.

The v0 description had 11 numbers per candidate: its recent pick rate, ban rate, presence, this team's usage of it, the opponent's usage, plus context flags (is this a ban? phase 2? blue side? which slot? fearless mode? game number in the series).

The model is a **GBM** — gradient-boosted trees. First principles version: one decision tree asks yes/no questions ("is pick_rate > 0.3?") and outputs a score. Boosting builds hundreds of small trees, where each new tree is fit specifically to correct the errors of all the trees before it. The sum of all trees is the score. It's very good at finding thresholds and interactions in *features you hand it* — and completely unable to invent features you didn't.

That last sentence is the entire plot of Era 1.

### v0.5 — teach it that a draft has structure

v0 treated each decision in isolation. But a draft is a constrained sequence: if your team already picked a jungler, you probably don't need another. Two new features: `role_need` (which roles are still open for this team) and `role_overlap_max` (how much this candidate overlaps what's already locked).

Notice what you did: you *hand-coded* sequence awareness. You looked at the draft, decided "roles are the structure that matters," and computed it yourself. The model didn't discover that — you told it.

### v0.6 — teach it that people matter

Two more features: `player_pool` (does the player in this seat actually play this champion?) and `player_wr` (how well?). Pro drafts revolve around comfort picks; now the model can see them. Again: your insight, encoded by hand.

### v0.7 — teach it about pairs, and stabilize the noise

Two additions:

1. **Pair features** (`pair_syn`, `pair_ctr`): historical synergy and counter stats between this candidate and champions already on the board. Drafts are fundamentally about combinations — Rakan wants Xayah, you ban what counters your comp.
2. **A 10-model ensemble**: 5 random seeds × 2 model families, scores averaged. Why: you discovered single fits swing ±1.5 points of top-1 just from the seed. One model's number is noise; ten models' average is signal.

Result: **13.0 / 32.5 / 42.8** vs the meta baseline's 9.9 / 26.6 / 38.0. Clearly learning.

### The ceiling that motivated everything after

Look at the pair features closely, because they're the tell. There are ~150 champions, so ~11,000+ pairs. Most pairs have appeared together a handful of times, ever. A hand-counted stat table for pairs is mostly empty or noisy. And triples? Full-team comps? Forget it.

More generally: by v0.7, *every* possible improvement was "Sahil thinks of another feature and computes it." You were the bottleneck. The model could only see the world through descriptions you invented, and the descriptions were running out.

---

## Era 2: The model describes the world (v0.8 → v0.8.1)

### v0.8 — the transformer: replace features with learned representation

The flip: stop describing champions with statistics. Instead:

- Give every champion a **learned 192-number vector** (the embeddings — random at first, tuned by gradient descent, the whole thing we spent last session on).
- Treat the draft as a **20-token sentence** and train the model to predict the next token — exactly how GPT is trained on text.
- **Attention** is what replaces your hand-coded features. `role_need`? Attention can look at all prior picks and infer what's covered. `pair_syn`? Attention relates the candidate to every champion on the board — for *all* pairs at once, through the geometry of the embedding space, not a sparse count table. The features you spent v0.5–v0.7 hand-building are now things the model can derive itself.

This is why the role clusters emerging in the embeddings mattered so much: it was proof the model *had* rediscovered your `role_need` insight on its own, from prediction pressure alone.

**The honest result:** solo, the transformer scored **10.2 / 22.7 / 30.9 — it lost** to the GBM ensemble refit on multi-year data (14.9 / 32.8 / 45.2). Two reasons, both instructive:

- **Data scale.** Transformers earn their keep with millions of examples. You have 102,916 decisions. At this size, a well-fed GBM with good hand features is genuinely hard to beat — that's not a failure, that's the known trade-off.
- **Date-blindness.** The transformer sees champions, not calendars. Bans track *this week's* meta, which lives in the GBM's trailing-28-day features. Ban top-1: transformer 7.6 vs GBM 15.0. Brutal and diagnostic.

But split it by decision type and the transformer **owned picks** (the blend hit 17.2 pick top-1 vs GBM's 14.8) — because picks depend on draft context, which is exactly what attention is for.

Side finding worth remembering: just feeding the *old* v0.7 GBM three years of data instead of one was worth +1.9 top-1 by itself. "More data" beat most of your cleverness.

### v0.8.1 — stop averaging, start assigning

If the transformer owns picks and the GBM owns bans, why use one blend weight for both? Per-decision-type weights, chosen on validation only: **picks = 0.75 transformer, bans = 0.0 (pure GBM)**. Test: **16.0 / 33.1 / 44.6 — best overall top-1 of any lineage.**

Two honesty caveats you insisted on recording: this was the second and final look at the EWC test set (it's spent now), and the gain actually came from bans returning to the GBM — the fancier picks weighting didn't transfer better than the flat blend. Right answer, slightly humbling mechanism.

---

## Era 3: Change what "good" means (v0.9 rung 0)

Everything above optimizes **mimicry**: predict what pros *will* do. Your insight from last session — a coach doesn't want a mirror, they want an edge — demands a different objective: what *should* be picked to **win**.

But before building anything, the falsifiable gate: *does the draft even carry win signal beyond team strength?* Maybe better teams just win regardless of draft, and any "draft model" would secretly be a team-strength model.

The test (`experiment_v09_outcome_baseline.py`, ran yesterday): predict game winners with side + Elo team ratings alone, then add the completed draft, and compare held-out log-loss. Result: **it passed.** Log-loss improved 0.6144 → 0.6024, AUC 0.708 → 0.729, and the improvement held in 99.3% of bootstrap resamples (95% CI [+0.003, +0.021]). A team-agnostic version also passed, so it's not just "recognizing signature picks of good teams."

Translation: champion select measurably moves win probability, beyond who's playing. The coach-tool direction is alive. The next rungs (per-slot win probability, then a real outcome head on the transformer) are in the ROADMAP and the v0.9 handoff.

---

## The one-paragraph version to keep

The ladder is three eras of answering "how does the machine see the draft." **Era 1 (v0–v0.7):** you describe the world with hand-made features; the GBM finds thresholds; progress = your next idea; it ends when your ideas stop scaling (the pair-feature wall). **Era 2 (v0.8–v0.8.1):** the model learns its own description (embeddings + attention); it loses overall at your data size but wins exactly where representation matters (picks), so you assign labor instead of picking a winner. **Era 3 (v0.9):** you change the objective itself, from "predict the pros" to "predict the win" — gated by a cheap experiment that proved the signal exists.

And running underneath all nine rungs, the thing that makes any of it trustworthy: baselines first, temporal splits, every choice made on validation, one blind look at a frozen test set. The discipline is the ladder's rails — without it, each rung's number would just be a story you told yourself.
