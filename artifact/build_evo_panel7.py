"""ATTEMPT #7: the comic-strip idiom, a real team comp, and a new reader.

Three decisions from Sahil on 2026-08-01, each made from rendered pixels:

  * KEEP THE COMIC STRIP, but 2x2 rather than 1x4, with the animation BELOW the
    grid instead of hidden behind a disclosure. Prototype v1 (1x4) was measured
    at 552px on desktop against the gated version's 1127px -- but at 390px it
    stacked into a 1813px tower, 60% TALLER than what it replaced, and showed
    about one frame per screenful, destroying the "whole argument at once"
    property that justified the idiom. 2x2 stays 2x2 at 390px and dissolves that
    problem. His captions instinct was right: the per-frame notes are what make
    the strip readable, so they carry the argument.

  * MERGE THE COMP EXHIBIT INTO BEAT 2. Beat 2 already owned the rivals claim
    (Rakan + Alistar, 0 of 10,294), so restating it under beat 3 was redundancy.
    Beat 2 now opens on a real line-up -- Hanwha Life Esports, game 3 of the MSI
    2026 final -- and closes on the geometry that line-up implies. This
    restructures beat 2, which earlier handoffs forbade; that constraint was
    lifted by his explicit decision, not overridden.

  * RETUNE BEAT 3 FOR A NEW READER. The old cardinal rule was "assume the reader
    does not play League and does not know ML". He replaced it with: knows
    League, watches pro play, does NOT know ML. The job is to bridge what a
    viewer already reads intuitively in a televised draft to what the model
    encodes. Only beat 3's prose is retuned here, plus the new beat-2 material;
    beats 1 and 4 keep their existing wording pending his review of the voice.

Two claims he proposed were TESTED AND REJECTED, and appear nowhere on the page:
  * "the five champions land in five different role regions" -- only 3 of 5 do,
    and the best comp in the entire MSI final manages 4 of 5.
  * "the comp fans out as it trains" -- mean pairwise distance among the five
    goes 0.509 -> 0.429, i.e. slightly CLOSER together, not further apart.

Also corrected here: the Rakan-Alistar layout distance was carried as 0.0398 in
a code comment and in the handoff's "verified numbers" list. It was never
asserted anywhere and does not reproduce. The measured value is 0.0262, and it
is now regression-checked like every other number.

Lineage: v5 fixed the on-ramp and added the exceptions beat; v6 was a cleanup
pass over v5. Both remain reproducible from their own builders.

Four beats a reader steps through:
  1. One champion as a whole vector. All 192 of Rakan's numbers as a colored
     strip at start / midway / final -- NOT four labelled slot curves, which
     implied the slot index meant something and then admitted it didn't. The
     "you don't need to know what the slots mean" line now arrives BEFORE the
     visual, as a setup rather than a retraction (pattern from Jay Alammar's
     Illustrated Word2vec).
  2. Why two champions end up alike -- DEMONSTRATED, not asserted. A pro team
     needs one champion per role, so two supports are competing answers to the
     same question and are never teammates: Rakan and Alistar appear together in
     0 of 10,294 team comps, while Rakan and Orianna (support + mid) appear
     together 64 times against 64 expected by chance. Then three stacked vector
     strips (Rakan / Alistar / Orianna), then the similarity curves.
  3. All 168 at once: the animated "fly-in" map, unchanged mechanically, with
     the false global claim replaced by the local claim the picture supports and
     the honesty caption cut from a 9-line wall to two sentences.
  4. The exceptions are the payoff (new). A dot sits in the "wrong" place for
     exactly two measured reasons: the model barely saw that champion (purity by
     exposure quartile 0.443 / 0.624 / 0.914 / 0.833) or it genuinely plays more
     than one role (flex picks 0.453 vs 0.736). Corki is the traced
     counter-example to Rakan: 1,808 appearances and purity 0.20.

Every structural claim on the page is asserted against the data at build time --
see check_claims() below. A drifting claim fails the build instead of shipping.

Splices into pristine artifact-v2.html -> artifact-v9.html.
Also writes evo_panel7_test.html (page CSS + fragment, all beats reachable via
#b0../#b3 and #b2end) for headless screenshot checks.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

SCRATCH = Path(__file__).parent
REPO = Path.home() / "Documents/repos/lol-meta-tracker"
SRC = SCRATCH / "artifact-v2.html"
DST = SCRATCH / "artifact-v9.html"
TEST = SCRATCH / "evo_panel7_test.html"
TEST_DARK = SCRATCH / "evo_panel7_test_dark.html"
DRAFTS = REPO / "data/processed/draft_sequences_multi.parquet"

meta = json.loads(
    (REPO / "data/processed/embedding_evolution_v08_demo.json").read_text())
z = np.load(REPO / "data/processed/embedding_evolution_v08_snapshots.npz",
            allow_pickle=False)
embs = z["embs"]
champs = [str(c) for c in z["champs"]]
role = np.array([str(r) for r in z["role"]])
labels = [str(l) for l in z["labels"]]
purity = z["purity"].astype(float)
F, C, D = embs.shape

ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]
role_idx = np.array([ROLE_ORDER.index(r) for r in role], dtype=int)


def center_norm(e: np.ndarray) -> np.ndarray:
    e = e - e.mean(0)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


final_n = center_norm(embs[-1])

# ---- final layout: true t-SNE of the final snapshot ----
ts = TSNE(n_components=2, perplexity=18, random_state=42, init="pca",
          metric="cosine")
P_final = ts.fit_transform(final_n)
P_final -= P_final.mean(0)
P_final /= np.abs(P_final).max()

# ---- per-champion progress toward final state, per frame ----
align = np.stack([(center_norm(embs[t]) * final_n).sum(1) for t in range(F)])
a0, a1 = align[0], align[-1]
alignN = np.clip((align - a0) / np.maximum(a1 - a0, 1e-6), 0, 1)
# light smoothing (window 3) then a mild power so the final snap reads
k = np.array([0.25, 0.5, 0.25])
alignS = np.apply_along_axis(
    lambda v: np.convolve(np.pad(v, 1, mode="edge"), k, mode="valid"),
    0, alignN)
alignS[0], alignS[-1] = 0.0, 1.0
alignS = alignS ** 1.3

rng = np.random.default_rng(16)  # same seed as the retrain, for the story
cloud = rng.normal(0, 0.55, size=(C, 2))
cloud -= cloud.mean(0)
cloud *= 0.95 / np.abs(cloud).max()  # same box as the final layout — no zoom jump

pos = cloud[None] * (1 - alignS[:, :, None]) + P_final[None] * alignS[:, :, None]

# quantize per axis to 0..1000 (base64 uint16); keep the true aspect ratio
xlo, xhi = pos[:, :, 0].min(), pos[:, :, 0].max()
ylo, yhi = pos[:, :, 1].min(), pos[:, :, 1].max()
aspect = float((xhi - xlo) / (yhi - ylo))
q = np.empty_like(pos, dtype="<u2")
q[:, :, 0] = np.round((pos[:, :, 0] - xlo) / (xhi - xlo) * 1000)
q[:, :, 1] = np.round((pos[:, :, 1] - ylo) / (yhi - ylo) * 1000)
pos_b64 = base64.b64encode(q.tobytes()).decode()
print(f"positions: {len(pos_b64) / 1024:.0f}KB b64, aspect {aspect:.2f}")

# The exact unit-square coordinates the browser decodes and draws. Any distance
# quoted on the page MUST be measured here, not in P_final: the two axes are
# normalised independently above, so P_final distances do not survive the
# round-trip. (The 0.0398 Rakan-Alistar figure carried in earlier handoffs came
# from the wrong space and never reproduced; the drawn value is 0.0262.)
P_unit = q.astype(float) / 1000.0

# ---- Rakan exhibit ----
ri = champs.index("Rakan")
ai = champs.index("Alistar")
oi = champs.index("Orianna")

delta = np.abs(embs[-1, ri] - embs[0, ri])
top_dims = sorted(np.argsort(delta)[-4:].tolist())
dim_curves = [[round(float(embs[t, ri, d]), 3) for t in range(F)]
              for d in top_dims]

mid_f = F // 2
slot_rows = [np.round(embs[f, ri, :12] * 1000).astype(int).tolist()
             for f in (0, mid_f, F - 1)]
# "epoch 19" is jargon to a lay reader, and it invites the question "why 19?".
slot_row_names = ["random start", "partway through training", "finished"]


def cos_curve(a, b):
    return [round(float((center_norm(embs[t])[a] * center_norm(embs[t])[b]).sum()), 3)
            for t in range(F)]


sim_ali = cos_curve(ri, ai)
sim_ori = cos_curve(ri, oi)
print(f"Rakan-Alistar {sim_ali[0]:+.2f} -> {sim_ali[-1]:+.2f}; "
      f"Rakan-Orianna {sim_ori[0]:+.2f} -> {sim_ori[-1]:+.2f}; "
      f"purity {purity[0]:.3f} -> {purity[-1]:.3f}")

# ---- beat 1/2 exhibit: whole vectors as strips ------------------------------
# All 192 numbers at three moments. The reader is shown the WHOLE list, so the
# pattern across it is the message; no slot is labelled, because no slot means
# anything. Values are scaled by a shared robust limit so the three rows and the
# three champions in beat 2 are directly comparable.
STRIP_FRAMES = [0, F // 2, F - 1]
vec_lim = float(np.percentile(np.abs(embs[[0, F // 2, F - 1]][:, [ri, ai, oi]]), 98))


def strip(vals: np.ndarray) -> list:
    """Quantize a 192-vector to -100..100 for compact transport."""
    return np.clip(np.round(vals / vec_lim * 100), -100, 100).astype(int).tolist()


rakan_strips = [strip(embs[f, ri]) for f in STRIP_FRAMES]

# Beat 2 compares three champions, so all three rows share ONE column order:
# sorted by Rakan's final values. Unsorted, a +0.51 resemblance is spread across
# 192 squares and invisible; sorted, Rakan's row becomes a gradient and a similar
# champion visibly tracks it. The order itself means nothing and the caption says
# so. (Raw cosine still separates the pair we contrast: Rakan-Alistar +0.51,
# Rakan-Orianna -0.06, because the shared mean vector is small.)
col_order = np.argsort(embs[-1, ri])
trio_strips = [strip(embs[-1, c][col_order]) for c in (ri, ai, oi)]
_raw_cos = {}
for _nm, _j in (("ali", ai), ("ori", oi)):
    _u, _v = embs[-1, ri], embs[-1, _j]
    _raw_cos[_nm] = round(
        float(_u @ _v / np.linalg.norm(_u) / np.linalg.norm(_v)), 3)
print("raw cosine for the strip exhibit:", _raw_cos)
assert _raw_cos["ali"] - _raw_cos["ori"] > 0.3, (
    "the strips can no longer show the contrast they claim — "
    f"Rakan-Alistar {_raw_cos['ali']} vs Rakan-Orianna {_raw_cos['ori']}")

# ---- beat 2 exhibit: substitutes are never teammates -----------------------
# The mechanism, demonstrated instead of asserted. A team comp = one side of one
# game, one champion per role. Two supports are competing answers to the same
# question, so they never co-occur; a support and a mid are teammates at exactly
# the chance rate.
drafts = pd.read_parquet(DRAFTS)
picks_only = drafts[drafts.is_ban == 0].copy()
picks_only["side"] = picks_only.gameid + "_" + picks_only.is_blue.astype(str)
comps = [frozenset(s) for s in picks_only.groupby("side").champion.apply(set)]
n_comps = len(comps)


def together(a: str, b: str) -> int:
    return sum(1 for s in comps if a in s and b in s)


def appears(a: str) -> int:
    return sum(1 for s in comps if a in s)


def pair_row(a: str, b: str, note: str) -> dict:
    obs = together(a, b)
    exp = appears(a) * appears(b) / n_comps
    return {"a": a, "b": b, "note": note, "obs": obs, "exp": round(exp, 1)}


pairs = [
    pair_row("Rakan", "Alistar", "both play support"),
    pair_row("Rakan", "Orianna", "support + mid"),
]  # a third row (Rakan+Varus) made the identical point as Orianna and was cut
print("team-comp pairs:", [(p["a"], p["b"], p["obs"], p["exp"]) for p in pairs])

# ---- beat 2, NEW in #7: one real line-up the reader has actually watched -----
# The bridge Sahil asked for. A pro viewer already knows two things without being
# told: Xayah and Rakan get drafted together, and Rakan and Alistar never do.
# Both facts are in the record, and the model -- which was told neither -- encodes
# them as distance. That flip is the exhibit: the pair you would NEVER draft
# together is the pair the model stacks, and the actual lane partner sits far off.
COMP = ["Olaf", "Xin Zhao", "Ryze", "Xayah", "Rakan"]
COMP_GAME = {"team": "Hanwha Life Esports", "event": "MSI 2026 final",
             "game": "game 3", "date": "2026-07-12"}

_msi = drafts[(drafts.league == "MSI")
              & (drafts.date.dt.date == pd.Timestamp(COMP_GAME["date"]).date())
              & (drafts.is_ban == 0)]
_found = None
for (_gid, _blue), _g in _msi.groupby(["gameid", "is_blue"]):
    if set(_g.champion) == set(COMP):
        _found = (_gid, _g.team.iloc[0], int(_g.game_in_series.iloc[0]))
assert _found is not None, (
    "the highlighted comp must be a REAL line-up in the draft record")
assert _found[1] == COMP_GAME["team"], f"team drifted: {_found[1]}"
assert _found[2] == 3, f"expected game 3 of the series, got {_found[2]}"
assert sorted(role_idx[champs.index(c)] for c in COMP) == [0, 1, 2, 3, 4], (
    "the highlighted comp must field exactly one champion per role")
print(f"comp: {_found[1]}, game {_found[2]}, {_found[0]} — {', '.join(COMP)}")


def layout_dist(a: str, b: str) -> float:
    """Distance in the unit square the browser actually draws — see P_unit."""
    return float(np.linalg.norm(P_unit[-1][champs.index(a)]
                                - P_unit[-1][champs.index(b)]))


# Three rows spanning the whole range of what a draft can mean: partners,
# rivals, and no relationship at all.
comp_pairs = [
    {**pair_row("Rakan", "Xayah", "the lane duo"),
     "kind": "partners", "dist": round(layout_dist("Rakan", "Xayah"), 3)},
    {**pair_row("Rakan", "Alistar", "both want the support seat"),
     "kind": "rivals", "dist": round(layout_dist("Rakan", "Alistar"), 3)},
    {**pair_row("Rakan", "Orianna", "no particular link"),
     "kind": "none", "dist": round(layout_dist("Rakan", "Orianna"), 3)},
]
_by = {p["b"]: p for p in comp_pairs}
FAR_RATIO = _by["Xayah"]["dist"] / _by["Alistar"]["dist"]
comp_block = {
    "names": COMP,
    "idx": [champs.index(c) for c in COMP],
    "roles": [int(role_idx[champs.index(c)]) for c in COMP],
    "pairs": comp_pairs,
    "ratio": int(round(FAR_RATIO)),
    "payoffIdx": [champs.index(n) for n in ["Rakan", "Alistar", "Xayah", "Orianna"]],
    **COMP_GAME,
}
print(f"comp pairs: {[(p['b'], p['obs'], p['exp'], p['dist']) for p in comp_pairs]}")
print(f"partner sits {FAR_RATIO:.1f}x further than the rival")

# ---- beat 4: the two measured reasons a dot sits in the wrong place ---------
# Exposure = how many times a champion appears in the real draft record (picks +
# bans). Per-champion purity uses the SAME metric as the headline number:
# 5 nearest neighbours by Euclidean distance in the raw 192-dim space.
champ_counts = drafts.champion.value_counts()
exposure = np.array([float(champ_counts.get(c, 0.0)) for c in champs])
_d = ((embs[-1][:, None] - embs[-1][None]) ** 2).sum(-1)
np.fill_diagonal(_d, np.inf)
_nn = np.argsort(_d, axis=1)[:, :5]
champ_purity = (role_idx[_nn] == role_idx[:, None]).mean(1)
assert abs(champ_purity.mean() - purity[-1]) < 1e-3, (
    f"per-champion purity {champ_purity.mean():.4f} must reproduce the shipped "
    f"headline {purity[-1]:.4f} — otherwise beats 3 and 4 disagree")

q_edges = np.quantile(exposure, [0, 0.25, 0.5, 0.75, 1.0])
quartiles = []
for _i in range(4):
    _m = ((exposure >= q_edges[_i]) &
          ((exposure <= q_edges[_i + 1]) if _i == 3 else (exposure < q_edges[_i + 1])))
    # Display the ACTUAL min/max inside each quartile, not the shared quantile
    # edges — edges make consecutive labels overlap ("1–29", "29–334"), which
    # reads as an error.
    quartiles.append({
        "lo": int(exposure[_m].min()), "hi": int(exposure[_m].max()),
        "n": int(_m.sum()), "purity": round(float(champ_purity[_m].mean()), 3),
    })
print("exposure quartiles:", [(q["lo"], q["hi"], q["purity"]) for q in quartiles])

flex = z["flex"]
flex_stat = {
    "flexN": int(flex.sum()),
    "flexPurity": round(float(champ_purity[flex].mean()), 3),
    "soloN": int((~flex).sum()),
    "soloPurity": round(float(champ_purity[~flex].mean()), 3),
}
print("flex:", flex_stat)


def champ_card(name: str) -> dict:
    i = champs.index(name)
    return {"name": name, "exposure": int(exposure[i]),
            "purity": round(float(champ_purity[i]), 2),
            "flex": bool(flex[i]), "role": int(role_idx[i])}


cards = [champ_card("Rakan"), champ_card("Corki")]
print("beat-4 cards:", cards)

# ---- final-frame label subset: well-known champs, collision-checked ----
# Only label champions the reader has been given a reason to care about. Twelve
# names were noise to a non-player ("I know none of them"); these four are the
# ones the story actually builds up, and four sits inside the 3-5 annotation cap
# that practitioner guidance recommends.
WISH = ["Rakan", "Alistar", "Orianna", "Corki"]
label_idx = []
for name in WISH:
    if name not in champs:
        continue
    i = champs.index(name)
    # 0.16 dropped Alistar for sitting too close to Rakan — but that adjacency
    # IS the payoff of beat 2, so allow a tighter spacing here.
    if all(np.linalg.norm(P_final[i] - P_final[j]) > 0.085 for j in label_idx):
        label_idx.append(i)
print("final-frame labels:", [champs[i] for i in label_idx])

# ---- beat 3, NEW in #7: the four keyframes of the comic strip ---------------
# Chosen BY SCORE, never by eye: the first snapshot, the first to clear 0.30
# purity, the first to clear 0.50, and the last. If the training curve ever
# changes shape these move with it instead of going stale.
def _first_at(t: float) -> int:
    for _i, _p in enumerate(purity):
        if _p >= t:
            return _i
    return int(F) - 1


KEYS = [0, _first_at(0.30), _first_at(0.50), int(F) - 1]
assert len(set(KEYS)) == 4, f"keyframes must be distinct, got {KEYS}"
assert KEYS == sorted(KEYS), "keyframes must run forwards in time"
print("keyframes:", [(k, labels[k], round(purity[k], 3)) for k in KEYS])


def _layout_role_purity(f: int) -> dict:
    """Per-role purity measured in the 2-D LAYOUT, for asserting captions only.

    These numbers are deliberately never printed. The page shows exactly one
    purity number, the raw 192-dim score, and putting a second and different
    purity on the same figure would be worse than showing none.
    """
    Q = P_unit[f]
    d = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, 1e9)
    nn = np.argsort(d, 1)[:, :5]
    hit = (role_idx[nn] == role_idx[:, None]).mean(1)
    return {ROLE_ORDER[r]: float(hit[role_idx == r].mean()) for r in range(5)}


# One caption per frame. Each makes a claim, and each claim is checked below.
KEY_NOTES = [
    "Kick-off. Every champion is still 192 random numbers, so Hanwha&rsquo;s five sit "
    "wherever chance dropped them. The model has not read a single draft yet.",
    "The bot lane splits off first. Marksmen and supports are the earliest pair "
    "the model learns to tell apart; top and mid are still one blob.",
    "Most jobs now have a home region. Top is the last to separate, which is why "
    "Olaf is still adrift.",
    "Trained. Xayah and Rakan &mdash; the duo you see picked together every split &mdash; "
    "settle a long way apart, while Alistar ends up almost on top of Rakan.",
]

payload = {
    "names": champs,
    "role": role_idx.tolist(),
    "pos": pos_b64,
    "aspect": round(aspect, 3),
    "nF": int(F),
    "labels": labels,
    "purity": [round(p, 3) for p in purity],
    "rakan": {
        "i": ri,
        "dims": top_dims,
        "curves": dim_curves,
        "slots": slot_rows,
        "slotNames": slot_row_names,
        "strips": rakan_strips,
        "stripNames": slot_row_names,
    },
    "sim": {"ali": sim_ali, "ori": sim_ori},
    "labelIdx": label_idx,
    "aliI": ai,
    "trio": {"strips": trio_strips,
             "names": ["Rakan", "Alistar", "Orianna"],
             "roles": [int(role_idx[ri]), int(role_idx[ai]), int(role_idx[oi])]},
    "pairs": pairs,
    "nComps": n_comps,
    "quart": quartiles,
    "flexStat": flex_stat,
    "cards": cards,
    "comp": comp_block,
    "keys": KEYS,
    "keyNotes": KEY_NOTES,
}
data_js = json.dumps(payload, separators=(",", ":"))
assert "</" not in data_js
print("payload total:", len(data_js) // 1024, "KB")

FRAGMENT = r"""
<div class="wrap">
  <section>
    <div class="stepno">Step 9 &middot; in motion</div>
    <h2>Watch the map assemble itself</h2>
    <p>The map above is the finish line. This is the race. The model starts with 192 random numbers per champion. Then it reads the drafts of @@NTRAIN@@ real pro games, and after every wrong guess it nudges every number a tiny step. Nobody ever tells it what a role is.</p>
    <p>To film the race, we retrained one copy of the model on the same data and saved every number 56 times along the way. The story below has four beats: one champion, then why two champions end up alike, then all 168 at once, then the champions that break the pattern.</p>
  </section>
</div>

<div class="panelwrap">
  <div class="panel" id="evo4-panel" style="position:relative">
    <h3>How role clusters form, in four beats</h3>
    <p class="sub">Step through with the buttons. Every number shown is real, from the saved snapshots and the draft record.</p>

    <div class="evo4-steps" role="tablist" aria-label="Story beats">
      <button type="button" id="evo4-t0" role="tab" aria-selected="true">1 &middot; A champion is a list of numbers</button>
      <button type="button" id="evo4-t1" role="tab" aria-selected="false">2 &middot; Why two champions end up alike</button>
      <button type="button" id="evo4-t2" role="tab" aria-selected="false">3 &middot; All 168 at once</button>
      <button type="button" id="evo4-t3" role="tab" aria-selected="false">4 &middot; The exceptions</button>
    </div>
    <div class="evo4-navrow" id="evo4-navrow">
      <button type="button" id="evo4-next" class="evo4-nav">Next &rarr;</button>
    </div>

    <!-- Beat 1 -->
    <div class="evo4-beat" id="evo4-b0" role="tabpanel">
      <p class="evo4-lead"><strong>The model has exactly one job.</strong> Before a match, the two teams take turns choosing characters. That is the draft, and the characters are called champions. There are 168 of them. Show the model the choices made so far, and it guesses which champion gets taken next. That is the only thing it was ever trained to do.</p>
      <p class="evo4-lead"><strong>To do that job, it keeps 192 numbers per champion.</strong> Here is Rakan, one of the 168. To the model he is nothing but those numbers &mdash; no image, no lore, not even a name. We will not tell you what any single number means. Nobody knows, and you do not need to know. Watch the shape of the whole list instead. Each of the 192 numbers is one bar, above or below the centre line:</p>
      <div id="evo4-strips0"></div>
      <p class="codecap">The list starts as random noise. Training nudges it millions of times, and it stops looking random. That shape is the model&rsquo;s entire idea of Rakan. So the next question is why any <em>other</em> champion would end up with a similar one.</p>
      <details class="evo4-more">
        <summary>Show me the actual digits</summary>
        <p class="sub">The first 12 of Rakan&rsquo;s 192 numbers, in thousandths. The column positions carry no meaning. They are just the order the numbers happen to sit in.</p>
        <div class="evo4-numwrap"><table class="evo4-num" id="evo4-num"></table></div>
      </details>
    </div>

    <!-- Beat 2 -->
    <div class="evo4-beat" id="evo4-b1" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>Start with a draft you have already watched.</strong> @@CTEAM@@, @@CGAME@@ of the @@CEVENT@@: @@CLIST@@. Nothing here was chosen to flatter the model &mdash; it is one real line-up out of the @@NCOMPS@@ in the record, and it is the comp you will see lit up on the map in the next beat.</p>
      <p class="evo4-lead"><strong>Champions that compete for the same job end up with the same numbers.</strong> Think of building a team. A pro team fields exactly five champions, one for each role. So if you have already chosen Rakan, you will never also choose Alistar. They want the same seat &mdash; rivals for one slot, not partners. Here is that pattern in @@NCOMPS@@ real team line-ups.</p>
      <div id="evo4-pairs" class="evo4-pairs"></div>
      <p class="codecap">Rakan and Alistar are alternatives to each other, so they never appear together. Rakan and Orianna do different jobs, so they turn up together about as often as pure chance would predict. The model is only ever asked one question: who gets picked next? Rakan and Alistar keep being the answer in the same spots, and every nudge that pushes one of them pushes the other the same way.</p>
      <p class="sub">So their lists of 192 numbers drift toward the same shape. To make that easy to see, all three rows below are sorted in one shared order &mdash; by Rakan&rsquo;s values, smallest to largest. The order itself means nothing. It is only a way to line the rows up. The dark line on each row is a running average, so you are comparing shapes rather than individual bars.</p>
      <div id="evo4-strips1"></div>
      <p class="codecap">Rakan&rsquo;s line climbs by construction, because the sort put his numbers in order. The question is what the other two do under that same order. Alistar&rsquo;s line climbs with it. Orianna&rsquo;s stays flat &mdash; her numbers have nothing to do with Rakan&rsquo;s.</p>
      <p class="sub">Measured properly, that resemblance is one number per pair. It runs from &minus;1 (opposite shapes) through 0 (nothing in common) to +1 (identical), and it uses all 192 numbers at once. Here it is across the whole of training:</p>
      <canvas id="evo4-c1" style="width:100%;display:block" role="img" aria-label="Line chart: similarity between Rakan and Alistar rises from near zero to 0.56 over training, while Rakan and Orianna stays near zero"></canvas>

      <p class="sub"><b>Now put that next to what the draft record actually did.</b> Two of Hanwha&rsquo;s five picks are Xayah and Rakan, and you already know how that pairing behaves: they get taken together. You also know Rakan and Alistar never do. Here are both, counted, against where the model ended up putting them.</p>
      <canvas id="evo4-pay" style="width:100%;display:block" role="img" aria-label="The finished layout with only Rakan, Alistar, Xayah and Orianna highlighted. Rakan and Alistar sit almost on the same point; Xayah sits far away, joined to Rakan by a dashed line."></canvas>
      <p class="codecap">The finished layout again, with only these four lit. Rakan and Alistar are the two dots you cannot tell apart.</p>
      <table class="evo4-ctbl" id="evo4-ctbl"></table>
      <p class="evo4-close"><b>That is the flip worth sitting with.</b> Rakan and Alistar are never drafted together &mdash; @@AT@@ times against an expected @@AE@@ &mdash; and the model puts them practically on the same point. Xayah, who actually plays beside Rakan @@XT@@ times, ends up <b>@@CRATIO@@&times; further away</b>. Champions that compete for the same seat get the same numbers. Champions that play beside each other do not have to.</p>

      <p class="codecap">Nobody ever told the model that Rakan and Alistar play the same role. It only read drafts. The same pull acted on every pair of champions at the same time &mdash; that is the next beat.</p>
    </div>

    <!-- Beat 3 -->
    <div class="evo4-beat" id="evo4-b2" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>All 168 champions, sorting themselves by role &mdash; and nobody ever told the model what a role is.</strong> Every dot is a champion, coloured by the position it actually gets played in. The model never saw those colours. The score is the share of a champion&rsquo;s five nearest neighbours that share its role: guessing gets you 0.20, and this run finishes at @@PF@@.</p>
      <p class="sub">Hanwha&rsquo;s five are still lit, so you can follow a comp you know through the whole run. Four snapshots, picked by score rather than by eye &mdash; the start, the first to clear 0.30, the first to clear 0.50, and the end.</p>
      <div class="evo4-keys" id="evo4-keys"></div>
      <div class="evo4-legend" id="evo4-legend" aria-hidden="true"></div>
      <p class="codecap"><b>Hanwha&rsquo;s five are the larger ringed dots; the other @@NOTHER@@ champions are drawn smaller in the same role colours.</b> When each dot settles is measured. Where it travels between snapshots is drawn. Read the timing, not the path.</p>

      <div class="evo4-animwrap">
        <div class="evo4-animhd">The same run moving, if you would rather watch it</div>
        <canvas id="evo4-map" style="width:100%;display:block" role="img" aria-label="Animated scatter map: 168 champion dots fly from a random cloud into five role clusters as training progresses, with Hanwha Life's five picks highlighted"></canvas>
        <div class="evo4-controls">
          <button id="evo4-play" type="button">&#9654; Play</button>
          <input id="evo4-scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Training snapshot">
        </div>
        <div class="evo4-readout" id="evo4-label"></div>
      </div>

      <canvas id="evo4-strip" style="width:100%;display:block" role="img" aria-label="Line chart: role purity across the 56 saved snapshots, rising from @@P0@@ to @@PF@@ against a chance line at 0.20. A playhead marks the snapshot shown on the map above."></canvas>
      <p class="codecap">The finished layout is real: a t-SNE of the last snapshot. The score is measured in the raw 192 numbers at every step. In between, each dot slides straight toward its final spot, paced by its own measured progress.</p>
      <p class="codecap">Parts of this map are tidy and parts are a jumble. Beat 4 is about who lands in the jumble, and why &mdash; Olaf, above, is already a hint.</p>
      <p class="codecap evo4-prov">Demo run: seed 16, config d192x4L6H, same 2024&ndash;2026 data and split as the shipped model (@@NTRAIN@@ training games), score @@P0@@ &rarr; @@PF@@.</p>
    </div>

    <!-- Beat 4 -->
    <div class="evo4-beat" id="evo4-b3" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>The champions in the wrong place explain the most.</strong> The finished map is not five tidy clusters. That is not sloppiness in the model. A dot lands away from its own kind for exactly two reasons, and we can measure both.</p>
      <p class="sub"><b>Reason one: the model hardly ever saw it.</b> Sort all 168 champions by how often they appear in the draft record, then measure how well each quarter sorted itself by role. The dotted line marks 0.20 &mdash; what you would get from pure chance:</p>
      <canvas id="evo4-c3" style="width:100%;display:block" role="img" aria-label="Bar chart: role purity by exposure quartile, rising from 0.44 for the least-seen champions to 0.91, then dipping to 0.83 for the most-seen"></canvas>
      <p class="codecap">Champions pros rarely draft give the model almost nothing to learn from, so their numbers stay close to the random noise they started as. Evidence is the raw material here: no drafts, no pattern.</p>
      <p class="sub"><b>Reason two: it genuinely does two jobs.</b> That dip in the last quarter is not noise. Some champions get played in more than one role, so there is no single right place to put them:</p>
      <div id="evo4-flex" class="evo4-flexstat"></div>
      <div id="evo4-cards" class="evo4-cards"></div>
      <p class="codecap">Corki is drafted about as often as Rakan, and the model still cannot pin him down. Pros play him in more than one role. That is not the model failing. That is the model correctly reporting that the question has no single answer.</p>
      <p class="evo4-close">So the jumble in beat 3 is not a mess to look past. It is the map naming the champions the pros themselves refuse to file under one role &mdash; and the ones they almost never draft. Read the tidy parts to see what the model learned. Read the messy parts to see what the drafts never told it.</p>
      <p class="codecap evo4-prov">Purity here is the same measure used on the map: the share of a champion&rsquo;s five nearest neighbours that share its role.</p>
    </div>

    <div id="evo4-tip" role="status"></div>
  </div>
</div>

<style>
  #evo4-panel > h3 { font-size: 21px; line-height: 1.25; margin-bottom: 6px; }

  /* Tab row: a 4-up grid so the beats read as four equal steps, 2x2 on mobile.
     "Next" sits on its own right-aligned line instead of being stranded in the
     wrap gutter — and it is no longer a non-tab child of the tablist. */
  .evo4-steps { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin:2px 0 0; }
  .evo4-steps button {
    font: 600 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); background: transparent;
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 8px 10px; cursor: pointer; text-align: left;
  }
  .evo4-steps button[aria-selected="true"] { color: var(--ink); border-color: var(--model); }
  .evo4-navrow { display:flex; justify-content:flex-end; margin: 10px 0 0; }
  .evo4-navrow[hidden] { display:none; }
  .evo4-navrow button.evo4-nav {
    font: 600 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--card); background: var(--ink); border: 1px solid var(--ink);
    border-radius: 7px; padding: 8px 14px; cursor: pointer;
  }
  @media (max-width: 640px) { .evo4-steps { grid-template-columns: repeat(2, 1fr); } }
  .evo4-steps button:focus-visible,
  .evo4-navrow button:focus-visible,
  .evo4-controls button:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
  .evo4-lead { font-size: 15px; max-width: 68ch; }

  /* ---- one spacing scale: 16px between prose blocks, 32px around figures ----
     Prefixed with #evo4-panel on purpose: the page's own `.panel .sub` rule is
     (0,2,0) and would otherwise win over a bare `.evo4-beat > *`, silently
     reinstating the old ragged 17/46/33/27px gaps. */
  #evo4-panel .evo4-beat { margin-top: 24px; }
  #evo4-panel .evo4-beat > * { margin-top: 0; margin-bottom: 0; }
  #evo4-panel .evo4-beat > * + * { margin-top: 16px; }
  #evo4-panel .evo4-beat > * + canvas,
  #evo4-panel .evo4-beat > * + div,
  #evo4-panel .evo4-beat > canvas + *,
  #evo4-panel .evo4-beat > div + * { margin-top: 32px; }
  #evo4-panel .evo4-beat > canvas + .evo4-controls { margin-top: 10px; }
  #evo4-panel .evo4-beat > .evo4-legend + canvas { margin-top: 8px; }
  #evo4-panel .evo4-beat > .evo4-controls + canvas { margin-top: 12px; }
  #evo4-panel .evo4-beat .sub,
  #evo4-panel .evo4-beat .codecap { max-width: 68ch; }
  #evo4-panel .evo4-close { font-size: 15px; max-width: 68ch; }
  .evo4-legend { display:flex; gap:14px; flex-wrap:wrap; margin: 2px 0 8px; }
  .evo4-legend span {
    font: 600 11.5px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); display:inline-flex; align-items:center; gap:5px;
  }
  .evo4-legend i { width:9px; height:9px; border-radius:50%; display:inline-block; }

  /* Beat 3's keyframe strip. 2x2 at EVERY width, deliberately: the 1x4
     prototype measured 552px on desktop but stacked to a 1813px single column
     at 390px — taller than the animation it replaced, and about one frame per
     screenful, which destroys the "whole argument at once" property that
     justified the idiom. 2x2 holds its shape all the way down. */
  .evo4-keys { display:grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }
  @media (max-width: 430px) { .evo4-keys { gap: 12px; } }
  .evo4-keycell { display:flex; flex-direction:column; gap:7px; }
  .evo4-keycell canvas {
    width:100%; display:block;
    border:1px solid var(--hairline); border-radius:8px; background: var(--inset);
  }
  .evo4-keyhd {
    font: 600 11.5px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); display:flex; justify-content:space-between; gap:8px;
    font-variant-numeric: tabular-nums;
  }
  .evo4-keyhd b { color: var(--ink); }
  .evo4-keynote {
    font: 500 12.5px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); margin: 0;
  }
  .evo4-animhd {
    font: 600 12px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); margin: 0 0 10px;
  }
  .evo4-readout {
    font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); margin-top: 8px;
  }

  /* Beat 2's counted table. */
  .evo4-ctbl { width:100%; border-collapse: collapse; }
  .evo4-ctbl th, .evo4-ctbl td {
    text-align:left; padding: 8px 10px 8px 0;
    border-bottom: 1px solid var(--hairline);
    font: 500 12.5px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted);
  }
  .evo4-ctbl th { font-weight:700; color: var(--ink); }
  .evo4-ctbl td b { color: var(--ink); }
  .evo4-ctbl .n {
    text-align:right; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; white-space:nowrap;
  }
  /* Below 560px four columns do not fit and the last one — the map distance,
     which is the entire point — ran off the panel. Each row becomes a block
     with its header carried inline. */
  @media (max-width: 560px) {
    .evo4-ctbl, .evo4-ctbl tbody, .evo4-ctbl tr, .evo4-ctbl td { display:block; width:100%; }
    .evo4-ctbl thead { position:absolute; width:1px; height:1px;
      overflow:hidden; clip: rect(0 0 0 0); white-space:nowrap; }
    .evo4-ctbl tr { padding: 10px 0; border-bottom: 1px solid var(--hairline); }
    .evo4-ctbl td { border:0; padding: 2px 0; }
    .evo4-ctbl td:first-child { padding-bottom: 6px; }
    .evo4-ctbl td.n { display:flex; justify-content:space-between; gap:12px; }
    .evo4-ctbl td.n::before {
      content: attr(data-l); font-family: system-ui, -apple-system, sans-serif;
      font-weight:600; color: var(--muted);
    }
  }

  .evo4-controls { display:flex; align-items:center; gap:12px; margin:10px 0 4px; }
  .evo4-controls button {
    font: 600 13px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 12px; cursor: pointer; white-space: nowrap;
  }
    /* neutral, not --model: that blue means 'Top lane' in the legend above */
  .evo4-controls input[type=range] { flex:1; accent-color: var(--muted); min-width:0; }
  /* In #7 the readout moved out of the flex controls row onto its own line, so
     the old right-alignment and 19ch reservation are wrong — and being an ID
     rule it silently beat the new .evo4-readout class. Left-aligned everywhere
     now. (v6 hid this below 560px; the caption tells the reader to read the
     timing, and the readout IS the timing, so it stays visible at every width.) */
  #evo4-label {
    font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); text-align: left;
  }
  #evo4-tip {
    position:absolute; display:none; pointer-events:none;
    font: 600 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
    font-variant-numeric: tabular-nums;
    background: var(--card); color: var(--ink);
    border:1px solid var(--hairline); border-radius:6px;
    padding:5px 8px; box-shadow: var(--shadow); z-index:5; white-space:nowrap;
  }
  .evo4-numwrap { overflow-x:auto; }
  table.evo4-num {
    border-collapse: collapse; width:100%; min-width:640px;
    font: 500 11.5px/1.2 ui-monospace, "SF Mono", Menlo, monospace;
  }
  table.evo4-num th {
    text-align:left; color: var(--muted); font-weight:600;
    padding:4px 8px 4px 0; white-space:nowrap;
  }
  table.evo4-num td {
    text-align:right; padding:4px 5px; min-width:44px;
    border-top:1px solid var(--hairline);
    font-variant-numeric: tabular-nums; white-space:nowrap;
  }
  @media (max-width: 560px) {
    .evo4-controls { flex-wrap: wrap; }
  }

  /* ---- v5: vector strips, team-comp pairs, beat-4 exhibits ---- */
  .evo4-sw {
    display:inline-block; width:10px; height:10px; border-radius:2px;
    vertical-align:-1px; margin:0 1px;
  }
  .evo4-sw-pos { background: var(--meta); }
  .evo4-sw-neg { background: var(--model); }
  .evo4-striprow { margin: 0 0 10px; }
  .evo4-striprow .evo4-striplab {
    font: 600 11.5px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    font-variant-numeric: tabular-nums;
    color: var(--muted); display:flex; justify-content:space-between; gap:10px;
  }
  .evo4-striprow canvas { width:100%; display:block; }
  .evo4-more { margin-top: 14px; }
  .evo4-more summary {
    font: 600 12px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); cursor: pointer; padding: 6px 0;
  }
  .evo4-more summary:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
  .evo4-prov { color: var(--faint); }
  .evo4-pairs { display:flex; flex-direction:column; gap:8px; margin: 4px 0 2px; }
  .evo4-pairrow {
    display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;
    padding: 9px 12px; border:1px solid var(--hairline); border-radius:8px;
    background: var(--inset);
    font: 500 12.5px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    font-variant-numeric: tabular-nums;
  }
  .evo4-pairrow b { color: var(--ink); font-weight:700; }
  .evo4-pairrow .evo4-pairnote { color: var(--faint); }
  .evo4-pairrow .evo4-obs { margin-left:auto; color: var(--muted); white-space:nowrap; }
  .evo4-pairrow .evo4-zero { color: var(--meta); font-weight:700; }
  /* Below ~560px the observation cannot share a line with the pair name. With
     white-space:nowrap and margin-left:auto it overflowed the card and was
     clipped mid-word, hiding the expected-by-chance figures — which are the
     entire point of the comparison. Force it onto its own full-width line —
     switching the row to column instead would break "Rakan + Alistar" across
     three lines. */
  @media (max-width: 560px) {
    .evo4-pairrow { row-gap: 4px; }
    .evo4-pairrow .evo4-obs {
      margin-left: 0; white-space: normal; flex: 1 0 100%;
    }
  }
  /* ---- beat 4: ONE stat-card template for all four boxes ----
     Same grid, same padding, same 1px hairline border, same three-line
     internal structure (heading / number / caption). Identity is carried by
     the heading colour, never by a heavier border. No inner boxes. */
  .evo4-flexstat,
  .evo4-cards { display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; }
  @media (max-width: 560px) {
    .evo4-flexstat,
    .evo4-cards { grid-template-columns: 1fr; }
  }
  .evo4-stat {
    padding:12px 14px; border:1px solid var(--hairline);
    border-radius:8px; background: var(--inset);
  }
  .evo4-stat-head {
    font: 600 12.5px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
  }
  .evo4-stat-big {
    font: 700 24px/1.15 ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; color: var(--ink); margin: 4px 0 2px;
  }
  .evo4-stat-cap {
    font: 500 11.5px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); font-variant-numeric: tabular-nums;
  }
</style>

<script>
(function () {
  "use strict";
  var D = @@DATA@@;
  var ROLE_NAMES = ["Top", "Jungle", "Mid", "Bot", "Support"];
  var ROLE_LIGHT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#4a3aa7"];
  var ROLE_DARK  = ["#3987e5", "#008300", "#d55181", "#c98500", "#9085e9"];
  var nF = D.nF, N = D.names.length;

  // decode positions: uint16 little-endian, (frame, champ, xy), 0..1000
  var raw = atob(D.pos);
  var pos = new Uint16Array(raw.length / 2);
  for (var bi = 0; bi < pos.length; bi++) {
    pos[bi] = raw.charCodeAt(2 * bi) | (raw.charCodeAt(2 * bi + 1) << 8);
  }
  function px(f, c) { return pos[(f * N + c) * 2] / 1000; }
  function py(f, c) { return pos[(f * N + c) * 2 + 1] / 1000; }

  var panel = document.getElementById("evo4-panel");
  var tip = document.getElementById("evo4-tip");
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var darkMq = window.matchMedia("(prefers-color-scheme: dark)");
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return darkMq.matches;
  }
  function roleColors() { return isDark() ? ROLE_DARK : ROLE_LIGHT; }
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function sizeCanvas(cv, hCss) {
    var w = cv.clientWidth || cv.parentNode.clientWidth || 600;
    var dpr = window.devicePixelRatio || 1;
    cv.style.height = hCss + "px";
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(hCss * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(hCss * dpr);
    }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /* ---------- beat switching ---------- */
  var NBEATS = 4;
  var beat = 0;
  var idx4 = [0, 1, 2, 3];
  var tabs = idx4.map(function (i) { return document.getElementById("evo4-t" + i); });
  var beats = idx4.map(function (i) { return document.getElementById("evo4-b" + i); });
  var nextBtn = document.getElementById("evo4-next");
  var navRow = document.getElementById("evo4-navrow");
  function setBeat(b) {
    beat = b;
    tabs.forEach(function (t, i) { t.setAttribute("aria-selected", String(i === b)); });
    beats.forEach(function (el, i) { el.hidden = (i !== b); });
    // hide the whole row, not just the button: an empty flex row still carries
    // its own top margin and would leave a phantom gap on the last beat.
    navRow.hidden = (b === NBEATS - 1);
    tip.style.display = "none";
    renderBeat();
    if (b === 2 && !reduced && !mapPlayedOnce) { mapPlayedOnce = true; startPlay(); }
  }
  tabs.forEach(function (t, i) { t.addEventListener("click", function () { setBeat(i); }); });
  nextBtn.addEventListener("click", function () { setBeat(Math.min(beat + 1, NBEATS - 1)); });

  /* ---------- shared line-chart helper (beats 1 & 2) ---------- */
  function drawLines(cv, series, opts) {
    var H = opts.height, padL = 44, padR = opts.padR, padT = 16, padB = 26;
    var ctx = sizeCanvas(cv, H);
    var w = cv.clientWidth, iw = w - padL - padR, ih = H - padT - padB;
    ctx.clearRect(0, 0, w, H);
    var mut = css("--muted"), faint = css("--faint"), hair = css("--hairline");
    var lo = opts.lo, hi = opts.hi;
    function X(i) { return padL + iw * i / (nF - 1); }
    function Y(v) { return padT + ih - ih * (v - lo) / (hi - lo); }
    // gridlines + y ticks
    ctx.font = "500 10.5px ui-monospace, Menlo, monospace";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    opts.ticks.filter(function (tv) { return tv >= lo && tv <= hi; })
        .forEach(function (tv) {
      ctx.strokeStyle = tv === 0 ? faint : hair; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, Y(tv)); ctx.lineTo(padL + iw, Y(tv)); ctx.stroke();
      ctx.fillStyle = mut;
      ctx.fillText((tv > 0 ? "+" : "") + tv.toFixed(1), padL - 6, Y(tv));
    });
    // series
    series.forEach(function (s) {
      ctx.beginPath();
      for (var i = 0; i < nF; i++) {
        i ? ctx.lineTo(X(i), Y(s.v[i])) : ctx.moveTo(X(i), Y(s.v[i]));
      }
      ctx.lineWidth = 2; ctx.strokeStyle = s.color; ctx.stroke();
      ctx.beginPath();
      ctx.arc(X(nF - 1), Y(s.v[nF - 1]), 3.2, 0, 7); ctx.fillStyle = s.color; ctx.fill();
    });
    // Direct labels at line ends, nudged apart. The text wears an INK token, not
    // the series colour: at --meta pink the label was 2.6:1 on the light card,
    // which fails as text even though it passes as a mark. The filled dot at the
    // end of each line, drawn just above, already carries the identity.
    ctx.textAlign = "left";
    ctx.font = '600 11px system-ui, -apple-system, "Segoe UI", sans-serif';
    var used = [];
    series.forEach(function (s) {
      var ly = Y(s.v[nF - 1]);
      used.sort(function (a, b) { return a - b; });
      for (var u = 0; u < used.length; u++) {
        if (Math.abs(used[u] - ly) < 13) { ly = used[u] + 13; }
      }
      used.push(ly);
      ctx.fillStyle = css("--ink");
      ctx.fillText(s.label, padL + iw + 7, ly);
    });
    // x axis caption (skip the left one when narrow — they collide)
    ctx.font = '500 10.5px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillStyle = faint; ctx.textBaseline = "alphabetic";
    if (w >= 520) {
      ctx.textAlign = "left";
      ctx.fillText("random start", padL, H - 8);
    }
    ctx.textAlign = "right";
    ctx.fillText("training time → final", padL + iw, H - 8);
    return { X: X, Y: Y, padL: padL, iw: iw, padT: padT, ih: ih };
  }

  function hoverLines(cv, series, geoRef, fmt) {
    cv.addEventListener("mousemove", function (e) {
      var g = geoRef();
      if (!g) return;
      var r = cv.getBoundingClientRect();
      var fx = (e.clientX - r.left - g.padL) / g.iw * (nF - 1);
      var i = Math.round(fx);
      if (i < 0 || i > nF - 1 || e.clientY - r.top > g.padT + g.ih + 10) {
        tip.style.display = "none"; return;
      }
      tip.innerHTML = fmt(i, series);
      tip.style.display = "block";
      var tx = Math.min(e.clientX - r.left + 14, r.width - 180);
      tip.style.left = tx + "px";
      tip.style.top = (cv.offsetTop + (e.clientY - r.top) - 34) + "px";
    });
    cv.addEventListener("mouseleave", function () { tip.style.display = "none"; });
  }

  /* ---------- beats 1 & 2: whole-vector strips ----------
     All 192 numbers as one row of squares. The reader is never asked to read an
     individual square: the message is the pattern across the row, and (in beat 2)
     the match between rows. Diverging scale, theme-aware, values pre-quantized to
     -100..100 in Python against a shared limit so rows are comparable. */
  function stripRow(host, label, right, values, accent) {
    var wrap = document.createElement("div");
    wrap.className = "evo4-striprow";
    var lab = document.createElement("div");
    lab.className = "evo4-striplab";
    var l = document.createElement("span");
    l.textContent = label;
    if (accent) { l.style.color = accent; l.style.fontWeight = "700"; }
    var r = document.createElement("span");
    r.textContent = right || "";
    lab.appendChild(l); lab.appendChild(r);
    var cv = document.createElement("canvas");
    cv.setAttribute("role", "img");
    cv.setAttribute("aria-label",
      label + ": all 192 learned values, shown as a colour strip");
    wrap.appendChild(lab); wrap.appendChild(cv);
    host.appendChild(wrap);
    return { cv: cv, values: values };
  }

  /* Sign is encoded by DIRECTION (up/down from a centre line), not by hue.
     Two reasons. (1) A naive reader could not see the resemblance between two
     champions in the earlier colour-alpha version -- "Alistar's row is
     indistinguishable from Orianna's" -- and a visual the reader cannot verify
     costs more trust than no visual at all. (2) The colour version reused
     blue/orange, which already mean Top lane and Bot lane on the map in beat 3;
     one hue pair cannot carry two meanings in one piece. Neutral ink here means
     the strips collide with nothing. `trend` overlays a moving average so the
     shared shape is perceptible rather than merely present. */
  function paintStrip(cv, values, trend) {
    var H = 46;
    var ctx = sizeCanvas(cv, H);
    var w = cv.clientWidth;
    ctx.clearRect(0, 0, w, H);
    var mut = css("--muted"), ink = css("--ink"), hair = css("--hairline");
    var n = values.length, cw = w / n, mid = H / 2, amp = (H - 8) / 2;
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid + 0.5); ctx.lineTo(w, mid + 0.5); ctx.stroke();
    ctx.fillStyle = mut;
    for (var i = 0; i < n; i++) {
      var v = Math.max(-1, Math.min(1, values[i] / 100));
      var h = Math.abs(v) * amp;
      ctx.fillRect(i * cw, v >= 0 ? mid - h : mid, Math.max(cw - 0.4, 0.6), h);
    }
    if (trend) {
      var win = 15, half = (win - 1) / 2;
      ctx.beginPath();
      for (var j = 0; j < n; j++) {
        var s = 0, c = 0;
        for (var k = Math.max(0, j - half); k <= Math.min(n - 1, j + half); k++) {
          s += values[k]; c++;
        }
        var y = mid - Math.max(-1, Math.min(1, (s / c) / 100)) * amp;
        j ? ctx.lineTo(j * cw + cw / 2, y) : ctx.moveTo(j * cw + cw / 2, y);
      }
      ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.stroke();
    }
  }

  var strips0 = [], strips1 = [];
  (function buildStrips() {
    var h0 = document.getElementById("evo4-strips0");
    D.rakan.strips.forEach(function (vals, i) {
      strips0.push(stripRow(h0, D.rakan.stripNames[i],
                            i === 0 ? "192 numbers →" : "", vals, null));
    });
    var h1 = document.getElementById("evo4-strips1");
    var rc = roleColors();
    D.trio.strips.forEach(function (vals, i) {
      strips1.push(stripRow(h1, D.trio.names[i],
                            i === 0 ? "192 numbers →" : "", vals,
                            rc[D.trio.roles[i]]));
    });
  })();

  function drawBeat1() {
    strips0.forEach(function (s) { paintStrip(s.cv, s.values, false); });
  }
  function paintTrio() {
    var rc = roleColors();
    strips1.forEach(function (s, i) {
      paintStrip(s.cv, s.values, true);   // trend line: the whole point of beat 2
      var lab = s.cv.parentNode.querySelector(".evo4-striplab span");
      if (lab) lab.style.color = rc[D.trio.roles[i]];
    });
  }

  /* ---------- beat 2: the team-comp evidence ---------- */
  (function buildPairs() {
    var host = document.getElementById("evo4-pairs");
    D.pairs.forEach(function (p) {
      var row = document.createElement("div");
      row.className = "evo4-pairrow";
      var zero = p.obs === 0;
      row.innerHTML =
        "<b>" + p.a + "</b> + <b>" + p.b + "</b>" +
        " <span class='evo4-pairnote'>" + p.note + "</span>" +
        "<span class='evo4-obs'>together in <span class='" +
        (zero ? "evo4-zero" : "") + "'>" + p.obs.toLocaleString() +
        "</span> of " + D.nComps.toLocaleString() + " line-ups · expected by chance: " + p.exp + " line-ups</span>";
      host.appendChild(row);
    });
  })();

  (function buildTable() {
    var t = document.getElementById("evo4-num");
    var rows = "<tr><th></th>";
    for (var d = 0; d < 12; d++) rows += "<th style='text-align:right'>s" + (d + 1) + "</th>";
    rows += "<th style='text-align:left;color:var(--faint)'>&hellip;+180</th></tr>";
    D.rakan.slots.forEach(function (vals, r) {
      rows += "<tr><th>" + D.rakan.slotNames[r] + "</th>";
      vals.forEach(function (v) {
        var mag = Math.min(Math.abs(v) / 260, 1);
        var bg = v >= 0
          ? "rgba(196,118,59," + (0.06 + 0.3 * mag).toFixed(2) + ")"
          : "rgba(62,124,184," + (0.06 + 0.3 * mag).toFixed(2) + ")";
        rows += "<td style='background:" + bg + "'>" + v + "</td>";
      });
      rows += "<td></td></tr>";
    });
    t.innerHTML = rows;
  })();

  /* ---------- beat 2: similarity curves ---------- */
  var c1 = document.getElementById("evo4-c1");
  var g1 = null;
  function beat2Series() {
    var rc = roleColors();
    return [
      { v: D.sim.ali, color: rc[4], label: "Rakan ↔ Alistar" },
      { v: D.sim.ori, color: rc[2], label: "Rakan ↔ Orianna" }
    ];
  }
  function drawBeat2() {
    paintTrio();
    g1 = drawLines(c1, beat2Series(), { height: 280, padR: 128, lo: -0.14,
                                        hi: 0.72, ticks: [0, 0.3, 0.6] });
    // endpoint annotation
    var ctx = c1.getContext("2d");
    var rc = roleColors();
    // numerals, so this one keeps the mono face — but in ink, same reason as the
    // direct labels below it
    ctx.font = '600 11px ui-monospace, "SF Mono", Menlo, monospace';
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    var yA = g1.Y(D.sim.ali[nF - 1]) - 14;
    ctx.fillStyle = css("--ink");
    ctx.fillText((D.sim.ali[0] >= 0 ? "+" : "") + D.sim.ali[0].toFixed(2) +
                 " → +" + D.sim.ali[nF - 1].toFixed(2),
                 g1.padL + g1.iw + 7, yA);
  }
  hoverLines(c1, null, function () { return g1; }, function (i) {
    var s = beat2Series();
    return "<b>" + D.labels[i] + "</b>" +
      "<br>vs Alistar: " + (s[0].v[i] >= 0 ? "+" : "") + s[0].v[i].toFixed(2) +
      "<br>vs Orianna: " + (s[1].v[i] >= 0 ? "+" : "") + s[1].v[i].toFixed(2);
  });

  /* ---------- beat 3: fly-in map ---------- */
  var map = document.getElementById("evo4-map");
  var stripCv = document.getElementById("evo4-strip");
  var playBtn = document.getElementById("evo4-play");
  var scrub = document.getElementById("evo4-scrub");
  var frameLabel = document.getElementById("evo4-label");
  scrub.max = String(nF - 1);

  // The highlighted line-up, shared by the keyframe strip, the animated map and
  // beat 2's payoff figure so all three agree by construction.
  var COMP_I = D.comp.idx, isComp = {};
  COMP_I.forEach(function (i) { isComp[i] = true; });

  (function buildLegend() {
    var lg = document.getElementById("evo4-legend");
    var rc = roleColors();
    lg.innerHTML = ROLE_NAMES.map(function (nm, i) {
      return "<span><i style='background:" + rc[i] + "' data-role='" + i + "'></i>" + nm + "</span>";
    }).join("");
  })();
  function refreshLegend() {
    var rc = roleColors();
    var dots = document.querySelectorAll("#evo4-legend i");
    for (var i = 0; i < dots.length; i++) {
      dots[i].style.background = rc[Number(dots[i].getAttribute("data-role"))];
    }
  }

  var ft = 0;              // fractional frame time 0..nF-1
  var playing = false, rafId = null, mapPlayedOnce = false;
  var mapGeo = null;

  function drawMap() {
    var w = map.clientWidth || 600;
    var H = Math.max(260, Math.min(Math.round(w / D.aspect), 560));
    var ctx = sizeCanvas(map, H);
    ctx.clearRect(0, 0, w, H);
    var rc = roleColors();
    var card = css("--card"), ink = css("--ink"), mut = css("--muted");
    var gold = css("--gold");
    var pad = 26;
    var iw = w - 2 * pad, ih = H - 2 * pad;
    function MX(v) { return pad + v * iw; }
    function MY(v) { return pad + v * ih; }
    mapGeo = { pad: pad, iw: iw, ih: ih };

    var f0 = Math.floor(ft), f1 = Math.min(f0 + 1, nF - 1), fr = ft - f0;
    function cx(c) { return MX(px(f0, c) * (1 - fr) + px(f1, c) * fr); }
    function cy(c) { return MY(py(f0, c) * (1 - fr) + py(f1, c) * fr); }

    // trails: last 6 integer frames
    if (!reduced) {
      ctx.lineWidth = 1;
      for (var c = 0; c < N; c++) {
        ctx.strokeStyle = rc[D.role[c]];
        ctx.globalAlpha = 0.16;
        ctx.beginPath();
        var started = false;
        for (var k = 6; k >= 1; k--) {
          var tf = Math.max(0, f0 - k);
          var tx = MX(px(tf, c)), ty = MY(py(tf, c));
          started ? ctx.lineTo(tx, ty) : ctx.moveTo(tx, ty);
          started = true;
        }
        ctx.lineTo(cx(c), cy(c));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // dots. EVERY dot keeps its role colour: the claim this beat makes is that
    // the roles separate, so dimming the field to make the comp pop would hide
    // the only thing the reader is here to check. The comp is distinguished by
    // size and a ring instead of by being the only thing coloured.
    for (var c2 = 0; c2 < N; c2++) {
      if (isComp[c2]) continue;
      ctx.beginPath();
      ctx.arc(cx(c2), cy(c2), 4.4, 0, 7);
      ctx.fillStyle = rc[D.role[c2]];
      ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = card; ctx.stroke();
    }
    var compDots = [];
    for (var c3 = 0; c3 < COMP_I.length; c3++) {
      var ci = COMP_I[c3], ccx = cx(ci), ccy = cy(ci);
      ctx.beginPath(); ctx.arc(ccx, ccy, 7.6, 0, 7);
      ctx.fillStyle = card; ctx.fill();
      ctx.beginPath(); ctx.arc(ccx, ccy, 5.8, 0, 7);
      ctx.fillStyle = rc[D.role[ci]]; ctx.fill();
      ctx.lineWidth = 1.6; ctx.strokeStyle = card; ctx.stroke();
      compDots.push({ x1: ccx - 9, y1: ccy - 9, x2: ccx + 9, y2: ccy + 9 });
    }

    // ---- labels: ONE placement pass, shared collision state ----
    // v6 shipped three independent label paths (Rakan, Alistar, the fade-ins)
    // that never checked each other. At 390px the dots sit closer together in
    // absolute pixels while the font stays the same size, so "Corki" and
    // "Alistar" fused into unreadable text on the final frame. Every label now
    // goes through placeLabel(), which tries candidate positions in priority
    // order and refuses to draw one that overlaps a label already placed.
    // Seeded with the comp dots themselves, so a name can never be drawn over a
    // highlighted dot — the first mock printed "Xayaḥ" that way.
    var placedBoxes = compDots.slice();
    function hits(b) {
      for (var i = 0; i < placedBoxes.length; i++) {
        var o = placedBoxes[i];
        if (!(b.x2 <= o.x1 || o.x2 <= b.x1 || b.y2 <= o.y1 || o.y2 <= b.y1)) return true;
      }
      return false;
    }
    // Every OTHER champion's dot is a soft obstacle: strongly preferred to avoid,
    // but not worth dropping a comp label over. With 168 dots on a 390px canvas
    // there is often no slot that clears all of them, and dropping the labels
    // would cost the reader the comp they were told to follow. So: two passes.
    // Pass 1 avoids labels, comp dots AND every other dot. Pass 2 relaxes to
    // labels and comp dots only, relying on the halo stroke for legibility.
    // Fused label-on-label text is still never allowed.
    var softBoxes = [];
    for (var sb = 0; sb < N; sb++) {
      if (isComp[sb]) continue;
      softBoxes.push({ x1: cx(sb) - 5.5, y1: cy(sb) - 5.5,
                       x2: cx(sb) + 5.5, y2: cy(sb) + 5.5 });
    }
    function overlaps(b, list) {
      for (var i = 0; i < list.length; i++) {
        var o = list[i];
        if (!(b.x2 <= o.x1 || o.x2 <= b.x1 || b.y2 <= o.y1 || o.y2 <= b.y1)) return true;
      }
      return false;
    }
    function placeLabel(text, dx, dy, fill) {
      var tw = ctx.measureText(text).width, half = 7;
      // right, left, then diagonals, then further out — first clear slot wins
      var cands = [
        [dx + 11, dy, 1], [dx - 11, dy, -1],
        [dx + 9, dy - 14, 1], [dx - 9, dy - 14, -1],
        [dx + 9, dy + 14, 1], [dx - 9, dy + 14, -1],
        [dx + 9, dy - 26, 1], [dx - 9, dy - 26, -1],
        [dx + 9, dy + 26, 1], [dx - 9, dy + 26, -1]
      ];
      for (var pass = 0; pass < 2; pass++) {
        for (var i = 0; i < cands.length; i++) {
          var x = cands[i][0], y = cands[i][1], dir = cands[i][2];
          var x1 = dir > 0 ? x : x - tw;
          var box = { x1: x1 - 2, y1: y - half, x2: x1 + tw + 2, y2: y + half };
          if (box.x1 < 1 || box.x2 > w - 1 || box.y1 < 1 || box.y2 > H - 1) continue;
          if (hits(box)) continue;
          if (pass === 0 && overlaps(box, softBoxes)) continue;
          placedBoxes.push(box);
          ctx.textAlign = dir > 0 ? "left" : "right";
          ctx.lineWidth = 3; ctx.strokeStyle = card; ctx.lineJoin = "round";
          ctx.strokeText(text, x, y);
          ctx.fillStyle = fill;
          ctx.fillText(text, x, y);
          ctx.textAlign = "left";
          return true;
        }
      }
      return false;   // no slot even relaxed: drop it rather than ship fused text
    }

    // The comp carries the labels now. Rakan keeps his gold ring — he is still
    // the thread — and is placed first so he always wins a slot. The old
    // Rakan/Alistar/Orianna/Corki fade-in was retired here: those three now
    // belong to beat 2's payoff figure and beat 4's cards, and dropping them
    // roughly halves the label pressure at 390px, which is where every
    // collision this panel has ever shipped came from.
    var rxi = D.rakan.i, rx = cx(rxi), ry = cy(rxi);
    ctx.beginPath(); ctx.arc(rx, ry, 9.4, 0, 7);
    ctx.lineWidth = 2; ctx.strokeStyle = gold; ctx.stroke();
    ctx.textBaseline = "middle";
    ctx.font = '600 11.5px system-ui, -apple-system, "Segoe UI", sans-serif';
    placeLabel("Rakan", rx, ry, ink);
    for (var li = 0; li < COMP_I.length; li++) {
      if (COMP_I[li] === rxi) continue;
      placeLabel(D.names[COMP_I[li]], cx(COMP_I[li]), cy(COMP_I[li]), ink);
    }

    var fi = Math.round(ft);
    frameLabel.textContent = D.labels[fi] + " · purity " + D.purity[fi].toFixed(2);
    scrub.value = String(fi);
  }

  /* ---- shared label placer for the static figures ------------------------
     Same contract as drawMap's: try candidate offsets in priority order, keep
     every placed rectangle, and DROP a label rather than draw it over another
     label or over a dot. */
  function makePlacer(ctx, w, h, seed) {
    var boxes = (seed || []).slice();
    return function (text, x, y, fill, halo) {
      var tw = ctx.measureText(text).width, half = 7;
      var cands = [
        [x + 10, y, 1], [x - 10, y, -1],
        [x + 8, y - 13, 1], [x - 8, y - 13, -1],
        [x + 8, y + 13, 1], [x - 8, y + 13, -1],
        [x + 8, y - 24, 1], [x - 8, y + 24, -1]
      ];
      for (var i = 0; i < cands.length; i++) {
        var bx = cands[i][0], by = cands[i][1], dir = cands[i][2];
        var x1 = dir > 0 ? bx : bx - tw;
        var box = { x1: x1 - 2, y1: by - half, x2: x1 + tw + 2, y2: by + half };
        if (box.x1 < 1 || box.x2 > w - 1 || box.y1 < 1 || box.y2 > h - 1) continue;
        var clash = false;
        for (var j = 0; j < boxes.length; j++) {
          var o = boxes[j];
          if (!(box.x2 <= o.x1 || o.x2 <= box.x1 || box.y2 <= o.y1 || o.y2 <= box.y1)) {
            clash = true; break;
          }
        }
        if (clash) continue;
        boxes.push(box);
        ctx.textAlign = dir > 0 ? "left" : "right";
        if (halo) {
          ctx.lineWidth = 3; ctx.strokeStyle = halo; ctx.lineJoin = "round";
          ctx.strokeText(text, bx, by);
        }
        ctx.fillStyle = fill;
        ctx.fillText(text, bx, by);
        ctx.textAlign = "left";
        return true;
      }
      return false;
    };
  }

  /* ---- beat 3: the 2x2 keyframe strip ------------------------------------
     Four snapshots chosen by score in the builder. Every dot keeps its role
     colour so the reader can actually SEE the roles separating — that is the
     claim the captions make. The comp is picked out by size and a ring. */
  function drawKey(cv, f) {
    var w = cv.clientWidth || 200;
    var h = Math.round(w * 0.86);
    var ctx = sizeCanvas(cv, h);
    ctx.clearRect(0, 0, w, h);
    var rc = roleColors(), card = css("--card"), ink = css("--ink");
    var pad = 11, iw = w - 2 * pad, ih = h - 2 * pad;
    function KX(c) { return pad + px(f, c) * iw; }
    function KY(c) { return pad + py(f, c) * ih; }
    var small = w < 240;
    for (var i = 0; i < N; i++) {
      if (isComp[i]) continue;
      ctx.beginPath(); ctx.arc(KX(i), KY(i), small ? 1.9 : 2.4, 0, 7);
      ctx.fillStyle = rc[D.role[i]]; ctx.globalAlpha = 0.55; ctx.fill();
    }
    ctx.globalAlpha = 1;
    var seed = [];
    for (var k = 0; k < COMP_I.length; k++) {
      var ci = COMP_I[k], X = KX(ci), Y = KY(ci), r = small ? 3.6 : 4.6;
      ctx.beginPath(); ctx.arc(X, Y, r + 1.8, 0, 7);
      ctx.fillStyle = card; ctx.fill();
      ctx.beginPath(); ctx.arc(X, Y, r, 0, 7);
      ctx.fillStyle = rc[D.role[ci]]; ctx.fill();
      seed.push({ x1: X - r - 3, y1: Y - r - 3, x2: X + r + 3, y2: Y + r + 3 });
    }
    // Under ~240px a cell cannot carry five names without them fusing, and the
    // 390px cells are ~150px. Drop the labels there; the animated map below
    // carries them at full size.
    if (!small) {
      ctx.textBaseline = "middle";
      ctx.font = '600 10.5px system-ui, -apple-system, "Segoe UI", sans-serif';
      var place = makePlacer(ctx, w, h, seed);
      for (var k2 = 0; k2 < COMP_I.length; k2++) {
        place(D.names[COMP_I[k2]], KX(COMP_I[k2]), KY(COMP_I[k2]), ink, card);
      }
    }
  }

  var keyCells = [];
  (function buildKeys() {
    var host = document.getElementById("evo4-keys");
    D.keys.forEach(function (f, i) {
      var cell = document.createElement("div");
      cell.className = "evo4-keycell";
      var hd = document.createElement("div");
      hd.className = "evo4-keyhd";
      hd.innerHTML = "<b>" + (i + 1) + " &middot; " + D.labels[f] + "</b>" +
                     "<span>purity " + D.purity[f].toFixed(2) + "</span>";
      var cv = document.createElement("canvas");
      cv.setAttribute("role", "img");
      cv.setAttribute("aria-label", "Snapshot " + (i + 1) + " of 4, " +
        D.labels[f] + ", role purity " + D.purity[f].toFixed(2) + ".");
      var note = document.createElement("p");
      note.className = "evo4-keynote";
      note.innerHTML = D.keyNotes[i];
      cell.appendChild(hd); cell.appendChild(cv); cell.appendChild(note);
      host.appendChild(cell);
      keyCells.push({ cv: cv, f: f });
    });
  })();

  /* ---- beat 2: the payoff figure ----------------------------------------
     The table quotes three map distances. A distance the reader cannot see is
     a distance they cannot check, so the finished layout is redrawn with only
     those four champions lit. */
  function drawPayoff() {
    var cv = document.getElementById("evo4-pay");
    if (!cv) return;
    var w = cv.clientWidth || 400;
    var h = Math.max(220, Math.min(Math.round(w / D.aspect), 460));
    var ctx = sizeCanvas(cv, h);
    ctx.clearRect(0, 0, w, h);
    var rc = roleColors(), card = css("--card"), ink = css("--ink");
    var mut = css("--muted"), faint = css("--faint");
    var pad = 20, iw = w - 2 * pad, ih = h - 2 * pad, f = nF - 1;
    function PX2(c) { return pad + px(f, c) * iw; }
    function PY2(c) { return pad + py(f, c) * ih; }
    var four = D.comp.payoffIdx, lit = {};
    four.forEach(function (i) { lit[i] = true; });
    for (var i = 0; i < N; i++) {
      if (lit[i]) continue;
      ctx.beginPath(); ctx.arc(PX2(i), PY2(i), 2.3, 0, 7);
      ctx.fillStyle = faint; ctx.fill();
    }
    var R = four[0], XA = four[2];
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1.4; ctx.strokeStyle = mut;
    ctx.beginPath(); ctx.moveTo(PX2(R), PY2(R)); ctx.lineTo(PX2(XA), PY2(XA));
    ctx.stroke(); ctx.setLineDash([]);
    var seed = [];
    four.forEach(function (i) {
      ctx.beginPath(); ctx.arc(PX2(i), PY2(i), 7.2, 0, 7);
      ctx.fillStyle = card; ctx.fill();
      ctx.beginPath(); ctx.arc(PX2(i), PY2(i), 5.4, 0, 7);
      ctx.fillStyle = rc[D.role[i]]; ctx.fill();
      seed.push({ x1: PX2(i) - 9, y1: PY2(i) - 9, x2: PX2(i) + 9, y2: PY2(i) + 9 });
    });
    ctx.textBaseline = "middle";
    ctx.font = '600 11.5px system-ui, -apple-system, "Segoe UI", sans-serif';
    var place = makePlacer(ctx, w, h, seed);
    four.forEach(function (i) { place(D.names[i], PX2(i), PY2(i), ink, card); });
    ctx.font = '600 11px ui-monospace, "SF Mono", Menlo, monospace';
    var pr = {}; D.comp.pairs.forEach(function (p) { pr[p.b] = p; });
    place(pr["Xayah"].dist.toFixed(3) + " apart",
          (PX2(R) + PX2(XA)) / 2, (PY2(R) + PY2(XA)) / 2, mut, card);
    place(pr["Alistar"].dist.toFixed(3) + " apart", PX2(four[1]), PY2(four[1]) + 18,
          mut, card);
  }

  /* ---- beat 2: the counted table ---------------------------------------- */
  (function buildCompTable() {
    var t = document.getElementById("evo4-ctbl");
    if (!t) return;
    var head = "<thead><tr><th>Picked in the same team</th>" +
      "<th class='n'>Together</th><th class='n'>If it were chance</th>" +
      "<th class='n'>Distance on the map</th></tr></thead>";
    var rows = D.comp.pairs.map(function (p) {
      var strong = p.kind !== "none";
      var nm = (strong ? "<b>" : "") + p.a + " + " + p.b + (strong ? "</b>" : "") +
               " &mdash; " + p.note;
      return "<tr><td>" + nm + "</td>" +
        "<td class='n' data-l='Together'>" + p.obs + "</td>" +
        "<td class='n' data-l='If it were chance'>" + p.exp.toFixed(1) + "</td>" +
        "<td class='n' data-l='Distance on the map'>" + p.dist.toFixed(3) + "</td></tr>";
    }).join("");
    t.innerHTML = head + "<tbody>" + rows + "</tbody>";
  })();

  // Rebuilt in v6. The v5 version was 56px tall with its own title printed
  // INSIDE the plot box, on top of the line's baseline, a bare playhead tick
  // hanging below the axis that read as a clipping artifact, and tick text in
  // --faint (2.9:1 on the light card — genuinely illegible, not just thin).
  // Now: 96px, title above the box, two labelled y-ticks, text in --muted.
  var STRIP_TITLE_FONT = '600 11px system-ui, -apple-system, "Segoe UI", sans-serif';
  var STRIP_T1 = "Role purity across training";
  var STRIP_T2 = "never an optimisation target";

  function drawStrip() {
    var w = stripCv.clientWidth || 600;
    var padL = 40, padR = 10, padB = 12;
    var iw = w - padL - padR;

    // Decide the title layout BEFORE sizing the canvas. At 390px the one-line
    // title overran the right edge and was clipped mid-word; the second half is
    // the honesty note, so it wraps rather than being dropped.
    var probe = stripCv.getContext("2d");
    probe.setTransform(1, 0, 0, 1, 0, 0);
    probe.font = STRIP_TITLE_FONT;
    var oneLine = STRIP_T1 + " — " + STRIP_T2;
    var twoLine = probe.measureText(oneLine).width > iw;
    var padT = twoLine ? 38 : 24;
    var h = twoLine ? 108 : 96;

    var ctx = sizeCanvas(stripCv, h);
    ctx.clearRect(0, 0, w, h);
    var gold = css("--gold"), faint = css("--faint"),
        hair = css("--hairline"), muted = css("--muted");
    var ih = h - padT - padB;
    var LO = 0.15, HI = 0.75;
    function X(i) { return padL + iw * i / (nF - 1); }
    function Y(v) { return padT + ih - ih * (v - LO) / (HI - LO); }

    // Title sits above the plot box. It is a text label, so it wears a text
    // token and the page's UI face — not the series colour, not monospace.
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.font = STRIP_TITLE_FONT;
    ctx.fillStyle = muted;
    if (twoLine) {
      ctx.fillText(STRIP_T1, padL, 13);
      ctx.fillText(STRIP_T2, padL, 27);
    } else {
      ctx.fillText(oneLine, padL, 13);
    }

    // Two y-ticks. 0.20 is the chance reference, so it earns a rule across the
    // whole box; 0.70 is a scale anchor and gets a tick mark only.
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(padL, Y(0.2)); ctx.lineTo(padL + iw, Y(0.2)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(padL - 4, Y(0.7)); ctx.lineTo(padL, Y(0.7)); ctx.stroke();

    ctx.font = '500 10.5px ui-monospace, Menlo, monospace';
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText("0.70", padL - 7, Y(0.7));
    ctx.fillText("0.20", padL - 7, Y(0.2));

    ctx.beginPath();
    for (var i = 0; i < nF; i++) {
      i ? ctx.lineTo(X(i), Y(D.purity[i])) : ctx.moveTo(X(i), Y(D.purity[i]));
    }
    ctx.lineWidth = 2; ctx.strokeStyle = gold; ctx.stroke();

    // playhead spans the plot box exactly — no overhang past the baseline
    var phx = X(ft);
    ctx.strokeStyle = faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(phx, padT); ctx.lineTo(phx, padT + ih); ctx.stroke();

    // Name the chance line at its right end — drawn last, and held 8px clear of
    // the right edge so it does not sit under the playhead on the final frame.
    ctx.font = '500 10px system-ui, -apple-system, sans-serif';
    ctx.textAlign = "right"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = muted;
    ctx.fillText("chance", padL + iw - 8, Y(0.2) - 5);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  }

  var MS_PER_FRAME = 150, SLOW_START = 3;
  var lastTs = null;
  function tick(ts) {
    if (!playing) return;
    if (lastTs == null) lastTs = ts;
    var dt = ts - lastTs; lastTs = ts;
    var speed = ft < SLOW_START ? 0.45 : 1;
    ft += speed * dt / MS_PER_FRAME;
    if (ft >= nF - 1) { ft = nF - 1; stopPlay(); }
    drawMap(); drawStrip();
    if (playing) rafId = requestAnimationFrame(tick);
  }
  function stopPlay() {
    playing = false; lastTs = null;
    playBtn.innerHTML = "&#9654; " + (ft >= nF - 1 ? "Replay" : "Play");
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    drawMap(); drawStrip();
  }
  function startPlay() {
    if (ft >= nF - 1) ft = 0;
    playing = true; playBtn.innerHTML = "&#10074;&#10074; Pause";
    if (reduced) {
      // discrete steps, no tweening
      var step = function () {
        if (!playing) return;
        ft = Math.min(Math.round(ft) + 1, nF - 1);
        drawMap(); drawStrip();
        if (ft >= nF - 1) { stopPlay(); return; }
        rafId = null; setTimeout(step, 400);
      };
      setTimeout(step, 400);
    } else {
      rafId = requestAnimationFrame(tick);
    }
  }
  playBtn.addEventListener("click", function () { playing ? stopPlay() : startPlay(); });
  scrub.addEventListener("input", function () {
    if (playing) stopPlay();
    ft = Number(scrub.value); drawMap(); drawStrip();
  });

  map.addEventListener("mousemove", function (e) {
    if (!mapGeo || playing) return;
    var r = map.getBoundingClientRect();
    var mx = e.clientX - r.left, my = e.clientY - r.top;
    var f0 = Math.floor(ft), f1 = Math.min(f0 + 1, nF - 1), fr = ft - f0;
    var best = -1, bd = 12 * 12;
    for (var c = 0; c < N; c++) {
      var dx = mapGeo.pad + (px(f0, c) * (1 - fr) + px(f1, c) * fr) * mapGeo.iw - mx;
      var dy = mapGeo.pad + (py(f0, c) * (1 - fr) + py(f1, c) * fr) * mapGeo.ih - my;
      var dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = c; }
    }
    if (best < 0) { tip.style.display = "none"; return; }
    tip.innerHTML = "<b>" + D.names[best] + "</b> · " + ROLE_NAMES[D.role[best]];
    tip.style.display = "block";
    var tx = Math.min(mx + 14, r.width - 150);
    tip.style.left = tx + "px";
    tip.style.top = (map.offsetTop + my - 32) + "px";
  });
  map.addEventListener("mouseleave", function () { tip.style.display = "none"; });

  /* ---------- beat 4: the exceptions ---------- */
  var c3 = document.getElementById("evo4-c3");
  var bars4 = [];

  function drawBeat4() {
    var w = c3.clientWidth || 600;
    // The range and count lines are dropped below 560px (they collide), so the
    // space reserved for them has to go too — otherwise the canvas keeps 54px of
    // bottom padding for text it never draws, which reads as a hole on mobile.
    var narrow = w < 560;
    var padL = 8, padR = 8, padT = 26, padB = narrow ? 26 : 54;
    var H = narrow ? 222 : 250;
    var ctx = sizeCanvas(c3, H);
    ctx.clearRect(0, 0, w, H);
    var mut = css("--muted"), faint = css("--faint"), hair = css("--hairline");
    var gold = css("--gold"), meta = css("--meta");
    var iw = w - padL - padR, ih = H - padT - padB;
    var q = D.quart, n = q.length;
    var hi = 1.0;
    function Y(v) { return padT + ih - ih * v / hi; }

    // chance baseline — the number that makes the bars mean something
    // The 0.20 baseline is labelled in the prose above the chart, not here —
    // at 390px every on-canvas position for it collided with a bar.
    ctx.setLineDash([3, 3]); ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, Y(0.2)); ctx.lineTo(padL + iw, Y(0.2)); ctx.stroke();
    ctx.setLineDash([]);

    // ONE hue for all four bars: this is a single series (role purity), so a
    // second hue for the last bar was wrong twice over -- it encoded emphasis as
    // identity, and gold-vs-orange failed the palette validator outright
    // (deutan ΔE 1.9, normal-vision ΔE 6.5, both below floor). The dip is called
    // out with a label instead. Dark mode uses a darker gold step because the
    // page's --gold sits at L 0.768 in dark, outside the 0.48-0.67 band for a
    // large fill; both values are validated full-opacity (alpha-compositing
    // washed the chroma out to gray, so these are drawn opaque).
    var barFill = isDark() ? "#AE8C34" : gold;
    var slot = iw / n, bw = Math.min(slot * 0.52, 86);
    bars4 = [];
    q.forEach(function (d, i) {
      var cx = padL + slot * (i + 0.5);
      var x = cx - bw / 2, y = Y(d.purity), h = Y(0) - y;
      var r = Math.min(4, bw / 2, h);
      ctx.fillStyle = barFill;
      ctx.beginPath();
      ctx.moveTo(x, y + h);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.lineTo(x + bw - r, y);
      ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
      ctx.lineTo(x + bw, y + h);
      ctx.closePath();
      ctx.fill();
      // Mark the dip by WEIGHT, not by hue. A reader noted the old orange bar
      // pre-announced "reason two" before they read the text, which is worth
      // keeping; an ink outline does that without giving one hue two meanings.
      if (i === n - 1) {
        ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.stroke();
      }
      bars4.push({ x: x, y: y, w: bw, h: h, d: d, i: i });

      ctx.textAlign = "center";
      ctx.font = "700 14px ui-monospace, Menlo, monospace";
      ctx.fillStyle = css("--ink");
      ctx.fillText(d.purity.toFixed(2), cx, y - 8);

      ctx.font = '600 11px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.fillStyle = mut;
      ctx.fillText(["least seen", "", "", "most seen"][i] || "", cx, padT + ih + 18);
      // At 390px the ranges and counts collide with their neighbours, so drop
      // them rather than ship overlapping text.
      if (!narrow) {
        ctx.font = "500 10.5px ui-monospace, Menlo, monospace";
        ctx.fillStyle = faint;
        ctx.fillText(d.lo.toLocaleString() + "–" + d.hi.toLocaleString(),
                     cx, padT + ih + 33);
        ctx.fillText(d.n + " champions", cx, padT + ih + 46);
      }
    });

    // mark the dip with a label, not a hue
    var last = bars4[bars4.length - 1], prev = bars4[bars4.length - 2];
    if (last && prev) {
      ctx.strokeStyle = faint; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(prev.x + prev.w + 3, prev.y - 2);
      ctx.lineTo(last.x - 3, last.y - 2);
      ctx.stroke();
      ctx.font = '600 10.5px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.fillStyle = mut; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
      ctx.fillText("dips", (prev.x + prev.w + last.x) / 2, prev.y - 8);
    }

    ctx.textAlign = "left"; ctx.fillStyle = mut;
    ctx.font = '600 11px system-ui, -apple-system, "Segoe UI", sans-serif';
    var cap = "share of 5 nearest neighbours sharing the same role  ↑";
    if (ctx.measureText(cap).width > iw) cap = "same role as its 5 neighbours  ↑";
    ctx.fillText(cap, padL, padT - 12);
  }

  /* hover: restores the range and champion count that 390px cannot show */
  c3.addEventListener("mousemove", function (e) {
    var r = c3.getBoundingClientRect();
    var mx = e.clientX - r.left, my = e.clientY - r.top;
    var hit = null;
    bars4.forEach(function (b) {
      if (mx >= b.x - 6 && mx <= b.x + b.w + 6 && my >= b.y - 10) hit = b;
    });
    if (!hit) { tip.style.display = "none"; return; }
    tip.innerHTML = "<b>" + ["least seen", "2nd quarter", "3rd quarter",
                             "most seen"][hit.i] + "</b><br>" +
      hit.d.lo.toLocaleString() + "–" + hit.d.hi.toLocaleString() +
      " draft appearances<br>" + hit.d.n + " champions · purity " +
      hit.d.purity.toFixed(2);
    tip.style.display = "block";
    tip.style.left = Math.min(mx + 14, r.width - 190) + "px";
    tip.style.top = (c3.offsetTop + my - 40) + "px";
  });
  c3.addEventListener("mouseleave", function () { tip.style.display = "none"; });

  // All four beat-4 boxes use ONE template: a heading line, one big number, and
  // a caption. v5 had three different internal structures across the four, plus
  // a coloured border weight the group boxes did not share.
  function statCard(head, headColor, big, cap) {
    var d = document.createElement("div");
    d.className = "evo4-stat";
    var h = document.createElement("div");
    h.className = "evo4-stat-head";
    if (headColor) { h.style.color = headColor; }
    h.textContent = head;
    var b = document.createElement("div");
    b.className = "evo4-stat-big";
    b.textContent = big;
    var c = document.createElement("div");
    c.className = "evo4-stat-cap";
    c.textContent = cap;
    d.appendChild(h); d.appendChild(b); d.appendChild(c);
    return d;
  }

  // Rebuilt on every beat-4 render, not once at load: the champion headings are
  // tinted with the ROLE palette, which differs between light and dark. Built
  // once, they kept the load-time palette after a theme toggle.
  function buildBeat4Cards() {
    var f = D.flexStat;
    var host = document.getElementById("evo4-flex");
    var cards = document.getElementById("evo4-cards");
    host.textContent = ""; cards.textContent = "";
    host.appendChild(statCard(
      "Played in one role", "", f.soloPurity.toFixed(2),
      "average purity across " + f.soloN + " champions"));
    host.appendChild(statCard(
      "Played in more than one role", "", f.flexPurity.toFixed(2),
      "average purity across " + f.flexN + " champions"));

    var rc = roleColors();
    D.cards.forEach(function (c) {
      cards.appendChild(statCard(
        c.name, rc[c.role], c.purity.toFixed(2),
        "purity · " + c.exposure.toLocaleString() + " draft appearances · " +
        (c.flex ? "played in more than one role" : "played in one role")));
    });
  }

  /* ---------- render orchestration ---------- */
  function renderBeat() {
    if (beat === 0) drawBeat1();
    else if (beat === 1) { drawBeat2(); drawPayoff(); }
    else if (beat === 3) { buildBeat4Cards(); drawBeat4(); }
    else {
      refreshLegend();
      keyCells.forEach(function (c) { drawKey(c.cv, c.f); });
      drawMap(); drawStrip();
    }
  }
  darkMq.addEventListener("change", renderBeat);
  new MutationObserver(renderBeat).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });
  window.addEventListener("resize", renderBeat);

  renderBeat();
})();
</script>
"""

def check_claims() -> None:
    """Assert every structural claim the page makes, against the data.

    v4 shipped "Five clusters form anyway" while the layout held ~10 blobs. The
    build could not catch it, because nothing tied the prose to a measurement.
    These assertions do. If a claim drifts, the build fails instead of shipping.
    """
    from sklearn.cluster import DBSCAN, KMeans

    # C1  the page must NOT claim five visible clusters
    blobs = sum(len(set(DBSCAN(eps=0.22, min_samples=4)
                        .fit_predict(P_final[role_idx == k])) - {-1})
                for k in range(5))
    assert blobs > 5, f"layout now has {blobs} blobs — re-check the beat-3 wording"
    assert "Five clusters form" not in FRAGMENT, "the falsified v4 claim is back"

    # C2  the local claim the page does make
    km = KMeans(n_clusters=5, n_init=20, random_state=0).fit_predict(P_final)
    km_pur = sum((role_idx[km == c] == np.bincount(role_idx[km == c]).argmax()).sum()
                 for c in range(5)) / len(role_idx)
    assert km_pur < 0.7, (
        f"KMeans(k=5) purity on the layout is {km_pur:.3f}; if this ever gets high, "
        "a five-cluster claim would become defensible and this guard should change")

    # C3  substitutes are never teammates; different roles are chance-level
    same = next(p for p in pairs if p["a"] == "Rakan" and p["b"] == "Alistar")
    diff = next(p for p in pairs if p["a"] == "Rakan" and p["b"] == "Orianna")
    assert same["obs"] == 0, f"Rakan+Alistar now co-occur {same['obs']}x — beat 2 is wrong"
    assert same["exp"] > 20, "expected-by-chance too small to be worth contrasting"
    assert 0.75 < diff["obs"] / diff["exp"] < 1.25, (
        f"Rakan+Orianna is {diff['obs'] / diff['exp']:.2f}x chance, not ~1x — "
        "the 'about as often as chance' wording no longer holds")

    # C4  exposure story: least-seen quartile is far worse, and the top dips
    assert quartiles[0]["purity"] < quartiles[2]["purity"] - 0.2, (
        "least-seen champions are no longer clearly worse — beat 4 reason one breaks")
    assert quartiles[3]["purity"] < quartiles[2]["purity"], (
        "the top-quartile dip is gone — beat 4 reason two loses its hook")

    # C5  flex champions really are less role-pure
    assert flex_stat["flexPurity"] < flex_stat["soloPurity"] - 0.15, (
        "flex champions are no longer clearly less pure — beat 4 reason two breaks")
    corki = next(c for c in cards if c["name"] == "Corki")
    rakan = next(c for c in cards if c["name"] == "Rakan")
    assert corki["flex"] and not rakan["flex"], "the traced pair lost its contrast"
    assert corki["purity"] < rakan["purity"] - 0.2, (
        "Corki is no longer a clear counter-example to Rakan")
    assert abs(corki["exposure"] - rakan["exposure"]) < 0.5 * rakan["exposure"], (
        "Corki and Rakan no longer have comparable exposure, so 'drafted about as "
        "often as Rakan' is no longer true")

    # C6  NEW in #7 — the comp exhibit. The page names a real game, so the game
    # has to still be in the record, with that exact line-up, on that date.
    assert _found is not None and _found[1] == COMP_GAME["team"] and _found[2] == 3
    assert len(set(comp_block["roles"])) == 5, (
        "the highlighted comp must field one champion per role, or the intro lies")

    # C7  NEW in #7 — the flip that beat 2 now closes on. Both halves must hold:
    # the rival pair is never drafted together AND sits far closer on the map
    # than the partner pair. If either reverses, the whole payoff inverts.
    xay = _by["Xayah"]
    ali = _by["Alistar"]
    non = _by["Orianna"]
    assert ali["obs"] == 0, "the rival pair now co-occurs — the flip breaks"
    assert xay["obs"] > 4 * xay["exp"], (
        f"Rakan+Xayah is only {xay['obs'] / xay['exp']:.1f}x chance; the page calls "
        "them a duo you see picked together every split")
    assert xay["dist"] > 5 * ali["dist"], (
        f"the partner sits {xay['dist'] / ali['dist']:.1f}x further than the rival; "
        "below 5x the 'flip' wording overstates what the map shows")
    assert comp_block["ratio"] >= 5, "the printed multiple must match the guard above"
    assert non["dist"] > xay["dist"], (
        "the unrelated pair should sit further off than the lane partner")

    # C8  NEW in #7 — the two claims REJECTED during design must stay rejected.
    # If the layout ever changes so that they become true, that is a deliberate
    # decision to revisit, not something to discover from a reader.
    _cent = {r: P_unit[-1][role_idx == r].mean(0) for r in range(5)}
    _own = sum(1 for c in COMP
               if min(range(5), key=lambda r: np.linalg.norm(
                   P_unit[-1][champs.index(c)] - _cent[r])) == role_idx[champs.index(c)])
    assert _own < 5, (
        f"{_own}/5 of the comp now land in their own role region; the page "
        "deliberately does NOT claim five-champions-five-regions, so if this is "
        "now 5 the copy is leaving a true and better claim on the table")
    _q = np.array([P_unit[0][champs.index(c)] for c in COMP])
    _r = np.array([P_unit[-1][champs.index(c)] for c in COMP])
    _sp0 = np.linalg.norm(_q[:, None] - _q[None], axis=-1)[np.triu_indices(5, 1)].mean()
    _spF = np.linalg.norm(_r[:, None] - _r[None], axis=-1)[np.triu_indices(5, 1)].mean()
    assert _spF <= _sp0, (
        f"the comp now spreads out over training ({_sp0:.3f} -> {_spF:.3f}); the "
        "'fans out' caption was rejected as false and would need rewriting, not "
        "reinstating by accident")

    # C9  NEW in #7 — every keyframe caption, checked in the 2-D layout the
    # reader is looking at. These per-role numbers are never printed; the page
    # shows only the raw 192-dim score.
    assert purity[KEYS[0]] < 0.25, "frame 1 calls the start random"
    assert 0.30 <= purity[KEYS[1]] < 0.50, "frame 2 must sit in the 0.30s or 0.40s"
    assert purity[KEYS[2]] >= 0.50, "frame 3 must have cleared 0.50"
    assert abs(purity[KEYS[3]] - purity[-1]) < 1e-9, "frame 4 must be the last snapshot"
    _k1 = _layout_role_purity(KEYS[1])
    assert min(_k1["bot"], _k1["sup"]) > max(_k1["top"], _k1["mid"]), (
        f"frame 2 claims the bot lane splits off first; got {_k1}")
    _k2 = _layout_role_purity(KEYS[2])
    assert min(_k2, key=_k2.get) == "top", (
        f"frame 3 claims top separates last; lowest is {min(_k2, key=_k2.get)} in {_k2}")
    _k3 = _layout_role_purity(KEYS[3])
    assert min(_k3, key=_k3.get) == "top", (
        f"frame 3's claim must still hold at the end; got {_k3}")
    # Olaf is named as "still adrift" in frame 3 and as the beat-4 hint.
    _olaf = champ_purity[champs.index("Olaf")]
    assert _olaf < 1.0, "Olaf is now perfectly placed; frame 3 should stop singling him out"

    print(f"claim checks OK — {blobs} blobs, KMeans(k=5) purity {km_pur:.3f}, "
          f"Rakan+Alistar {same['obs']}/{same['exp']}, "
          f"quartiles {[q['purity'] for q in quartiles]}, "
          f"flex {flex_stat['flexPurity']} vs {flex_stat['soloPurity']}")
    print(f"comp checks OK — {_own}/5 in own region (must be <5), spread "
          f"{_sp0:.3f}->{_spF:.3f} (must not grow), partner/rival distance "
          f"{xay['dist'] / ali['dist']:.1f}x, Xayah+Rakan {xay['obs']} vs {xay['exp']}")


check_claims()

fragment = (FRAGMENT
            .replace("@@DATA@@", data_js)
            .replace("@@NTRAIN@@", f"{meta['n_train_games']:,}")
            .replace("@@NCOMPS@@", f"{n_comps:,}")
            .replace("@@P0@@", f"{purity[0]:.3f}")
            .replace("@@PF@@", f"{purity[-1]:.3f}")
            .replace("@@CTEAM@@", COMP_GAME["team"])
            .replace("@@CEVENT@@", COMP_GAME["event"])
            .replace("@@CGAME@@", COMP_GAME["game"])
            .replace("@@CLIST@@", ", ".join(COMP))
            .replace("@@CRATIO@@", str(comp_block["ratio"]))
            .replace("@@XT@@", str(_by["Xayah"]["obs"]))
            .replace("@@AT@@", str(_by["Alistar"]["obs"]))
            .replace("@@AE@@", f'{_by["Alistar"]["exp"]:.1f}')
            .replace("@@NOTHER@@", str(len(champs) - len(COMP))))
assert "@@" not in fragment

ANCHOR = (
    '  :root[data-theme="light"] .embed-tsne-light { display: block; }\n'
    '  :root[data-theme="light"] .embed-tsne-dark { display: none; }\n'
    "</style>"
)

s = SRC.read_text()
assert s.count(ANCHOR) == 1, f"anchor count = {s.count(ANCHOR)}"
s = s.replace(ANCHOR, ANCHOR + "\n" + fragment)

for tag in ("div", "section", "canvas", "script", "style", "table", "button"):
    o, c = s.count("<" + tag), s.count("</" + tag + ">")
    print(tag, o, c)

DST.write_text(s)
print("wrote", DST, f"{len(s) / 1024:.0f}KB")

# ---- standalone test page: page CSS + fragment, all beats visible ----
head_css = s.split("<style>", 1)[1].split("</style>", 1)[0]
# NOTE: #b2end must PAUSE before it scrubs. Selecting beat 3 starts autoplay,
# and setting the scrub to max while the loop is still running makes the very
# next tick() overrun the last frame, finish, and reset to frame 0 — so the
# screenshot silently showed the FIRST frame while claiming to be the last.
# v5's harness had this bug, which means "verified on the final frame" in the
# previous handoff was actually verified on frame 0.
HASH_JS = ("<script>var m=location.hash.match(/b([0-3])(end)?/);"
           "if(m){document.getElementById('evo4-t'+m[1]).click();"
           "if(m[2]){var pb=document.getElementById('evo4-play');"
           "if(pb.textContent.indexOf('Pause')>-1){pb.click();}"
           "var sc=document.getElementById('evo4-scrub');"
           "sc.value=sc.max;sc.dispatchEvent(new Event('input'));}}</script>")


def test_page(theme_attr: str) -> str:
    return ("<!doctype html><html" + theme_attr + "><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>" + head_css + "</style></head><body>"
            + fragment + HASH_JS + "</body></html>")


# A media-query dark preference alone does not stamp data-theme, and the panel's
# colour tokens key off the attribute — so emit an explicit dark page too instead
# of hand-editing one every session.
TEST.write_text(test_page(""))
TEST_DARK.write_text(test_page(' data-theme="dark"'))
print("wrote", TEST, f"{len(TEST.read_text()) / 1024:.0f}KB")
print("wrote", TEST_DARK)
