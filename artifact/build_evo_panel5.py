"""v5 of the evolution panel: fixes the ON-RAMP, adds the exceptions beat.

Why v5 exists (reader feedback on v4, 2026-07-25): "beat 1 and 2 are the biggest
problems to me, I think that's where most folks get lost." Plus a real defect
found by scripts/viz_claim_checks.py: v4's beat 3 claimed "Five clusters form
anyway" when the layout holds ~10 blobs. See docs/viz-selfgrade-2026-07-25.md and
the v6 addendum in docs/viz-research-report.md.

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

Splices into pristine artifact-v2.html -> artifact-v7.html.
Also writes evo_panel5_test.html (page CSS + fragment, all beats reachable via
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
DST = SCRATCH / "artifact-v7.html"
TEST = SCRATCH / "evo_panel5_test.html"
TEST_DARK = SCRATCH / "evo_panel5_test_dark.html"
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
slot_row_names = ["random start", labels[mid_f], "final"]


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
    pair_row("Rakan", "Varus", "support + bot"),
]
print("team-comp pairs:", [(p["a"], p["b"], p["obs"], p["exp"]) for p in pairs])

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
    quartiles.append({
        "lo": int(q_edges[_i]), "hi": int(q_edges[_i + 1]),
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
WISH = ["Rakan", "Alistar", "Leona", "Orianna", "Ahri", "LeBlanc", "Jinx",
        "Kai'Sa", "Ashe", "Lee Sin", "Viego", "Sejuani", "Darius", "K'Sante",
        "Gwen", "Yuumi"]
label_idx = []
for name in WISH:
    if name not in champs:
        continue
    i = champs.index(name)
    if all(np.linalg.norm(P_final[i] - P_final[j]) > 0.16 for j in label_idx):
        label_idx.append(i)
print("final-frame labels:", [champs[i] for i in label_idx])

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
    "trio": {"strips": trio_strips,
             "names": ["Rakan", "Alistar", "Orianna"],
             "roles": [int(role_idx[ri]), int(role_idx[ai]), int(role_idx[oi])]},
    "pairs": pairs,
    "nComps": n_comps,
    "quart": quartiles,
    "flexStat": flex_stat,
    "cards": cards,
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
    <p>To film the race, we retrained one copy of the model on the same data and saved all of its numbers 56 times along the way. The story below has four beats: one champion, then why two champions end up alike, then all 168 at once, then the champions that break the pattern.</p>
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
      <span class="evo4-spacer"></span>
      <button type="button" id="evo4-next" class="evo4-nav">Next &rarr;</button>
    </div>

    <!-- Beat 1 -->
    <div class="evo4-beat" id="evo4-b0" role="tabpanel">
      <p class="evo4-lead"><strong>To the model, Rakan is a list of 192 numbers.</strong> No image, no lore, not even a name. We are not going to tell you what any single number means, because nobody knows and you do not need to know. Watch the <em>pattern</em> across the whole list instead. Each square below is one of the 192 numbers: <span class="evo4-sw evo4-sw-pos"></span> above zero, <span class="evo4-sw evo4-sw-neg"></span> below.</p>
      <div id="evo4-strips0"></div>
      <p class="codecap">The list starts as random noise. Training nudges it millions of times, and a stable pattern settles in. That pattern is the model&rsquo;s whole idea of Rakan &mdash; so the next question is why any <em>other</em> champion would end up with a similar one.</p>
      <details class="evo4-more">
        <summary>Show me the actual digits</summary>
        <p class="sub" style="margin:10px 0 6px">The first 12 of Rakan&rsquo;s 192 numbers, in thousandths. The column positions carry no meaning; they are just the order the numbers happen to sit in.</p>
        <div class="evo4-numwrap"><table class="evo4-num" id="evo4-num"></table></div>
      </details>
    </div>

    <!-- Beat 2 -->
    <div class="evo4-beat" id="evo4-b1" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>Champions that compete for the same job end up with the same numbers.</strong> Think of building a team. A pro team fields exactly five champions, one for each role. So if you have already chosen Rakan, you will never also choose Alistar &mdash; they want the same seat. That makes them rivals for one slot, not partners. Here is that pattern in @@NCOMPS@@ real team line-ups.</p>
      <div id="evo4-pairs" class="evo4-pairs"></div>
      <p class="codecap">Rakan and Alistar are alternatives to each other, so they never appear together. Rakan and Orianna do different jobs, so they turn up together about as often as pure chance would predict. The model is only ever asked one question &mdash; who gets picked next? &mdash; and Rakan and Alistar keep being the answer in the same spots. Every nudge that pushes one of them pushes the other the same way.</p>
      <p class="sub" style="margin:20px 0 8px">So their lists of 192 numbers drift toward the same pattern. All three rows below are sorted the same way &mdash; by Rakan&rsquo;s values, smallest to largest &mdash; so his row runs blue to orange. Alistar&rsquo;s row follows the same slope. Orianna&rsquo;s does not:</p>
      <div id="evo4-strips1"></div>
      <p class="codecap">Sorting only makes the comparison easy to see; the order of the 192 numbers carries no meaning of its own, and the same order is applied to all three rows.</p>
      <p class="sub" style="margin:20px 0 6px">Measured over training, that resemblance is a single number per pair:</p>
      <canvas id="evo4-c1" style="width:100%;display:block" role="img" aria-label="Line chart: similarity between Rakan and Alistar rises from near zero to 0.56 over training, while Rakan and Orianna stays near zero"></canvas>
      <p class="codecap">Similarity runs from &minus;1 (opposite patterns) through 0 (nothing in common) to +1 (identical direction), across all 192 numbers at once. Nobody ever told the model that Rakan and Alistar are supports. It only read drafts. The same pull acted on every pair of champions at the same time &mdash; that is the next beat.</p>
    </div>

    <!-- Beat 3 -->
    <div class="evo4-beat" id="evo4-b2" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>Every champion ends up surrounded by others who play its role.</strong> Press play. Each dot is one champion, colored by its true role &mdash; a label the model never saw. Watch who each dot settles next to. The purity score under the map measures exactly that: how often a champion&rsquo;s five nearest neighbours share its role. Chance is 0.20; it finishes at @@PF@@. You will count more than five groups on the finished map, and that is real &mdash; the next beat explains it.</p>
      <div class="evo4-legend" id="evo4-legend" aria-hidden="true"></div>
      <canvas id="evo4-map" style="width:100%;display:block" role="img" aria-label="Animated scatter map: 168 champion dots fly from a random cloud into five role clusters as training progresses"></canvas>
      <div class="evo4-controls">
        <button id="evo4-play" type="button">&#9654; Play</button>
        <input id="evo4-scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Training snapshot">
        <span id="evo4-label"></span>
      </div>
      <canvas id="evo4-strip" style="width:100%;display:block" role="img" aria-label="Purity over training, with playhead"></canvas>
      <p class="codecap"><b>When each dot settles is real. Where it passes through is drawn.</b> The final frame is a true t-SNE of the last snapshot and purity is measured in the raw 192 dimensions at every snapshot, but in between each dot simply slides toward its final spot, paced by its own measured progress &mdash; so read the timing, not the path. <span class="evo4-prov">Demo run: seed 16, config d192x4L6H, same 2024&ndash;2026 data and split as the shipped model (@@NTRAIN@@ training games), 5-NN purity @@P0@@ &rarr; @@PF@@.</span></p>
    </div>

    <!-- Beat 4 -->
    <div class="evo4-beat" id="evo4-b3" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>The champions in the wrong place explain the most.</strong> The finished map is not five tidy clusters, and the reason is not sloppiness in the model. A dot lands away from its own kind for exactly two reasons, and both are measurable.</p>
      <p class="sub" style="margin:16px 0 6px"><b>Reason one: the model hardly ever saw it.</b> Sort all 168 champions by how often they appear in the draft record, then measure how well each quarter sorted itself by role. The dotted line marks 0.20 &mdash; what you would get from pure chance:</p>
      <canvas id="evo4-c3" style="width:100%;display:block" role="img" aria-label="Bar chart: role purity by exposure quartile, rising from 0.44 for the least-seen champions to 0.91, then dipping to 0.83 for the most-seen"></canvas>
      <p class="codecap">Champions pros rarely draft give the model almost nothing to learn from, so their numbers stay close to the random noise they started as. Evidence is the raw material here: no drafts, no pattern.</p>
      <p class="sub" style="margin:22px 0 6px"><b>Reason two: it genuinely does two jobs.</b> That dip in the last quarter is not noise. Some champions get played in more than one role, so there is no single right place to put them:</p>
      <div id="evo4-flex" class="evo4-flexstat"></div>
      <div id="evo4-cards" class="evo4-cards"></div>
      <p class="codecap">Corki is drafted about as often as Rakan and the model still cannot pin him down, because pros play him in more than one role. He is not a failure of the model &mdash; he is the model correctly reporting that the question has no single answer. Purity is the same measure used on the map: the share of a champion&rsquo;s five nearest neighbours that share its role.</p>
    </div>

    <div id="evo4-tip" role="status"></div>
  </div>
</div>

<style>
  .evo4-steps { display:flex; gap:8px; margin:2px 0 16px; flex-wrap:wrap; align-items:center; }
  .evo4-steps .evo4-spacer { flex:1; }
  .evo4-steps button {
    font: 600 12px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 8px 12px; cursor: pointer;
  }
  .evo4-steps button[aria-selected="true"] { color: var(--ink); border-color: var(--model); }
  .evo4-steps button.evo4-nav { color: var(--card); background: var(--ink); border-color: var(--ink); }
  .evo4-steps button:focus-visible,
  .evo4-controls button:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
  .evo4-lead { font-size: 15px; max-width: 62ch; }
  .evo4-legend { display:flex; gap:14px; flex-wrap:wrap; margin: 2px 0 8px; }
  .evo4-legend span {
    font: 600 11.5px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); display:inline-flex; align-items:center; gap:5px;
  }
  .evo4-legend i { width:9px; height:9px; border-radius:50%; display:inline-block; }
  .evo4-controls { display:flex; align-items:center; gap:12px; margin:10px 0 4px; }
  .evo4-controls button {
    font: 600 13px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--ink); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 12px; cursor: pointer; white-space: nowrap;
  }
  .evo4-controls input[type=range] { flex:1; accent-color: var(--model); min-width:0; }
  #evo4-label {
    font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); min-width: 19ch; text-align: right;
  }
  #evo4-tip {
    position:absolute; display:none; pointer-events:none;
    font: 600 12px/1.4 ui-monospace, "SF Mono", Menlo, monospace;
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
  @media (max-width:560px) { #evo4-label { display:none; } }

  /* ---- v5: vector strips, team-comp pairs, beat-4 exhibits ---- */
  .evo4-sw {
    display:inline-block; width:10px; height:10px; border-radius:2px;
    vertical-align:-1px; margin:0 1px;
  }
  .evo4-sw-pos { background: var(--meta); }
  .evo4-sw-neg { background: var(--model); }
  .evo4-striprow { margin: 0 0 10px; }
  .evo4-striprow .evo4-striplab {
    font: 600 11.5px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); display:flex; justify-content:space-between; gap:10px;
  }
  .evo4-striprow canvas { width:100%; display:block; }
  .evo4-more { margin-top: 14px; }
  .evo4-more summary {
    font: 600 12px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); cursor: pointer; padding: 6px 0;
  }
  .evo4-more summary:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
  .evo4-prov { color: var(--faint); }
  .evo4-pairs { display:flex; flex-direction:column; gap:8px; margin: 4px 0 2px; }
  .evo4-pairrow {
    display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;
    padding: 9px 12px; border:1px solid var(--hairline); border-radius:8px;
    background: var(--inset);
    font: 500 12.5px/1.45 ui-monospace, "SF Mono", Menlo, monospace;
  }
  .evo4-pairrow b { color: var(--ink); font-weight:700; }
  .evo4-pairrow .evo4-pairnote { color: var(--faint); }
  .evo4-pairrow .evo4-obs { margin-left:auto; color: var(--muted); white-space:nowrap; }
  .evo4-pairrow .evo4-zero { color: var(--meta); font-weight:700; }
  .evo4-flexstat { display:flex; gap:10px; flex-wrap:wrap; margin: 4px 0 2px; }
  .evo4-flexstat div {
    flex:1 1 210px; padding:11px 13px; border:1px solid var(--hairline);
    border-radius:8px; background: var(--inset);
  }
  .evo4-flexstat .evo4-big {
    font: 700 21px/1.15 ui-monospace, "SF Mono", Menlo, monospace; color: var(--ink);
  }
  .evo4-flexstat .evo4-cap {
    font: 500 11.5px/1.45 ui-monospace, "SF Mono", Menlo, monospace; color: var(--muted);
  }
  .evo4-cards { display:flex; gap:10px; flex-wrap:wrap; margin: 12px 0 2px; }
  .evo4-cards div {
    flex:1 1 210px; padding:11px 13px; border:1px solid var(--hairline);
    border-radius:8px;
    font: 500 12px/1.6 ui-monospace, "SF Mono", Menlo, monospace; color: var(--muted);
  }
  .evo4-cards .evo4-cname { font-weight:700; font-size:14px; }
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
  function setBeat(b) {
    beat = b;
    tabs.forEach(function (t, i) { t.setAttribute("aria-selected", String(i === b)); });
    beats.forEach(function (el, i) { el.hidden = (i !== b); });
    nextBtn.hidden = (b === NBEATS - 1);
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
    // direct labels at line ends, nudged apart
    ctx.textAlign = "left";
    var used = [];
    series.forEach(function (s) {
      var ly = Y(s.v[nF - 1]);
      used.sort(function (a, b) { return a - b; });
      for (var u = 0; u < used.length; u++) {
        if (Math.abs(used[u] - ly) < 13) { ly = used[u] + 13; }
      }
      used.push(ly);
      ctx.fillStyle = s.color;
      ctx.fillText(s.label, padL + iw + 7, ly);
    });
    // x axis caption (skip the left one when narrow — they collide)
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

  function paintStrip(cv, values) {
    var H = 30;
    var ctx = sizeCanvas(cv, H);
    var w = cv.clientWidth;
    ctx.clearRect(0, 0, w, H);
    var pos = css("--meta"), neg = css("--model"), hair = css("--hairline");
    var n = values.length, cw = w / n;
    for (var i = 0; i < n; i++) {
      var v = values[i] / 100;                      // -1 .. 1
      var a = Math.min(Math.abs(v), 1);
      ctx.globalAlpha = 0.10 + 0.90 * a * a;        // squared: pattern, not noise
      ctx.fillStyle = v >= 0 ? pos : neg;
      ctx.fillRect(i * cw, 0, Math.max(cw - 0.35, 0.6), H);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, H - 1);
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

  function drawBeat1() { strips0.forEach(function (s) { paintStrip(s.cv, s.values); }); }
  function paintTrio() {
    var rc = roleColors();
    strips1.forEach(function (s, i) {
      paintStrip(s.cv, s.values);
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
        "</span> line-ups · chance would give " + p.exp + "</span>";
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
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    var yA = g1.Y(D.sim.ali[nF - 1]) - 14;
    ctx.fillStyle = rc[4];
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

    // dots
    for (var c2 = 0; c2 < N; c2++) {
      ctx.beginPath();
      ctx.arc(cx(c2), cy(c2), 4.4, 0, 7);
      ctx.fillStyle = rc[D.role[c2]];
      ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = card; ctx.stroke();
    }

    // Rakan: gold ring + always-on label (the story's thread)
    var rx = cx(D.rakan.i), ry = cy(D.rakan.i);
    ctx.beginPath(); ctx.arc(rx, ry, 7, 0, 7);
    ctx.lineWidth = 2; ctx.strokeStyle = gold; ctx.stroke();
    ctx.font = "600 11.5px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.lineWidth = 3; ctx.strokeStyle = card; ctx.lineJoin = "round";
    ctx.strokeText("Rakan", rx + 11, ry);
    ctx.fillStyle = ink;
    ctx.fillText("Rakan", rx + 11, ry);

    // final frame: named champions fade in
    var settle = Math.max(0, (ft - (nF - 3)) / 2);   // 0 -> 1 over last 2 frames
    if (settle > 0) {
      ctx.globalAlpha = Math.min(settle, 1);
      ctx.font = "500 11px ui-monospace, Menlo, monospace";
      ctx.lineWidth = 3; ctx.strokeStyle = card; ctx.lineJoin = "round";
      D.labelIdx.forEach(function (i) {
        if (i === D.rakan.i) return;
        var lx = cx(i) + 8, ly = cy(i) - 6;
        var tw = ctx.measureText(D.names[i]).width;
        if (lx + tw > w - 2) { lx = cx(i) - 8 - tw; }
        ctx.strokeText(D.names[i], lx, ly);
        ctx.fillStyle = mut;
        ctx.fillText(D.names[i], lx, ly);
      });
      ctx.globalAlpha = 1;
    }

    var fi = Math.round(ft);
    frameLabel.textContent = D.labels[fi] + " · purity " + D.purity[fi].toFixed(2);
    scrub.value = String(fi);
  }

  function drawStrip() {
    var h = 56;
    var ctx = sizeCanvas(stripCv, h);
    var w = stripCv.clientWidth;
    ctx.clearRect(0, 0, w, h);
    var gold = css("--gold"), faint = css("--faint"), hair = css("--hairline");
    var pad = 8, iw = w - 2 * pad, ih = h - 22;
    function X(i) { return pad + iw * i / (nF - 1); }
    function Y(v) { return pad + ih - ih * (v - 0.15) / (0.75 - 0.15); }
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, Y(0.2)); ctx.lineTo(pad + iw, Y(0.2)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    for (var i = 0; i < nF; i++) {
      i ? ctx.lineTo(X(i), Y(D.purity[i])) : ctx.moveTo(X(i), Y(D.purity[i]));
    }
    ctx.lineWidth = 1.8; ctx.strokeStyle = gold; ctx.stroke();
    var phx = X(ft);
    ctx.strokeStyle = faint;
    ctx.beginPath(); ctx.moveTo(phx, pad - 2); ctx.lineTo(phx, pad + ih + 2); ctx.stroke();
    ctx.font = "600 10.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = gold; ctx.fillText("role purity ↑ (never optimized)", pad, h - 5);
    ctx.fillStyle = faint; ctx.textAlign = "right";
    ctx.fillText("chance 0.20", pad + iw * 0.8, Y(0.2) - 4);
    ctx.textAlign = "left";
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
    var H = 250;
    var ctx = sizeCanvas(c3, H);
    var w = c3.clientWidth;
    ctx.clearRect(0, 0, w, H);
    var mut = css("--muted"), faint = css("--faint"), hair = css("--hairline");
    var gold = css("--gold"), meta = css("--meta");
    var padL = 8, padR = 8, padT = 26, padB = 54;
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
    var narrow = w < 560;

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
      bars4.push({ x: x, y: y, w: bw, h: h, d: d, i: i });

      ctx.textAlign = "center";
      ctx.font = "700 14px ui-monospace, Menlo, monospace";
      ctx.fillStyle = css("--ink");
      ctx.fillText(d.purity.toFixed(2), cx, y - 8);

      ctx.font = "600 11px ui-monospace, Menlo, monospace";
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
      ctx.font = "600 10.5px ui-monospace, Menlo, monospace";
      ctx.fillStyle = mut; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
      ctx.fillText("dips", (prev.x + prev.w + last.x) / 2, prev.y - 8);
    }

    ctx.textAlign = "left"; ctx.fillStyle = mut;
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
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

  (function buildBeat4Static() {
    var f = D.flexStat;
    var host = document.getElementById("evo4-flex");
    [[f.soloPurity, f.soloN, "champions played in ONE role"],
     [f.flexPurity, f.flexN, "champions played in MORE than one role"]]
      .forEach(function (row) {
        var d = document.createElement("div");
        d.innerHTML = "<div class='evo4-big'>" + row[0].toFixed(2) + "</div>" +
                      "<div class='evo4-cap'>" + row[2] + " (" + row[1] + ")</div>";
        host.appendChild(d);
      });
    var rc = roleColors();
    var cards = document.getElementById("evo4-cards");
    D.cards.forEach(function (c) {
      var d = document.createElement("div");
      d.style.borderColor = rc[c.role];
      d.innerHTML =
        "<div class='evo4-cname' style='color:" + rc[c.role] + "'>" + c.name + "</div>" +
        "appears in " + c.exposure.toLocaleString() + " drafts<br>" +
        "purity " + c.purity.toFixed(2) + "<br>" +
        (c.flex ? "played in more than one role" : "played in one role");
      cards.appendChild(d);
    });
  })();

  /* ---------- render orchestration ---------- */
  function renderBeat() {
    if (beat === 0) drawBeat1();
    else if (beat === 1) drawBeat2();
    else if (beat === 3) drawBeat4();
    else { refreshLegend(); drawMap(); drawStrip(); }
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

    print(f"claim checks OK — {blobs} blobs, KMeans(k=5) purity {km_pur:.3f}, "
          f"Rakan+Alistar {same['obs']}/{same['exp']}, "
          f"quartiles {[q['purity'] for q in quartiles]}, "
          f"flex {flex_stat['flexPurity']} vs {flex_stat['soloPurity']}")


check_claims()

fragment = (FRAGMENT
            .replace("@@DATA@@", data_js)
            .replace("@@NTRAIN@@", f"{meta['n_train_games']:,}")
            .replace("@@NCOMPS@@", f"{n_comps:,}")
            .replace("@@P0@@", f"{purity[0]:.3f}")
            .replace("@@PF@@", f"{purity[-1]:.3f}"))
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
HASH_JS = ("<script>var m=location.hash.match(/b([0-3])(end)?/);"
           "if(m){document.getElementById('evo4-t'+m[1]).click();"
           "if(m[2]){var sc=document.getElementById('evo4-scrub');"
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
