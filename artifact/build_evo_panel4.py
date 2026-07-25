"""v4 of the evolution panel: guided story + fly-in map (replaces the matrix).

Three beats a reader steps through:
  1. One champion (Rakan): four of his 192 numbers tuning, plus the raw
     first-12-slots table at start / midway / final.
  2. Friends emerge: Rakan<->Alistar vs Rakan<->Orianna centered-cosine
     similarity curves over all 56 snapshots.
  3. All 168 at once: animated "fly-in" map. Final frame = true t-SNE of the
     final snapshot; earlier positions interpolate each champion from a seeded
     random cloud toward its final spot, paced by its measured progress
     (cosine of its centered-normed vector vs its final state). Honestly
     captioned: paths show real per-champion pace, in-between map distances
     are illustrative.

Splices into pristine artifact-v2.html -> artifact-v6.html.
Also writes evo_panel4_test.html (page CSS + fragment, all beats forced
visible) for headless screenshot checks.
"""
import base64
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

SCRATCH = Path(__file__).parent
REPO = Path.home() / "Documents/repos/lol-meta-tracker"
SRC = SCRATCH / "artifact-v2.html"
DST = SCRATCH / "artifact-v6.html"
TEST = SCRATCH / "evo_panel4_test.html"

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
    },
    "sim": {"ali": sim_ali, "ori": sim_ori},
    "labelIdx": label_idx,
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
    <p>To film the race, we retrained one copy of the model on the same data and saved all of its numbers 56 times along the way. The story below has three beats: one champion, then one friendship, then all 168 champions at once.</p>
  </section>
</div>

<div class="panelwrap">
  <div class="panel" id="evo4-panel" style="position:relative">
    <h3>How role clusters form, in three beats</h3>
    <p class="sub">Step through with the buttons. Every number shown is real, from the saved snapshots.</p>

    <div class="evo4-steps" role="tablist" aria-label="Story beats">
      <button type="button" id="evo4-t0" role="tab" aria-selected="true">1 &middot; One champion</button>
      <button type="button" id="evo4-t1" role="tab" aria-selected="false">2 &middot; Friends</button>
      <button type="button" id="evo4-t2" role="tab" aria-selected="false">3 &middot; All 168</button>
      <span class="evo4-spacer"></span>
      <button type="button" id="evo4-next" class="evo4-nav">Next &rarr;</button>
    </div>

    <!-- Beat 1 -->
    <div class="evo4-beat" id="evo4-b0" role="tabpanel">
      <p class="evo4-lead"><strong>Meet Rakan.</strong> To the model he is nothing but a list of 192 numbers &mdash; no image, no lore, not even a name. The list starts as random noise. Training nudges it millions of times. Here are the four of his numbers that moved the most.</p>
      <canvas id="evo4-c0" style="width:100%;display:block" role="img" aria-label="Line chart: four of Rakan's 192 embedding values changing across 56 training snapshots"></canvas>
      <p class="sub" style="margin:14px 0 6px">The first 12 of Rakan&rsquo;s 192 numbers, in thousandths:</p>
      <div class="evo4-numwrap"><table class="evo4-num" id="evo4-num"></table></div>
      <p class="codecap">No single number means anything on its own &mdash; the model never looks at slot 3 of Rakan and concludes something. What training shapes is where the whole list of 192 <em>points</em>. That&rsquo;s the next beat.</p>
    </div>

    <!-- Beat 2 -->
    <div class="evo4-beat" id="evo4-b1" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>Nobody told the model Rakan is a support.</strong> But pro teams pick Rakan in the same drafting situations as Alistar &mdash; another support. Every nudge pushes their two lists the same way, so the lists drift together. Orianna shows up in different situations; her list stays unrelated to Rakan&rsquo;s.</p>
      <canvas id="evo4-c1" style="width:100%;display:block" role="img" aria-label="Line chart: similarity between Rakan and Alistar rises from near zero to 0.56 over training, while Rakan and Orianna stays near zero"></canvas>
      <p class="codecap">Similarity = do the two lists point the same way, from &minus;1 (opposite) through 0 (unrelated) to +1 (same direction), measured across all 192 numbers after removing the shared average direction. This drift-toward-your-own-kind happened for every champion pair at once &mdash; that&rsquo;s the last beat.</p>
    </div>

    <!-- Beat 3 -->
    <div class="evo4-beat" id="evo4-b2" role="tabpanel" hidden>
      <p class="evo4-lead"><strong>Every champion ran this same race, all at once.</strong> Press play. Each dot is one champion, colored by its true role &mdash; a label the model never saw. Five clusters form anyway. The purity score under the map tracks it: how often a champion&rsquo;s five nearest neighbors share its role. Chance is 0.20.</p>
      <div class="evo4-legend" id="evo4-legend" aria-hidden="true"></div>
      <canvas id="evo4-map" style="width:100%;display:block" role="img" aria-label="Animated scatter map: 168 champion dots fly from a random cloud into five role clusters as training progresses"></canvas>
      <div class="evo4-controls">
        <button id="evo4-play" type="button">&#9654; Play</button>
        <input id="evo4-scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Training snapshot">
        <span id="evo4-label"></span>
      </div>
      <canvas id="evo4-strip" style="width:100%;display:block" role="img" aria-label="Purity over training, with playhead"></canvas>
      <p class="codecap"><b>How this animation is honest &mdash; and where it isn&rsquo;t.</b> The finish is real: the final frame is a true t-SNE of the last snapshot, and the purity number is measured in the raw 192-dimensional space at every snapshot. The in-between is paced, not mapped: each champion moves from a random start toward its final spot, and its progress along that path is its measured similarity to its own final state at that snapshot. So <em>when</em> each champion settles is real; <em>where</em> it passes through on the way is interpolation. We do not re-run t-SNE per snapshot on purpose &mdash; t-SNE re-optimizes its layout every time, which animates layout noise, not learning. Demo run: single seed (16), production config d192x4L6H, same 2024&ndash;2026 data and split as the shipped model (@@NTRAIN@@ training games), 5-NN purity @@P0@@ &rarr; @@PF@@.</p>
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
  var beat = 0;
  var tabs = [0, 1, 2].map(function (i) { return document.getElementById("evo4-t" + i); });
  var beats = [0, 1, 2].map(function (i) { return document.getElementById("evo4-b" + i); });
  var nextBtn = document.getElementById("evo4-next");
  function setBeat(b) {
    beat = b;
    tabs.forEach(function (t, i) { t.setAttribute("aria-selected", String(i === b)); });
    beats.forEach(function (el, i) { el.hidden = (i !== b); });
    nextBtn.hidden = (b === 2);
    tip.style.display = "none";
    renderBeat();
    if (b === 2 && !reduced && !mapPlayedOnce) { mapPlayedOnce = true; startPlay(); }
  }
  tabs.forEach(function (t, i) { t.addEventListener("click", function () { setBeat(i); }); });
  nextBtn.addEventListener("click", function () { setBeat(Math.min(beat + 1, 2)); });

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

  /* ---------- beat 1: four dims + slots table ---------- */
  var c0 = document.getElementById("evo4-c0");
  var g0 = null;
  var dimPalette = [css("--model"), css("--meta"), css("--gold"), css("--team")];
  function beat1Series() {
    var p = [css("--model"), css("--meta"), css("--gold"), css("--team")];
    return D.rakan.curves.map(function (v, i) {
      return { v: v, color: p[i], label: "slot " + (D.rakan.dims[i] + 1) };
    });
  }
  function drawBeat1() {
    var s = beat1Series();
    var flat = [];
    s.forEach(function (x) { flat = flat.concat(x.v); });
    var lo = Math.min.apply(null, flat), hi = Math.max.apply(null, flat);
    var m = 0.06;
    g0 = drawLines(c0, s, { height: 240, padR: 64, lo: lo - m, hi: hi + m,
                            ticks: [-0.2, 0, 0.2] });
  }
  hoverLines(c0, null, function () { return g0; }, function (i) {
    var s = beat1Series();
    var out = "<b>" + D.labels[i] + "</b>";
    s.forEach(function (x) {
      out += "<br>" + x.label + ": " + (x.v[i] >= 0 ? "+" : "") + x.v[i].toFixed(3);
    });
    return out;
  });

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

  /* ---------- render orchestration ---------- */
  function renderBeat() {
    if (beat === 0) drawBeat1();
    else if (beat === 1) drawBeat2();
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

fragment = (FRAGMENT
            .replace("@@DATA@@", data_js)
            .replace("@@NTRAIN@@", f"{meta['n_train_games']:,}")
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
test = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>" + head_css + "</style></head><body>"
        + fragment +
        "<script>var m=location.hash.match(/b([0-2])(end)?/);"
        "if(m){document.getElementById('evo4-t'+m[1]).click();"
        "if(m[2]){var sc=document.getElementById('evo4-scrub');"
        "sc.value=sc.max;sc.dispatchEvent(new Event('input'));}}</script>"
        "</body></html>")
TEST.write_text(test)
print("wrote", TEST, f"{len(test) / 1024:.0f}KB")
