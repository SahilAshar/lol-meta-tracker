"""v3 of the evolution panel, per viz-research-report.md:

Design #1 (primary): role-block-sorted 168x168 cosine-similarity heatmap,
scrubbed over ~18 keyframes baked as grayscale PNGs, colorized in JS through
a theme-aware LUT. Design #2 (toggle): champs x 192-dims raw-value heatmap,
columns reordered by final-epoch correlation clustering. The animated t-SNE
scatter is gone. The raw-numbers exhibit (12 slots + cosine bars) stays.

Splices into pristine artifact-v2.html -> artifact-v5.html.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

SCRATCH = Path(__file__).parent
REPO = Path.home() / "Documents/repos/lol-meta-tracker"
SRC = SCRATCH / "artifact-v2.html"
DST = SCRATCH / "artifact-v5.html"

meta = json.loads(
    (REPO / "data/processed/embedding_evolution_v08_demo.json").read_text())
z = np.load(REPO / "data/processed/embedding_evolution_v08_snapshots.npz",
            allow_pickle=False)
embs = z["embs"]
champs = np.array([str(c) for c in z["champs"]])
role = np.array([str(r) for r in z["role"]])
labels = [str(l) for l in z["labels"]]
val_loss = z["val_loss"].astype(float)
purity = z["purity"].astype(float)
F, C, D = embs.shape

# ---- role-block ordering (fixed across all frames) ----
ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]
ROLE_NAMES = {"top": "Top", "jng": "Jungle", "mid": "Mid",
              "bot": "Bot", "sup": "Support"}
# Center before cosine: all champions share a growing common direction
# (frequency/popularity); removing the mean exposes the contrasts, matching
# what the (translation-invariant) 5-NN purity metric actually measures.
def center_norm(e: np.ndarray) -> np.ndarray:
    e = e - e.mean(0)
    return e / np.linalg.norm(e, axis=1, keepdims=True)

final_n = center_norm(embs[-1])
order = []
bounds = []
for r in ROLE_ORDER:
    idx = np.where(role == r)[0]
    sub = final_n[idx]
    if len(idx) > 2:
        lk = linkage(sub, method="average", metric="cosine")
        idx = idx[leaves_list(lk)]
    bounds.append({"name": ROLE_NAMES[r], "start": len(order),
                   "end": len(order) + len(idx)})
    order.extend(idx.tolist())
order = np.array(order)
N = len(order)
print(f"{N} of {C} champs role-sorted;", [(b['name'], b['end']-b['start']) for b in bounds])

# ---- dimension column order for design #2: cluster correlated dims ----
fe = embs[-1][order]
corr_d = np.corrcoef(fe.T)
dist_d = squareform(np.clip(1 - corr_d, 0, 2), checks=False)
dim_order = leaves_list(linkage(dist_d, method="average"))

# ---- keyframes: dense where val_loss moves, always first and last ----
NK = 18
delta = np.abs(np.diff(val_loss))
cum = np.concatenate([[0], np.cumsum(delta + 1e-6)])
targets = np.linspace(0, cum[-1], NK)
keyframes = sorted(set(int(np.argmin(np.abs(cum - t))) for t in targets)
                   | {0, F - 1})
print(f"{len(keyframes)} keyframes:", keyframes)

SIM_CLIP = 0.45
DIM_CLIP = 0.25

def to_png_b64(mat: np.ndarray, clip: float) -> str:
    """Lossy WebP (q80): the random-init frames are incompressible losslessly;
    hover values are shown as approximate. Typical error well under ±0.05."""
    u8 = np.round((np.clip(mat, -clip, clip) / clip + 1) * 127.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").convert("RGB").save(
        buf, "WEBP", quality=80, method=6)
    return base64.b64encode(buf.getvalue()).decode()

frames_payload = []
for f in keyframes:
    e = embs[f][order]
    en = center_norm(embs[f])[order]
    sim_b64 = to_png_b64(en @ en.T, SIM_CLIP)
    dim_b64 = to_png_b64(e[:, dim_order], DIM_CLIP)
    frames_payload.append({
        "f": int(f), "l": labels[f], "v": round(val_loss[f], 3),
        "p": round(purity[f], 3), "sim": sim_b64, "dim": dim_b64,
    })
sim_kb = sum(len(fr["sim"]) for fr in frames_payload) / 1024
dim_kb = sum(len(fr["dim"]) for fr in frames_payload) / 1024
print(f"sim PNGs {sim_kb:.0f}KB b64, dim PNGs {dim_kb:.0f}KB b64")

# ---- raw-numbers exhibit (kept from v2 panel), indexed by keyframe ----
ch = list(champs)
def pick(*names):
    for n in names:
        if n in ch:
            return ch.index(n)
    raise KeyError(names)

i1, i2, i3 = pick("Rakan", "Leona"), pick("Alistar", "Braum"), pick("Orianna", "Ahri")
def cosv(f, a, b):
    cn = center_norm(embs[f])
    return float(cn[a] @ cn[b])

exhibit = {
    "names": [ch[i1], ch[i2], ch[i3]],
    "roles": [str(role[i1]), str(role[i2]), str(role[i3])],
    "dims": [[np.round(embs[f, i, :12] * 1000).astype(int).tolist()
              for f in keyframes] for i in (i1, i2, i3)],
    "sim": [[round(cosv(f, i1, i2), 3), round(cosv(f, i1, i3), 3)]
            for f in keyframes],
}

payload = {
    "bounds": bounds,
    "names": [ch[i] for i in order],
    "nDims": int(D),
    "simClip": SIM_CLIP, "dimClip": DIM_CLIP,
    "loss": [round(v, 3) for v in val_loss],
    "purity": [round(p, 3) for p in purity],
    "ex": exhibit,
    "frames": frames_payload,
}
data_js = json.dumps(payload, separators=(",", ":"))
assert "</" not in data_js
print("payload total:", len(data_js) // 1024, "KB")

FRAGMENT = r"""
<div class="wrap">
  <section>
    <div class="stepno">Step 9 &middot; in motion</div>
    <h2>Watch the clusters form</h2>
    <p>How do 192 random numbers per champion become the map above? Nothing places a champion. Training repeats one loop: show the model a partial draft, ask for the next pick, measure how much probability it put on the true answer, then nudge every number in the model &mdash; embedding slots included &mdash; a small step in the direction that would have reduced the miss. The loop never sees a role label.</p>
    <p>To film that loop, we retrained one seed on the same 2024&ndash;2026 data and split as the production run, with the production configuration, and saved the full embedding table at random start, mid-pass early on, and after every pass through the data. The grid below shows every pair of champions at once. Champions are sorted into five role blocks. Each cell answers one question for one pair: do these two champions&rsquo; 192 numbers point the same way (orange), opposite ways (blue), or neither (neutral)? At random start the grid is static. Press play and watch the five diagonal squares ignite &mdash; that is role structure crystallizing, in real similarity values, with no projection tricks in between.</p>
  </section>
</div>

<div class="panelwrap">
  <div class="panel" id="evo-panel" style="position:relative">
    <h3>Every champion pair, snapshot by snapshot</h3>
    <div class="evo-views" role="group" aria-label="View">
      <button id="evo-view-sim" type="button" class="on">Champion &times; champion similarity</button>
      <button id="evo-view-dim" type="button">Go deeper: the raw 192 numbers</button>
    </div>
    <p class="sub" id="evo-sub">Rows and columns are the same champions, sorted into role blocks (order fixed across all frames). Orange = this pair&rsquo;s numbers point the same way; blue = opposite; the diagonal is each champion with itself. Hover any cell for the pair and its value.</p>
    <canvas id="evo-matrix" style="width:100%;display:block"></canvas>
    <div class="evo-controls">
      <button id="evo-play" type="button">&#9654; Play</button>
      <input id="evo-scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Training snapshot">
      <span id="evo-label"></span>
    </div>
    <canvas id="evo-strip" style="width:100%;display:block"></canvas>
    <div id="evo-tip" role="status"></div>

    <h3 style="margin-top:26px">The actual numbers, tuning</h3>
    <p class="sub">The first 12 of each champion's 192 slots, in thousandths. Scrub the timeline and watch the values move. An arrow marks which way each value moved since the previous snapshot. No slot means anything by itself &mdash; what training shapes is how each champion's full list of 192 numbers <em>points</em> relative to the others. The bars measure that: cosine similarity, from &minus;1 (opposite) through 0 (unrelated) to +1 (same direction), across all 192 slots.</p>
    <div class="evo-numwrap"><table class="evo-num" id="evo-num"></table></div>
    <div class="evo-dist">
      <div><span class="evo-dist-pair" id="evo-d1-label"></span><div class="evo-bar"><span class="evo-bar-mid"></span><div id="evo-d1-fill"></div></div><span class="evo-dist-val" id="evo-d1-val"></span></div>
      <div><span class="evo-dist-pair" id="evo-d2-label"></span><div class="evo-bar"><span class="evo-bar-mid"></span><div id="evo-d2-fill"></div></div><span class="evo-dist-val" id="evo-d2-val"></span></div>
    </div>
    <p class="codecap"><b>Honestly labeled demo:</b> single seed (16), production config d192x4L6H, and the same multi-year 2024&ndash;2026 data and split as the production run (@@NTRAIN@@ training games) &mdash; but not the shipped 5-seed ensemble. The frozen EWC test set was never touched. 5-NN role purity climbs from @@P0@@ at random start (chance &asymp; 0.20) to @@PF@@. Similarity is measured after removing the shared average direction (every champion&rsquo;s numbers drift in one common direction as they grow; subtracting it exposes the contrasts), and cells are clipped at &plusmn;0.45 for contrast. Not every bright off-diagonal region is an error: Top &harr; Jungle stays warm because the two roles genuinely share a bruiser champion pool. In the raw-numbers view, columns are reordered so correlated dimensions sit together &mdash; the column order itself carries no meaning, because no single dimension does. And why not animate the t-SNE map above instead? Because t-SNE re-optimizes its layout every time: cluster sizes and distances between clusters are not stable quantities, so animating them shows layout noise, not learning. Every cell here is a real, stable number.</p>
  </div>
</div>

<style>
  .evo-views { display:flex; gap:8px; margin:2px 0 10px; flex-wrap:wrap; }
  .evo-views button {
    font: 600 12px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 11px; cursor: pointer;
  }
  .evo-views button.on { color: var(--ink); border-color: var(--model); }
  .evo-views button:focus-visible, .evo-controls button:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
  .evo-controls { display:flex; align-items:center; gap:12px; margin:10px 0 4px; }
  .evo-controls button {
    font: 600 13px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--ink); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 12px; cursor: pointer; white-space: nowrap;
  }
  .evo-controls input[type=range] { flex: 1; accent-color: var(--model); min-width: 0; }
  #evo-label {
    font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); min-width: 21ch; text-align: right;
  }
  #evo-tip {
    position: absolute; display: none; pointer-events: none;
    font: 600 12px/1 ui-monospace, "SF Mono", Menlo, monospace;
    background: var(--card); color: var(--ink);
    border: 1px solid var(--hairline); border-radius: 6px;
    padding: 5px 8px; box-shadow: var(--shadow); z-index: 5;
  }
  .evo-numwrap { overflow-x: auto; }
  table.evo-num {
    border-collapse: collapse; width: 100%; min-width: 640px;
    font: 500 11.5px/1.2 ui-monospace, "SF Mono", Menlo, monospace;
  }
  table.evo-num th {
    text-align: left; color: var(--muted); font-weight: 600;
    padding: 4px 8px 4px 0; white-space: nowrap;
  }
  table.evo-num td {
    text-align: right; padding: 4px 5px; min-width: 44px;
    border-top: 1px solid var(--hairline);
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  table.evo-num td .d { font-size: 9px; margin-left: 1px; }
  .evo-dist { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  .evo-dist > div { display: flex; align-items: center; gap: 10px; }
  .evo-dist-pair {
    font: 600 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); min-width: 30ch;
  }
  .evo-bar { flex: 1; height: 8px; background: var(--inset); border-radius: 4px; position: relative; overflow: hidden; }
  .evo-bar > div { height: 100%; border-radius: 4px; width: 0; position: absolute; top: 0; }
  .evo-bar-mid { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--faint); }
  @media (max-width: 560px) { #evo-label, .evo-dist-pair { display: none; } }
</style>

<script>
(function () {
  "use strict";
  var D = @@DATA@@;
  var ROLE_NAMES = { top: "Top", jng: "Jungle", mid: "Mid", bot: "Bot", sup: "Support" };
  var matrix = document.getElementById("evo-matrix");
  var strip = document.getElementById("evo-strip");
  var play = document.getElementById("evo-play");
  var scrub = document.getElementById("evo-scrub");
  var label = document.getElementById("evo-label");
  var tip = document.getElementById("evo-tip");
  var panel = document.getElementById("evo-panel");
  var subEl = document.getElementById("evo-sub");
  var btnSim = document.getElementById("evo-view-sim");
  var btnDim = document.getElementById("evo-view-dim");
  var numTable = document.getElementById("evo-num");
  var frames = D.frames, nK = frames.length, N = D.names.length, nD = D.nDims;
  scrub.max = String(nK - 1);

  var SUBS = {
    sim: "Rows and columns are the same champions, sorted into role blocks (order fixed across all frames). Orange = this pair’s numbers point the same way; blue = opposite; the diagonal is each champion with itself. Hover any cell for the pair and its value.",
    dim: "Now each row is one champion (same role-block order) and each column is one of its 192 raw numbers, orange positive, blue negative. Columns are reordered so correlated dimensions sit together — watch faint vertical bands emerge per role block. The column order itself means nothing; no single dimension does."
  };

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var darkMq = window.matchMedia("(prefers-color-scheme: dark)");
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return darkMq.matches;
  }

  // Decode keyframe PNGs into Image objects.
  var imgs = { sim: [], dim: [] };
  var loaded = 0, needed = nK * 2;
  frames.forEach(function (f, i) {
    ["sim", "dim"].forEach(function (kind) {
      var im = new Image();
      im.onload = function () { if (++loaded === needed) render(); };
      im.src = "data:image/webp;base64," + f[kind];
      imgs[kind][i] = im;
    });
  });

  var view = "sim";     // 'sim' | 'dim'
  var ki = 0;           // current keyframe index
  var playing = false, timer = null, hoverCell = null;
  var geo = null;       // last draw geometry for hover mapping

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function parseColor(c) {
    var m = c.match(/^#([0-9a-f]{6})$/i);
    if (!m) return [128, 128, 128];
    var v = parseInt(m[1], 16);
    return [v >> 16 & 255, v >> 8 & 255, v & 255];
  }
  function buildLUT() {
    var neg = parseColor(css("--model")), pos = parseColor(css("--meta"));
    var mid = parseColor(isDark() ? "#232D3D" : "#F4F2EC");
    var lut = new Uint8ClampedArray(256 * 3);
    for (var i = 0; i < 256; i++) {
      var t = i / 127.5 - 1, a = t < 0 ? neg : pos, w = Math.abs(t);
      w = Math.pow(w, 0.85); // slight boost so faint structure reads
      for (var c = 0; c < 3; c++) {
        lut[3 * i + c] = Math.round(mid[c] + (a[c] - mid[c]) * w);
      }
    }
    return lut;
  }

  function sizeCanvas(cv, hCss) {
    var w = cv.clientWidth || cv.parentNode.clientWidth;
    var dpr = window.devicePixelRatio || 1;
    cv.style.height = hCss + "px";
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(hCss * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(hCss * dpr);
    }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  var work = document.createElement("canvas");
  var pkc = document.createElement("canvas");
  pkc.width = 1; pkc.height = 1;
  var pk = pkc.getContext("2d", { willReadFrequently: true });
  function drawMatrix() {
    if (loaded < needed) return;
    var im = imgs[view][ki];
    var iw = im.naturalWidth, ih = im.naturalHeight;
    var marginL = matrix.clientWidth > 620 ? 64 : 0, marginT = 20;
    var availW = (matrix.clientWidth || 600) - marginL;
    var scale = availW / iw;
    var mh = Math.min(Math.round(ih * scale), 560);
    scale = Math.min(scale, mh / ih);
    var mw = Math.round(iw * scale), mhpx = Math.round(ih * scale);
    var ctx = sizeCanvas(matrix, mhpx + marginT);
    var w = matrix.clientWidth;
    ctx.clearRect(0, 0, w, mhpx + marginT);

    // colorize native-res PNG through the LUT
    work.width = iw; work.height = ih;
    var wctx = work.getContext("2d", { willReadFrequently: true });
    wctx.drawImage(im, 0, 0);
    var id = wctx.getImageData(0, 0, iw, ih), px = id.data, lut = buildLUT();
    for (var i = 0; i < px.length; i += 4) {
      var g = px[i];
      px[i] = lut[3 * g]; px[i + 1] = lut[3 * g + 1]; px[i + 2] = lut[3 * g + 2];
    }
    wctx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(work, marginL, marginT, mw, mhpx);
    geo = { marginL: marginL, marginT: marginT, scale: scale, iw: iw, ih: ih,
            mw: mw, mh: mhpx };

    // role-block boundaries + labels
    var ink = css("--ink"), mut = css("--muted"), hair = css("--hairline");
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    D.bounds.forEach(function (b) {
      var y0 = marginT + b.start * scale, y1 = marginT + b.end * scale;
      ctx.beginPath();
      ctx.moveTo(marginL, y1); ctx.lineTo(marginL + mw, y1); ctx.stroke();
      if (view === "sim") {
        var x1 = marginL + b.end * scale;
        ctx.beginPath();
        ctx.moveTo(x1, marginT); ctx.lineTo(x1, marginT + mhpx); ctx.stroke();
        // column label
        ctx.fillStyle = mut;
        var cx = marginL + (b.start + b.end) / 2 * scale;
        ctx.textAlign = "center";
        ctx.fillText(b.name, cx, 13);
      }
      if (marginL > 0) {
        ctx.fillStyle = mut; ctx.textAlign = "right";
        ctx.fillText(b.name, marginL - 6, (y0 + y1) / 2 + 3);
      }
    });
    if (view === "dim") {
      ctx.fillStyle = mut; ctx.textAlign = "left";
      ctx.fillText("192 dimensions → (columns reordered by similarity)", marginL, 13);
    }
    ctx.textAlign = "left";
    var f = frames[ki];
    label.textContent = f.l + " · loss " + f.v.toFixed(2) + " · purity " + f.p.toFixed(2);
    scrub.value = String(ki);
  }

  function drawStrip() {
    var h = 74;
    var ctx = sizeCanvas(strip, h);
    var w = strip.clientWidth;
    var gold = css("--gold"), model = css("--model"), hair = css("--hairline");
    ctx.clearRect(0, 0, w, h);
    var pad = 8, iw = w - 2 * pad, ih = h - 26, nF = D.loss.length;
    var vmin = Math.min.apply(null, D.loss), vmax = Math.max.apply(null, D.loss);
    function fx(i) { return pad + iw * i / (nF - 1); }
    function line(arr, lo, hi, color) {
      ctx.beginPath();
      for (var i = 0; i < nF; i++) {
        var yv = pad + ih - ih * (arr[i] - lo) / (hi - lo);
        i ? ctx.lineTo(fx(i), yv) : ctx.moveTo(fx(i), yv);
      }
      ctx.lineWidth = 1.8; ctx.strokeStyle = color; ctx.stroke();
    }
    line(D.loss, vmin, vmax, model);
    line(D.purity, 0.15, 0.75, gold);
    // keyframe ticks + playhead at the current keyframe's true frame index
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    frames.forEach(function (f) {
      ctx.beginPath();
      ctx.moveTo(fx(f.f), pad + ih - 3); ctx.lineTo(fx(f.f), pad + ih); ctx.stroke();
    });
    var phx = fx(frames[ki].f);
    ctx.beginPath(); ctx.moveTo(phx, pad); ctx.lineTo(phx, pad + ih); ctx.stroke();
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.fillStyle = model; ctx.fillText("val loss ↓", pad, h - 6);
    ctx.fillStyle = gold; ctx.fillText("role purity ↑ (never optimized)", pad + 76, h - 6);
  }

  // ---- raw numbers exhibit ----
  var ex = D.ex, nSlots = ex.dims[0][0].length;
  var cells = [];
  (function buildTable() {
    var rows = "<tr><th></th>";
    for (var d = 0; d < nSlots; d++) rows += "<th style='text-align:right'>s" + (d + 1) + "</th>";
    rows += "<th style='text-align:left;color:var(--faint)'>&hellip;+180</th></tr>";
    for (var r = 0; r < 3; r++) {
      rows += "<tr><th>" + ex.names[r] + " <span style='color:var(--faint)'>(" +
        (ROLE_NAMES[ex.roles[r]] || "?") + ")</span></th>";
      for (var d2 = 0; d2 < nSlots; d2++) rows += "<td id='evo-c" + r + "_" + d2 + "'></td>";
      rows += "<td></td></tr>";
    }
    numTable.innerHTML = rows;
    for (var r2 = 0; r2 < 3; r2++) {
      cells.push([]);
      for (var d3 = 0; d3 < nSlots; d3++) {
        cells[r2].push(document.getElementById("evo-c" + r2 + "_" + d3));
      }
    }
  })();
  document.getElementById("evo-d1-label").textContent =
    ex.names[0] + " ↔ " + ex.names[1] + " (both supports)";
  document.getElementById("evo-d2-label").textContent =
    ex.names[0] + " ↔ " + ex.names[2] + " (support vs mid)";

  function setBar(fill, valEl, v, posColor, negColor) {
    valEl.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
    var half = 50, wpct = Math.min(Math.abs(v), 1) * half;
    fill.style.width = wpct.toFixed(1) + "%";
    fill.style.left = (v >= 0 ? half : half - wpct).toFixed(1) + "%";
    fill.style.background = v >= 0 ? posColor : negColor;
  }

  function updateNumbers() {
    var model = css("--model"), metaC = css("--meta");
    for (var r = 0; r < 3; r++) {
      var cur = ex.dims[r][ki];
      var prev = ki > 0 ? ex.dims[r][ki - 1] : null;
      for (var d = 0; d < nSlots; d++) {
        var v = cur[d], el = cells[r][d], arrow = "";
        if (prev && v !== prev[d]) {
          arrow = v > prev[d]
            ? " <span class='d' style='color:" + metaC + "'>&#9650;</span>"
            : " <span class='d' style='color:" + model + "'>&#9660;</span>";
        } else if (prev) {
          arrow = " <span class='d' style='opacity:0'>&#9650;</span>";
        }
        el.innerHTML = v + arrow;
        var mag = Math.min(Math.abs(v) / 260, 1);
        el.style.background = v >= 0
          ? "rgba(196,118,59," + (0.06 + 0.3 * mag).toFixed(2) + ")"
          : "rgba(62,124,184," + (0.06 + 0.3 * mag).toFixed(2) + ")";
      }
    }
    var pair = ex.sim[ki];
    setBar(document.getElementById("evo-d1-fill"),
           document.getElementById("evo-d1-val"), pair[0], metaC, model);
    setBar(document.getElementById("evo-d2-fill"),
           document.getElementById("evo-d2-val"), pair[1], metaC, model);
  }

  function render() { drawMatrix(); drawStrip(); updateNumbers(); }

  function stop() {
    playing = false; play.innerHTML = "&#9654; Play";
    if (timer) { clearTimeout(timer); timer = null; }
  }
  function start() {
    if (ki >= nK - 1) ki = 0;
    playing = true; play.innerHTML = "&#10074;&#10074; Pause";
    var step = function () {
      if (!playing) return;
      ki = Math.min(ki + 1, nK - 1);
      render();
      if (ki >= nK - 1) { stop(); return; }
      timer = setTimeout(step, ki < 6 ? 900 : 550);
    };
    timer = setTimeout(step, 500);
  }

  play.addEventListener("click", function () { playing ? stop() : start(); });
  scrub.addEventListener("input", function () {
    stop(); ki = Number(scrub.value); render();
  });

  function setView(v) {
    view = v;
    btnSim.classList.toggle("on", v === "sim");
    btnDim.classList.toggle("on", v === "dim");
    subEl.textContent = SUBS[v];
    hoverCell = null; tip.style.display = "none";
    drawMatrix();
  }
  btnSim.addEventListener("click", function () { setView("sim"); });
  btnDim.addEventListener("click", function () { setView("dim"); });

  matrix.addEventListener("mousemove", function (e) {
    if (!geo) return;
    var r = matrix.getBoundingClientRect();
    var cx = (e.clientX - r.left - geo.marginL) / geo.scale;
    var cy = (e.clientY - r.top - geo.marginT) / geo.scale;
    var col = Math.floor(cx), row = Math.floor(cy);
    if (col < 0 || row < 0 || col >= geo.iw || row >= geo.ih) {
      hoverCell = null; tip.style.display = "none"; return;
    }
    var key = row + "_" + col;
    if (hoverCell === key) return;
    hoverCell = key;
    var txt;
    if (view === "sim") {
      txt = D.names[row] + " × " + D.names[col];
      // read the untinted grayscale PNG to recover the similarity value
      pk.drawImage(imgs.sim[ki], col, row, 1, 1, 0, 0, 1, 1);
      var gg = pk.getImageData(0, 0, 1, 1).data[0];
      var val = (gg / 127.5 - 1) * D.simClip;
      txt += " ≈ " + (val >= 0 ? "+" : "") + val.toFixed(2);
    } else {
      txt = D.names[row] + " · one of its 192 numbers";
    }
    tip.textContent = txt;
    tip.style.display = "block";
    var tx = Math.min(e.clientX - r.left + 14, r.width - 170);
    tip.style.left = tx + "px";
    tip.style.top = (matrix.offsetTop + (e.clientY - r.top) - 30) + "px";
  });
  matrix.addEventListener("mouseleave", function () {
    hoverCell = null; tip.style.display = "none";
  });

  darkMq.addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });
  window.addEventListener("resize", render);

  var played = false;
  if (!reduced && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (en) {
        if (en.isIntersecting && !played && loaded === needed) {
          played = true; start(); obs.disconnect();
        }
      });
    }, { threshold: 0.4 }).observe(panel);
  }

  render();
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
print("wrote", DST, len(s), "chars")
