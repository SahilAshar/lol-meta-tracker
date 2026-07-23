"""v2 of the evolution panel: per-frame t-SNE layouts (warm-started neighbor to
neighbor, Procrustes-aligned) instead of the failed flat PCA projection, plus a
raw-numbers exhibit showing actual embedding values tuning between snapshots.

Splices into the pristine artifact-v2.html (no old panel) -> artifact-v4.html.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE

SCRATCH = Path(__file__).parent
REPO = Path.home() / "Documents/repos/lol-meta-tracker"
SRC = SCRATCH / "artifact-v2.html"
DST = SCRATCH / "artifact-v4.html"

meta = json.loads(
    (REPO / "data/processed/embedding_evolution_v08_demo.json").read_text())
z = np.load(REPO / "data/processed/embedding_evolution_v08_snapshots.npz",
            allow_pickle=False)
embs = z["embs"]            # (F, C, 192)
champs = [str(c) for c in z["champs"]]
role = [str(r) for r in z["role"]]
flex = z["flex"].astype(bool)
labels = [str(l) for l in z["labels"]]
val_loss = z["val_loss"]
purity = z["purity"]
F, C, D = embs.shape
print(f"{F} frames x {C} champs x {D} dims")

# ---- t-SNE chain: final frame first, then walk backward warm-starting each
# frame from its successor so the camera never jumps.
def norm(xy):
    xy = xy - xy.mean(0)
    return xy / np.sqrt((xy ** 2).mean())

layouts = [None] * F
t = TSNE(n_components=2, perplexity=30, init="pca", random_state=16,
         learning_rate="auto")
layouts[F - 1] = norm(t.fit_transform(embs[F - 1]))
for i in range(F - 2, -1, -1):
    t = TSNE(n_components=2, perplexity=30, init=np.asarray(layouts[i + 1], dtype=np.float32),
             random_state=16, learning_rate="auto", early_exaggeration=4)
    xy = norm(t.fit_transform(embs[i]))
    R, _ = orthogonal_procrustes(xy, layouts[i + 1])
    layouts[i] = xy @ R
    if i % 10 == 0:
        print("tsne frame", i)

np.savez(SCRATCH / "evo_layouts.npz", layouts=np.stack(layouts))

# ---- raw-numbers exhibit: two same-role champions + one different-role.
def pick(*names):
    for n in names:
        if n in champs:
            return champs.index(n)
    raise KeyError(names)

i1, i2, i3 = pick("Rakan", "Leona"), pick("Alistar", "Braum"), pick("Orianna", "Ahri")
SHOW_DIMS = 12

def cos(f, a, b):
    x, y = embs[f, a], embs[f, b]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

exhibit = {
    "names": [champs[i1], champs[i2], champs[i3]],
    "roles": [role[i1], role[i2], role[i3]],
    "dims": [[np.round(embs[f, i, :SHOW_DIMS] * 1000).astype(int).tolist()
              for f in range(F)] for i in (i1, i2, i3)],
    "sim": [[round(cos(f, i1, i2), 3), round(cos(f, i1, i3), 3)]
            for f in range(F)],
}

# ---- canvas labels: role anchors first, then flex picks.
anchor_names = ["Nautilus", "Rakan", "Braum", "Orianna", "Ahri", "Aatrox",
                "K'Sante", "Lee Sin", "Viego", "Jinx", "Kai'Sa", "Ezreal"]
label_idx = [champs.index(n) for n in anchor_names if n in champs]
label_idx += [i for i in range(C) if flex[i] and i not in label_idx]

payload = {
    "champs": champs, "role": role, "flex": [int(b) for b in flex],
    "labelIdx": label_idx,
    "ex": exhibit,
    "frames": [
        {"l": labels[f], "v": round(float(val_loss[f]), 3),
         "p": round(float(purity[f]), 3),
         "c": [int(round(v * 1000)) for xy in layouts[f] for v in xy]}
        for f in range(F)
    ],
}
data_js = json.dumps(payload, separators=(",", ":"))
assert "</" not in data_js
print("payload bytes:", len(data_js))

FRAGMENT = r"""
<div class="wrap">
  <section>
    <div class="stepno">Step 9 &middot; in motion</div>
    <h2>Watch the clusters form</h2>
    <p>How do 192 random numbers per champion become the map above? Nothing places a champion. Training repeats one loop: show the model a partial draft, ask for the next pick, measure how much probability it put on the true answer, then nudge every number in the model &mdash; embedding slots included &mdash; a small step in the direction that would have reduced the miss. The loop never sees a role label.</p>
    <p>To film that loop, we retrained one seed on the same 2024&ndash;2026 data and split as the production run, with the production configuration, and saved the full embedding table @@NF@@ times: at random start, after every pass through the data, and mid-pass during the first two passes, where most of the movement happens. Each frame below is a t-SNE map of one snapshot &mdash; the same kind of map as the chart above. Consecutive frames share a warm start and are rotation-aligned, so the view stays steady while the champions move.</p>
  </section>
</div>

<div class="panelwrap">
  <div class="panel" id="evo-panel" style="position:relative">
    <h3>One seed learning, snapshot by snapshot</h3>
    <p class="sub">Each dot is a champion. Color = its majority role in pro play &mdash; shown to you, never to the model. Rings = flex picks. Hover any dot for its name. The strip tracks validation loss (the only thing training optimizes) against role purity (which nobody asked for).</p>
    <canvas id="evo-scatter" style="width:100%;display:block"></canvas>
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
    <p class="codecap"><b>Honestly labeled demo:</b> single seed (16), production config d192x4L6H, and the same multi-year 2024&ndash;2026 data and split as the production run (@@NTRAIN@@ training games) &mdash; but not the shipped 5-seed ensemble, whose averaged embeddings score 0.682 on role purity. The frozen EWC test set was never touched. Role purity here climbs from @@P0@@ at random start (chance &asymp; 0.20) to @@PF@@. One caution about the picture: t-SNE preserves neighborhoods, not raw distances, so trust which champions sit together, not how far apart the groups look. The raw values above are the ground truth.</p>
  </div>
</div>

<style>
  .evo-controls { display:flex; align-items:center; gap:12px; margin:10px 0 4px; }
  .evo-controls button {
    font: 600 13px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--ink); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 12px; cursor: pointer; white-space: nowrap;
  }
  .evo-controls button:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
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
  .evo-dist-val {
    font: 700 13px/1 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--ink); min-width: 5ch; text-align: right;
    font-variant-numeric: tabular-nums;
  }
  @media (max-width: 560px) { #evo-label, .evo-dist-pair { display: none; } }
</style>

<script>
(function () {
  "use strict";
  var D = @@DATA@@;
  var ROLE_COLORS = {
    light: { top: "#2a78d6", jng: "#008300", mid: "#e87ba4", bot: "#eda100", sup: "#4a3aa7" },
    dark:  { top: "#3987e5", jng: "#008300", mid: "#d55181", bot: "#c98500", sup: "#9085e9" }
  };
  var ROLE_NAMES = { top: "Top", jng: "Jungle", mid: "Mid", bot: "Bot", sup: "Support" };
  var scatter = document.getElementById("evo-scatter");
  var strip = document.getElementById("evo-strip");
  var play = document.getElementById("evo-play");
  var scrub = document.getElementById("evo-scrub");
  var label = document.getElementById("evo-label");
  var tip = document.getElementById("evo-tip");
  var panel = document.getElementById("evo-panel");
  var numTable = document.getElementById("evo-num");
  var frames = D.frames, nF = frames.length, nC = D.champs.length;
  scrub.max = String(nF - 1);

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var darkMq = window.matchMedia("(prefers-color-scheme: dark)");
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return darkMq.matches;
  }

  var xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
  frames.forEach(function (f) {
    for (var i = 0; i < nC; i++) {
      var x = f.c[2 * i], y = f.c[2 * i + 1];
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
  });
  var xr = xmax - xmin, yr = ymax - ymin;
  xmin -= xr * 0.06; xmax += xr * 0.06; ymin -= yr * 0.06; ymax += yr * 0.06;

  var pos = 0, playing = false, raf = null, lastT = 0, hoverIdx = -1;
  var lastIntFrame = -1;
  var px = new Float32Array(nC), py = new Float32Array(nC);

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

  function frameDur(i) { return i < 10 ? 430 : 165; }

  function lerpCoord(i, k) {
    var a = Math.floor(pos), t = pos - a, b = Math.min(a + 1, nF - 1);
    return frames[a].c[2 * i + k] + (frames[b].c[2 * i + k] - frames[a].c[2 * i + k]) * t;
  }

  function drawScatter() {
    var h = Math.max(320, Math.min(480, scatter.clientWidth * 0.66));
    var ctx = sizeCanvas(scatter, h);
    var w = scatter.clientWidth;
    var pal = ROLE_COLORS[isDark() ? "dark" : "light"];
    var css = getComputedStyle(document.documentElement);
    var ink = css.getPropertyValue("--ink").trim();
    var mut = css.getPropertyValue("--muted").trim();
    var inset = css.getPropertyValue("--inset").trim();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = inset;
    ctx.fillRect(0, 0, w, h);
    for (var i = 0; i < nC; i++) {
      var x = (lerpCoord(i, 0) - xmin) / (xmax - xmin) * w;
      var y = h - (lerpCoord(i, 1) - ymin) / (ymax - ymin) * h;
      px[i] = x; py[i] = y;
      ctx.beginPath();
      ctx.arc(x, y, i === hoverIdx ? 6.5 : 4.5, 0, 6.2832);
      ctx.fillStyle = pal[D.role[i]] || "#888";
      ctx.globalAlpha = 0.92;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (D.flex[i] || i === hoverIdx) {
        ctx.lineWidth = i === hoverIdx ? 2 : 1.2;
        ctx.strokeStyle = ink;
        ctx.stroke();
      }
    }
    // labels: greedy, skip when close to an already-labeled point
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = mut;
    ctx.globalAlpha = playing ? 0.55 : 0.95;
    var placed = [];
    for (var j = 0; j < D.labelIdx.length; j++) {
      var k = D.labelIdx[j], ok = true;
      for (var q = 0; q < placed.length; q++) {
        var dx = px[k] - placed[q][0], dy = py[k] - placed[q][1];
        if (dx * dx + dy * dy < 900) { ok = false; break; }
      }
      if (!ok) continue;
      ctx.fillText(D.champs[k], px[k] + 7, py[k] + 3.5);
      placed.push([px[k], py[k]]);
    }
    ctx.globalAlpha = 1;
    var fi = Math.round(pos), f = frames[fi];
    label.textContent = f.l + " · loss " + f.v.toFixed(2) + " · purity " + f.p.toFixed(2);
    scrub.value = String(fi);
    if (fi !== lastIntFrame) { lastIntFrame = fi; updateNumbers(fi); }
  }

  function drawStrip() {
    var h = 74;
    var ctx = sizeCanvas(strip, h);
    var w = strip.clientWidth;
    var css = getComputedStyle(document.documentElement);
    var gold = css.getPropertyValue("--gold").trim();
    var model = css.getPropertyValue("--model").trim();
    var hair = css.getPropertyValue("--hairline").trim();
    ctx.clearRect(0, 0, w, h);
    var pad = 8, iw = w - 2 * pad, ih = h - 26;
    var vmin = 1e9, vmax = -1e9;
    frames.forEach(function (f) {
      if (f.v < vmin) vmin = f.v; if (f.v > vmax) vmax = f.v;
    });
    function fx(i) { return pad + iw * i / (nF - 1); }
    function line(get, lo, hi, color) {
      ctx.beginPath();
      for (var i = 0; i < nF; i++) {
        var yv = pad + ih - ih * (get(frames[i]) - lo) / (hi - lo);
        i ? ctx.lineTo(fx(i), yv) : ctx.moveTo(fx(i), yv);
      }
      ctx.lineWidth = 1.8; ctx.strokeStyle = color; ctx.stroke();
    }
    line(function (f) { return f.v; }, vmin, vmax, model);
    line(function (f) { return f.p; }, 0.15, 0.6, gold);
    ctx.strokeStyle = hair; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(fx(pos), pad); ctx.lineTo(fx(pos), pad + ih); ctx.stroke();
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.fillStyle = model; ctx.fillText("val loss ↓", pad, h - 6);
    ctx.fillStyle = gold; ctx.fillText("role purity ↑ (never optimized)", pad + 76, h - 6);
  }

  // ---- raw numbers exhibit ----
  var ex = D.ex, nD = ex.dims[0][0].length;
  var cells = [];
  (function buildTable() {
    var thead = "<tr><th></th>";
    for (var d = 0; d < nD; d++) thead += "<th style='text-align:right'>s" + (d + 1) + "</th>";
    thead += "<th style='text-align:left;color:var(--faint)'>&hellip;+180</th></tr>";
    var rows = thead;
    for (var r = 0; r < 3; r++) {
      rows += "<tr><th>" + ex.names[r] + " <span style='color:var(--faint)'>(" +
        (ROLE_NAMES[ex.roles[r]] || "?") + ")</span></th>";
      for (var d2 = 0; d2 < nD; d2++) {
        rows += "<td id='evo-c" + r + "_" + d2 + "'></td>";
      }
      rows += "<td></td></tr>";
    }
    numTable.innerHTML = rows;
    for (var r2 = 0; r2 < 3; r2++) {
      cells.push([]);
      for (var d3 = 0; d3 < nD; d3++) {
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

  function updateNumbers(fi) {
    var css = getComputedStyle(document.documentElement);
    var model = css.getPropertyValue("--model").trim();
    var meta = css.getPropertyValue("--meta").trim();
    for (var r = 0; r < 3; r++) {
      var cur = ex.dims[r][fi];
      var prev = fi > 0 ? ex.dims[r][fi - 1] : null;
      for (var d = 0; d < nD; d++) {
        var v = cur[d], el = cells[r][d];
        var arrow = "";
        if (prev && v !== prev[d]) {
          arrow = v > prev[d]
            ? " <span class='d' style='color:" + meta + "'>&#9650;</span>"
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
    var pair = ex.sim[fi];
    setBar(document.getElementById("evo-d1-fill"),
           document.getElementById("evo-d1-val"), pair[0], meta, model);
    setBar(document.getElementById("evo-d2-fill"),
           document.getElementById("evo-d2-val"), pair[1], meta, model);
  }

  function render() { drawScatter(); drawStrip(); }

  function stop() {
    playing = false; play.innerHTML = "&#9654; Play";
    if (raf) { cancelAnimationFrame(raf); clearTimeout(raf); raf = null; }
    render();
  }

  function tick(t) {
    if (!playing) return;
    if (!lastT) lastT = t;
    pos += (t - lastT) / frameDur(Math.floor(pos));
    lastT = t;
    if (pos >= nF - 1) { pos = nF - 1; render(); stop(); return; }
    render();
    raf = requestAnimationFrame(tick);
  }

  function start() {
    if (pos >= nF - 1) pos = 0;
    playing = true; lastT = 0; play.innerHTML = "&#10074;&#10074; Pause";
    if (reduced) {
      var step = function () {
        if (!playing) return;
        pos = Math.min(Math.floor(pos) + 1, nF - 1);
        render();
        if (pos >= nF - 1) { stop(); return; }
        raf = window.setTimeout(step, 650);
      };
      raf = window.setTimeout(step, 650);
      return;
    }
    raf = requestAnimationFrame(tick);
  }

  play.addEventListener("click", function () { playing ? stop() : start(); });
  scrub.addEventListener("input", function () {
    playing = false; play.innerHTML = "&#9654; Play";
    if (raf) { cancelAnimationFrame(raf); clearTimeout(raf); raf = null; }
    pos = Number(scrub.value); render();
  });

  scatter.addEventListener("mousemove", function (e) {
    var r = scatter.getBoundingClientRect();
    var mx = e.clientX - r.left, my = e.clientY - r.top;
    var best = -1, bd = 121;
    for (var i = 0; i < nC; i++) {
      var d2 = (px[i] - mx) * (px[i] - mx) + (py[i] - my) * (py[i] - my);
      if (d2 < bd) { bd = d2; best = i; }
    }
    if (best !== hoverIdx) {
      hoverIdx = best;
      if (best >= 0) {
        tip.textContent = D.champs[best] + " · " + (ROLE_NAMES[D.role[best]] || "?") +
          (D.flex[best] ? " · flex" : "");
        tip.style.display = "block";
        tip.style.left = Math.min(px[best] + 14, r.width - 130) + "px";
        tip.style.top = (scatter.offsetTop + py[best] - 30) + "px";
      } else {
        tip.style.display = "none";
      }
      if (!playing) drawScatter();
    }
  });
  scatter.addEventListener("mouseleave", function () {
    hoverIdx = -1; tip.style.display = "none";
    if (!playing) drawScatter();
  });

  darkMq.addEventListener("change", function () { lastIntFrame = -1; render(); });
  new MutationObserver(function () { lastIntFrame = -1; render(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.addEventListener("resize", render);

  var played = false;
  if (!reduced && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (en) {
        if (en.isIntersecting && !played) { played = true; start(); obs.disconnect(); }
      });
    }, { threshold: 0.45 }).observe(panel);
  }

  render();
})();
</script>
"""

fragment = (FRAGMENT
            .replace("@@DATA@@", data_js)
            .replace("@@NF@@", str(F))
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

for tag in ("div", "section", "canvas", "script", "style", "table"):
    print(tag, s.count("<" + tag), s.count("</" + tag + ">"))

DST.write_text(s)
print("wrote", DST, len(s), "chars")
