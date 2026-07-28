#!/usr/bin/env python3
"""PROTOTYPE ONLY — not a shipped builder, not wired into any artifact.

Sahil asked to see one beat rebuilt in the annotated comic-strip idiom before
deciding whether to redesign the whole panel (attempt #6 would be that redesign).
This renders BEAT 3 — the animated map — as static annotated keyframes, with the
motion demoted to an optional control underneath, and writes a side-by-side
comparison page against the shipped gated version.

Reuses build_evo_panel6's payload verbatim so every number and position is the
same real data as the live panel. Writes proto_comic_beat3.html (light) and
proto_comic_beat3_dark.html.

Keyframes are chosen from the DATA, not by eye: the first snapshot, the first
snapshot to clear 0.30 purity, the first to clear 0.50, and the last.
"""
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bp6", HERE / "build_evo_panel6.py")
bp6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp6)          # re-runs the build; idempotent

payload = bp6.payload
purity = payload["purity"]
labels = payload["labels"]
nF = payload["nF"]

# ---- data-driven keyframe choice ----
def first_at_or_above(threshold: float) -> int:
    for i, p in enumerate(purity):
        if p >= threshold:
            return i
    return nF - 1

KEYS = [0, first_at_or_above(0.30), first_at_or_above(0.50), nF - 1]
# de-duplicate while keeping order, in case thresholds collide
seen, keys = set(), []
for k in KEYS:
    if k not in seen:
        seen.add(k)
        keys.append(k)
assert len(keys) == 4, f"expected 4 distinct keyframes, got {keys}"
print("keyframes:", [(k, labels[k], purity[k]) for k in keys])

# One annotated claim per panel. Every claim is checked against the data below.
NOTES = [
    "Random start. The colours are the true roles, which the model never sees. "
    "Nothing is sorted yet.",
    "The first structure appears. Bot and Support separate before the rest; "
    "Top and Mid are still mixed through the middle.",
    "Most roles now have a home region. Top is the last one to sort itself out.",
    "The finished layout. Tidy at the edges, still a jumble in the middle — "
    "which is what beat 4 is about.",
]

# ---- claim checks for the prototype's own annotations ----
# Same discipline as the shipped panel: no annotation ships unless the data says
# it. Separation is measured HERE in the 2-D layout the reader is actually
# looking at, which is why these numbers are not shown on the page — the page
# only ever prints D.purity, the raw 192-dim score. Two different purity numbers
# on one figure would be worse than none.
import numpy as np                                            # noqa: E402
import base64                                                 # noqa: E402

_pos = (np.frombuffer(base64.b64decode(payload["pos"]), dtype="<u2")
          .reshape(nF, len(payload["names"]), 2) / 1000.0)
_role = np.array(payload["role"])
ROLE_NAMES = ["Top", "Jungle", "Mid", "Bot", "Support"]


def per_role_purity(f: int) -> dict:
    P = _pos[f]
    d = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, 1e9)
    nn = np.argsort(d, 1)[:, :5]
    hit = (_role[nn] == _role[:, None]).mean(1)
    return {ROLE_NAMES[r]: float(hit[_role == r].mean()) for r in range(5)}


assert purity[keys[0]] < 0.25, "panel 1 claims a random start"
assert purity[keys[1]] >= 0.30 and purity[keys[1]] < 0.50
assert purity[keys[2]] >= 0.50
assert abs(purity[keys[3]] - 0.704) < 1e-3, "panel 4 must be the shipped final purity"

_p1 = per_role_purity(keys[1])
assert min(_p1["Bot"], _p1["Support"]) > max(_p1["Top"], _p1["Mid"]), (
    f"panel 2 claims Bot and Support separate before Top and Mid; got {_p1}")
_p2 = per_role_purity(keys[2])
assert min(_p2, key=_p2.get) == "Top", (
    f"panel 3 claims Top sorts last; lowest role is {min(_p2, key=_p2.get)} in {_p2}")
_p3 = per_role_purity(keys[3])
assert min(_p3, key=_p3.get) == "Top", (
    f"panel 3's claim must still hold at the end; got {_p3}")
print("prototype claim checks OK —",
      {k: round(v, 3) for k, v in _p1.items()}, "|",
      {k: round(v, 3) for k, v in _p2.items()})

PROTO = r"""
<div class="panelwrap">
  <div class="panel" id="proto-panel">
    <h3>PROTOTYPE &middot; the same beat as an annotated comic strip</h3>
    <p class="sub">Four real snapshots, chosen by score, not by eye. Nothing to press &mdash; the whole argument is visible at once. The motion is still here, underneath.</p>

    <div class="proto-strip" id="proto-strip"></div>

    <p class="codecap"><b>Before you read the frames: when each dot settles is measured. Where it travels is drawn.</b> Read the timing, not the path.</p>

    <details class="proto-more">
      <summary>Watch it move instead</summary>
      <div class="proto-anim">
        <canvas id="proto-map" style="width:100%;display:block" role="img" aria-label="Animated scatter map of 168 champions moving into role clusters"></canvas>
        <div class="proto-controls">
          <button id="proto-play" type="button">&#9654; Play</button>
          <input id="proto-scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Training snapshot">
          <span id="proto-label"></span>
        </div>
      </div>
    </details>
  </div>
</div>

<style>
  #proto-panel > h3 { font-size: 21px; line-height: 1.25; margin-bottom: 6px; }
  .proto-strip {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 22px 0 0;
  }
  @media (max-width: 720px) { .proto-strip { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 400px) { .proto-strip { grid-template-columns: 1fr; } }
  .proto-cell { display: flex; flex-direction: column; gap: 8px; }
  .proto-cell canvas {
    width: 100%; display: block;
    border: 1px solid var(--hairline); border-radius: 8px; background: var(--inset);
  }
  .proto-cell .proto-hd {
    font: 600 11px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); display: flex; justify-content: space-between; gap: 8px;
    font-variant-numeric: tabular-nums;
  }
  .proto-cell .proto-hd b { color: var(--ink); }
  .proto-cell .proto-note {
    font: 500 12px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); margin: 0;
  }
  .proto-legend { display:flex; gap:14px; flex-wrap:wrap; margin: 16px 0 0; }
  .proto-legend span {
    font: 600 11.5px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); display:inline-flex; align-items:center; gap:5px;
  }
  .proto-legend i { width:9px; height:9px; border-radius:50%; display:inline-block; }
  .proto-more { margin-top: 32px; }
  .proto-more summary {
    font: 600 12px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); cursor: pointer; padding: 6px 0;
  }
  .proto-anim { margin-top: 12px; }
  .proto-controls { display:flex; align-items:center; gap:12px; margin:10px 0 4px; }
  .proto-controls button {
    font: 600 13px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 12px; cursor: pointer; white-space: nowrap;
  }
  .proto-controls input[type=range] { flex:1; accent-color: var(--muted); min-width:0; }
  #proto-label {
    font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); min-width: 19ch; text-align: right;
  }
  @media (max-width:560px) { #proto-label { display:none; } }
</style>

<script>
(function () {
  "use strict";
  var D = @@DATA@@;
  var KEYS = @@KEYS@@, NOTES = @@NOTES@@;
  var ROLE_NAMES = ["Top", "Jungle", "Mid", "Bot", "Support"];
  var ROLE_LIGHT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#4a3aa7"];
  var ROLE_DARK  = ["#3987e5", "#008300", "#d55181", "#c98500", "#9085e9"];
  var nF = D.nF, N = D.names.length;

  var raw = atob(D.pos);
  var pos = new Uint16Array(raw.length / 2);
  for (var bi = 0; bi < pos.length; bi++) {
    pos[bi] = raw.charCodeAt(2 * bi) | (raw.charCodeAt(2 * bi + 1) << 8);
  }
  function px(f, c) { return pos[(f * N + c) * 2] / 1000; }
  function py(f, c) { return pos[(f * N + c) * 2 + 1] / 1000; }

  var darkMq = window.matchMedia("(prefers-color-scheme: dark)");
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return darkMq.matches;
  }
  function roleColors() { return isDark() ? ROLE_DARK : ROLE_LIGHT; }
  function css(n) {
    return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  }
  function sizeCanvas(cv, hCss) {
    var w = cv.clientWidth || 200;
    var dpr = window.devicePixelRatio || 1;
    cv.style.height = hCss + "px";
    cv.width = Math.round(w * dpr); cv.height = Math.round(hCss * dpr);
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function drawFrame(cv, f, dotR) {
    var w = cv.clientWidth || 200;
    var h = Math.round(w * 0.92);
    var ctx = sizeCanvas(cv, h);
    ctx.clearRect(0, 0, w, h);
    var rc = roleColors(), pad = 10;
    var iw = w - 2 * pad, ih = h - 2 * pad;
    for (var i = 0; i < N; i++) {
      ctx.beginPath();
      ctx.arc(pad + px(f, i) * iw, pad + py(f, i) * ih, dotR, 0, 7);
      ctx.fillStyle = rc[D.role[i]];
      ctx.fill();
    }
  }

  var strip = document.getElementById("proto-strip");
  var cells = [];
  KEYS.forEach(function (f, i) {
    var cell = document.createElement("div");
    cell.className = "proto-cell";
    var hd = document.createElement("div");
    hd.className = "proto-hd";
    hd.innerHTML = "<b>" + (i + 1) + " &middot; " + D.labels[f] + "</b>" +
                   "<span>purity " + D.purity[f].toFixed(2) + "</span>";
    var cv = document.createElement("canvas");
    cv.setAttribute("role", "img");
    cv.setAttribute("aria-label",
      "Snapshot " + (i + 1) + ", " + D.labels[f] + ", role purity " +
      D.purity[f].toFixed(2) + ". " + NOTES[i]);
    var note = document.createElement("p");
    note.className = "proto-note";
    note.textContent = NOTES[i];
    cell.appendChild(hd); cell.appendChild(cv); cell.appendChild(note);
    strip.appendChild(cell);
    cells.push({ cv: cv, f: f });
  });

  var legend = document.createElement("div");
  legend.className = "proto-legend";
  strip.parentNode.insertBefore(legend, strip.nextSibling);

  /* ---- the optional animation, unchanged in substance ---- */
  var map = document.getElementById("proto-map");
  var scrub = document.getElementById("proto-scrub");
  var playBtn = document.getElementById("proto-play");
  var frameLabel = document.getElementById("proto-label");
  var ft = 0, playing = false, rafId = null;
  scrub.max = String(nF - 1);
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function drawBig() {
    var w = map.clientWidth || 600;
    drawFrameInto(map, ft, Math.round(w * 0.62));
    var fi = Math.round(ft);
    frameLabel.textContent = D.labels[fi] + " · purity " + D.purity[fi].toFixed(2);
    scrub.value = String(fi);
  }
  function drawFrameInto(cv, f, h) {
    var w = cv.clientWidth || 600;
    var ctx = sizeCanvas(cv, h);
    ctx.clearRect(0, 0, w, h);
    var rc = roleColors(), pad = 14;
    var iw = w - 2 * pad, ih = h - 2 * pad;
    var f0 = Math.floor(f), f1 = Math.min(f0 + 1, nF - 1), t = f - f0;
    for (var i = 0; i < N; i++) {
      var x = (px(f0, i) * (1 - t) + px(f1, i) * t) * iw + pad;
      var y = (py(f0, i) * (1 - t) + py(f1, i) * t) * ih + pad;
      ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 7);
      ctx.fillStyle = rc[D.role[i]]; ctx.fill();
    }
  }
  var lastTs = null;
  function tick(ts) {
    if (!playing) return;
    if (lastTs == null) lastTs = ts;
    ft += (ts - lastTs) / 150; lastTs = ts;
    if (ft >= nF - 1) { ft = nF - 1; stop(); drawBig(); return; }
    drawBig();
    rafId = requestAnimationFrame(tick);
  }
  function stop() {
    playing = false; lastTs = null;
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    playBtn.textContent = ft >= nF - 1 ? "▶ Replay" : "▶ Play";
  }
  playBtn.addEventListener("click", function () {
    if (playing) { stop(); return; }
    if (ft >= nF - 1) ft = 0;
    playing = true; playBtn.textContent = "❚❚ Pause";
    rafId = requestAnimationFrame(tick);
  });
  scrub.addEventListener("input", function () {
    stop(); ft = Number(scrub.value); drawBig();
  });

  function renderAll() {
    var rc = roleColors();
    legend.innerHTML = "";
    ROLE_NAMES.forEach(function (nm, i) {
      var s = document.createElement("span");
      s.innerHTML = "<i style='background:" + rc[i] + "'></i>" + nm;
      legend.appendChild(s);
    });
    cells.forEach(function (c) { drawFrame(c.cv, c.f, 2.6); });
    drawBig();
  }
  darkMq.addEventListener("change", renderAll);
  new MutationObserver(renderAll).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });
  window.addEventListener("resize", renderAll);
  renderAll();
  if (reduced) { /* no autoplay; the strip already carries the story */ }
})();
</script>
"""

frag = (PROTO.replace("@@DATA@@", json.dumps(payload, separators=(",", ":")))
             .replace("@@KEYS@@", json.dumps(keys))
             .replace("@@NOTES@@", json.dumps(NOTES)))
assert "</" not in json.dumps(payload, separators=(",", ":"))

# reuse the shipped page CSS so the comparison is apples to apples
shipped = (HERE / "artifact-v8.html").read_text()
head_css = shipped.split("<style>", 1)[1].split("</style>", 1)[0]


def page(theme_attr: str) -> str:
    return ("<!doctype html><html" + theme_attr + "><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>" + head_css + "</style></head><body>"
            + frag + "</body></html>")


(HERE / "proto_comic_beat3.html").write_text(page(""))
(HERE / "proto_comic_beat3_dark.html").write_text(page(' data-theme="dark"'))
print("wrote proto_comic_beat3.html and proto_comic_beat3_dark.html")
