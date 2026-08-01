#!/usr/bin/env python3
"""PROTOTYPE v2 — MOCK ONLY. Not a shipped builder, publishes nowhere.

Sahil's direction after seeing prototype v1:
  - keep the comic strip, the captions are what make it work
  - 2x2 grid, not 1x4, with the animation BELOW the grid (not hidden)
  - highlight a full real team comp, not a single dot
  - audience is League-literate and pro-watching, but does NOT know ML;
    the job is to bridge "what I see in a draft on TV" to "what the model sees"

The comp is Hanwha Life Esports, GAME 3 of the MSI 2026 final (2026-07-12):
Olaf / Xin Zhao / Ryze / Xayah / Rakan — one champion for each of the five jobs.

EVERY claim below is asserted against the data before the page is written.
Two claims Sahil suggested were tested and REJECTED, and are deliberately absent:
  * "the five land in five different role regions" — only 3 of 5 do; the best
    comp in the entire final manages 4 of 5.
  * "the comp fans out as it trains" — mean pairwise distance among the five
    goes 0.509 -> 0.429, i.e. very slightly CLOSER, not further apart.
"""
import base64
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bp6", HERE / "build_evo_panel6.py")
bp6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp6)

payload = bp6.payload
names = payload["names"]
idx = {n: i for i, n in enumerate(names)}
role = np.array(payload["role"])
purity = payload["purity"]
labels = payload["labels"]
nF = payload["nF"]
pos = (np.frombuffer(base64.b64decode(payload["pos"]), dtype="<u2")
       .reshape(nF, len(names), 2) / 1000.0)
ROLE_NAMES = ["Top", "Jungle", "Mid", "Bot", "Support"]

# ---------------------------------------------------------------- the comp ---
COMP = ["Olaf", "Xin Zhao", "Ryze", "Xayah", "Rakan"]
GAME = dict(team="Hanwha Life Esports", event="MSI 2026 final", game="game 3",
            date="2026-07-12")

drafts = pd.read_parquet(HERE.parent / "data/processed/draft_sequences_multi.parquet")

# the comp must be a real line-up in the record, exactly as stated
_fin = drafts[(drafts.league == "MSI")
              & (drafts.date.dt.date == pd.Timestamp(GAME["date"]).date())
              & (drafts.is_ban == 0)]
_hit = None
for (gid, blue), g in _fin.groupby(["gameid", "is_blue"]):
    if set(g.champion) == set(COMP):
        _hit = (gid, g.team.iloc[0], int(g.game_in_series.iloc[0]))
assert _hit is not None, "the comp must exist in the draft record as a real line-up"
assert _hit[1] == GAME["team"], f"team mismatch: {_hit[1]}"
assert _hit[2] == 3, f"expected game 3 of the series, got {_hit[2]}"
print(f"comp verified: {_hit[1]}, game {_hit[2]}, {_hit[0]}")

assert sorted(role[idx[c]] for c in COMP) == [0, 1, 2, 3, 4], \
    "the comp must field one champion per role"

# ------------------------------------------------- co-occurrence, verified ---
_picks = drafts[drafts.is_ban == 0]
_side = _picks.groupby(["gameid", "is_blue"]).champion.apply(set)
NCOMPS = len(_side)


def together(a, b):
    return int(sum(1 for s in _side if a in s and b in s))


def lineups(a):
    return int(sum(1 for s in _side if a in s))


def expected(a, b):
    return lineups(a) * lineups(b) / NCOMPS


PAIRS = []
for a, b, kind in [("Rakan", "Xayah", "partners"),
                   ("Rakan", "Alistar", "rivals"),
                   ("Rakan", "Orianna", "unrelated")]:
    PAIRS.append(dict(a=a, b=b, kind=kind, tog=together(a, b),
                      exp=round(expected(a, b), 1),
                      dist=round(float(np.linalg.norm(pos[-1, idx[a]] - pos[-1, idx[b]])), 4)))
P = {p["b"]: p for p in PAIRS}
assert NCOMPS == 10294, f"line-up count drifted: {NCOMPS}"
assert P["Alistar"]["tog"] == 0, "Rakan and Alistar must never share a line-up"
assert P["Xayah"]["tog"] > 4 * P["Xayah"]["exp"], "Xayah/Rakan must beat chance heavily"
assert P["Xayah"]["dist"] > 5 * P["Alistar"]["dist"], (
    "the whole exhibit: the rival must sit far closer than the partner")
RATIO = P["Xayah"]["dist"] / P["Alistar"]["dist"]
XR = P["Xayah"]["tog"] / P["Xayah"]["exp"]
print(f"pairs: {PAIRS}")
print(f"partner is {RATIO:.1f}x further than rival; Xayah/Rakan at {XR:.1f}x chance")

# ------------------------------------------------ per-champion, RAW space ----
# champ_purity is the 192-dim score the page already prints. The 2-D layout
# gives DIFFERENT numbers; those never appear on a page. Keep this separation.
cp = bp6.champ_purity
CARDS = [dict(name=c, role=int(role[idx[c]]),
              purity=round(float(cp[idx[c]]), 2),
              exposure=bp6.champ_card(c)["exposure"]) for c in COMP]
COMP_PUR = round(float(np.mean([cp[idx[c]] for c in COMP])), 3)
ALL_PUR = round(float(cp.mean()), 3)
assert abs(ALL_PUR - purity[-1]) < 1e-3, "per-champion mean must match the headline"
assert COMP_PUR > ALL_PUR, "the exhibit claims a pro comp is tidier than average"
print(f"comp mean purity {COMP_PUR} vs all-168 {ALL_PUR}; cards {CARDS}")

# --------------------------------------------------------- keyframe choice ---
def first_at(t):
    for i, p in enumerate(purity):
        if p >= t:
            return i
    return nF - 1


KEYS = [0, first_at(0.30), first_at(0.50), nF - 1]
assert len(set(KEYS)) == 4
print("keyframes:", [(k, labels[k], purity[k]) for k in KEYS])


def per_role_purity(f):
    Q = pos[f]
    d = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, 1e9)
    nn = np.argsort(d, 1)[:, :5]
    hit = (role[nn] == role[:, None]).mean(1)
    return {ROLE_NAMES[r]: float(hit[role == r].mean()) for r in range(5)}


_p1, _p2 = per_role_purity(KEYS[1]), per_role_purity(KEYS[2])
assert min(_p1["Bot"], _p1["Support"]) > max(_p1["Top"], _p1["Mid"]), _p1
assert min(_p2, key=_p2.get) == "Top", _p2

NOTES = [
    "Kick-off. Every champion is still 192 random numbers, so Hanwha&rsquo;s five sit "
    "wherever chance dropped them. The model has not read a single draft yet.",
    "After a few passes the bot lane splits off first &mdash; marksmen and supports "
    "are the first pair the model learns to tell apart. Top and mid are still one blob.",
    "Most jobs now have a home region. Top is the last to separate, which is why "
    "Olaf is still drifting.",
    "Trained. Xayah and Rakan &mdash; the duo you see picked together every split &mdash; "
    "settle a long way apart. Alistar, who Rakan would never be drafted alongside, "
    "ends up almost on top of him.",
]

FRAG = r"""
<div class="panelwrap">
  <div class="panel" id="p2-panel">
    <h3>MOCK &middot; 2&times;2 strip with a real comp highlighted</h3>
    <p class="sub">You know this draft: <b>@@TEAM@@</b>, @@GAME@@ of the @@EVENT@@ &mdash;
    @@COMPLIST@@. One champion for each of the five jobs. The model was never told
    that there are five jobs, or which champion plays which. It only read
    @@NCOMPS@@ pro line-ups and tried to guess the next pick.</p>

    <div class="p2-strip" id="p2-strip"></div>
    <div class="p2-legend" id="p2-legend"></div>

    <p class="codecap"><b>Grey dots are the other @@NOTHER@@ champions. Coloured dots are Hanwha&rsquo;s five.</b>
    When each dot settles is measured; where it travels between snapshots is drawn. Read the timing, not the path.</p>

    <div class="p2-anim">
      <div class="p2-animhd">The same thing moving, if you would rather watch it</div>
      <canvas id="p2-map" style="width:100%;display:block" role="img" aria-label="Animated map of 168 champions, with Hanwha Life's five picks highlighted, moving from a random cloud into role regions"></canvas>
      <div class="p2-controls">
        <button id="p2-play" type="button">&#9654; Play</button>
        <input id="p2-scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Training snapshot">
      </div>
      <div class="p2-readout" id="p2-label"></div>
    </div>

    <div class="p2-payoff">
      <p class="p2-po-h">What you already knew from watching, and the model worked out on its own</p>
      <canvas id="p2-pay" style="width:100%;display:block" role="img" aria-label="The finished layout with only Rakan, Alistar, Xayah and Orianna highlighted. Rakan and Alistar sit almost on the same point; Xayah sits far away."></canvas>
      <p class="codecap p2-paycap">The same finished layout, with only these four lit. Rakan and Alistar are the two dots you cannot tell apart.</p>
      <table class="p2-tbl">
        <thead><tr><th>Picked in the same team</th><th class="n">Together</th><th class="n">If it were chance</th><th class="n">Distance on the map</th></tr></thead>
        <tbody>
          <tr><td><b>Rakan + Xayah</b> &mdash; the lane duo</td><td class="n" data-l="Together">@@XT@@</td><td class="n" data-l="If it were chance">@@XE@@</td><td class="n" data-l="Distance on the map">@@XD@@</td></tr>
          <tr><td><b>Rakan + Alistar</b> &mdash; both want the support seat</td><td class="n" data-l="Together">@@AT@@</td><td class="n" data-l="If it were chance">@@AE@@</td><td class="n" data-l="Distance on the map">@@AD@@</td></tr>
          <tr><td>Rakan + Orianna &mdash; no particular link</td><td class="n" data-l="Together">@@OT@@</td><td class="n" data-l="If it were chance">@@OE@@</td><td class="n" data-l="Distance on the map">@@OD@@</td></tr>
        </tbody>
      </table>
      <p class="p2-po-t"><b>That is the flip worth sitting with.</b> Rakan and Alistar are never drafted
      together &mdash; @@AT@@ times out of an expected @@AE@@ &mdash; and the model puts them practically
      on top of each other. Xayah, who actually plays beside Rakan @@XT@@ times, ends up
      <b>@@RATIO@@&times; further away</b>. Champions that compete for the same seat get the same numbers;
      champions that play beside each other do not have to.</p>
    </div>
  </div>
</div>

<style>
  #p2-panel > h3 { font-size: 21px; line-height: 1.25; margin-bottom: 6px; }
  .p2-strip { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; margin: 24px 0 0; }
  @media (max-width: 430px) { .p2-strip { gap: 14px; } }
  .p2-cell { display: flex; flex-direction: column; gap: 8px; }
  .p2-cell canvas {
    width: 100%; display: block;
    border: 1px solid var(--hairline); border-radius: 8px; background: var(--inset);
  }
  .p2-hd {
    font: 600 11.5px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); display: flex; justify-content: space-between; gap: 8px;
    font-variant-numeric: tabular-nums;
  }
  .p2-hd b { color: var(--ink); }
  .p2-note {
    font: 500 12.5px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); margin: 0;
  }
  .p2-legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 18px 0 0; }
  .p2-legend span {
    font: 600 11.5px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); display: inline-flex; align-items: center; gap: 5px;
  }
  .p2-legend i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .p2-anim { margin-top: 32px; }
  .p2-animhd {
    font: 600 12px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); margin-bottom: 10px;
  }
  .p2-controls { display: flex; align-items: center; gap: 12px; margin: 10px 0 0; }
  .p2-controls button {
    font: 600 13px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink); background: var(--inset);
    border: 1px solid var(--hairline); border-radius: 7px;
    padding: 7px 12px; cursor: pointer; white-space: nowrap;
  }
  .p2-controls input[type=range] { flex: 1; accent-color: var(--muted); min-width: 0; }
  .p2-readout {
    font: 500 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); margin-top: 8px;
  }
  .p2-payoff { margin-top: 32px; }
  .p2-po-h {
    font: 700 13.5px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink); margin: 0 0 10px;
  }
  .p2-tbl { width: 100%; border-collapse: collapse; }
  .p2-tbl th, .p2-tbl td {
    text-align: left; padding: 8px 10px 8px 0;
    border-bottom: 1px solid var(--hairline);
    font: 500 12.5px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted);
  }
  .p2-tbl th { font-weight: 700; color: var(--ink); }
  .p2-tbl td b { color: var(--ink); }
  .p2-tbl .n {
    text-align: right; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .p2-paycap { margin-top: 10px; }
  /* Below 560px four columns cannot fit: the last one was running off the
     panel entirely, so the map distance — the whole point — was invisible on a
     phone. Each row becomes its own block with the header carried inline. */
  @media (max-width: 560px) {
    .p2-tbl, .p2-tbl tbody, .p2-tbl tr, .p2-tbl td { display: block; width: 100%; }
    .p2-tbl thead { position: absolute; width: 1px; height: 1px;
      overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
    .p2-tbl tr { padding: 10px 0; border-bottom: 1px solid var(--hairline); }
    .p2-tbl td { border: 0; padding: 2px 0; }
    .p2-tbl td:first-child { padding-bottom: 6px; }
    .p2-tbl td.n {
      display: flex; justify-content: space-between; gap: 12px; text-align: right;
    }
    .p2-tbl td.n::before {
      content: attr(data-l); font-family: system-ui, -apple-system, sans-serif;
      font-weight: 600; color: var(--muted); text-align: left;
    }
  }
  .p2-po-t {
    font: 500 13px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--muted); margin: 14px 0 0;
  }
  .p2-po-t b { color: var(--ink); }
</style>

<script>
(function () {
  "use strict";
  var D = @@DATA@@, KEYS = @@KEYS@@, NOTES = @@NOTES@@, COMP = @@COMP@@;
  var ROLE_NAMES = ["Top", "Jungle", "Mid", "Bot", "Support"];
  var ROLE_LIGHT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#4a3aa7"];
  var ROLE_DARK  = ["#3987e5", "#008300", "#d55181", "#c98500", "#9085e9"];
  var nF = D.nF, N = D.names.length;

  var raw = atob(D.pos), pos = new Uint16Array(raw.length / 2);
  for (var bi = 0; bi < pos.length; bi++) {
    pos[bi] = raw.charCodeAt(2 * bi) | (raw.charCodeAt(2 * bi + 1) << 8);
  }
  function px(f, c) { return pos[(f * N + c) * 2] / 1000; }
  function py(f, c) { return pos[(f * N + c) * 2 + 1] / 1000; }

  var CI = COMP.map(function (n) { return D.names.indexOf(n); });
  var isComp = {}; CI.forEach(function (i) { isComp[i] = true; });

  var darkMq = window.matchMedia("(prefers-color-scheme: dark)");
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return darkMq.matches;
  }
  function rc() { return isDark() ? ROLE_DARK : ROLE_LIGHT; }
  function cssv(n) {
    return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  }
  function sizeCanvas(cv, hCss) {
    var w = cv.clientWidth || 200, dpr = window.devicePixelRatio || 1;
    cv.style.height = hCss + "px";
    cv.width = Math.round(w * dpr); cv.height = Math.round(hCss * dpr);
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // One label placer shared by every drawing path, so no two labels can ever
  // overlap. Tries candidate offsets, and DROPS a label rather than collide.
  // It also avoids the DOTS, not just other labels: a name sitting on top of a
  // grey dot reads as a typo (we shipped "Xayaḥ" that way in the first mock).
  function makePlacer(ctx, dots) {
    var boxes = (dots || []).map(function (d) {
      return [d[0] - d[2], d[1] - d[2], d[0] + d[2], d[1] + d[2]];
    });
    return function (text, x, y, fill) {
      var w = ctx.measureText(text).width, h = 11, pad = 3;
      var cands = [[8, 0], [-8 - w, 0], [0, -13], [0, 13], [8, -12], [-8 - w, -12],
                   [8, 12], [-8 - w, 12], [0, -22], [0, 22]];
      for (var i = 0; i < cands.length; i++) {
        var bx = x + cands[i][0], by = y + cands[i][1] - h / 2;
        var r = [bx - pad, by - pad, bx + w + pad, by + h + pad];
        var clash = false;
        for (var j = 0; j < boxes.length; j++) {
          var o = boxes[j];
          if (r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1]) { clash = true; break; }
        }
        if (clash) continue;
        boxes.push(r);
        ctx.fillStyle = fill;
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(text, bx, y + cands[i][1]);
        return true;
      }
      return false;
    };
  }

  function drawKey(cv, f) {
    var w = cv.clientWidth || 200, h = Math.round(w * 0.86);
    var ctx = sizeCanvas(cv, h);
    ctx.clearRect(0, 0, w, h);
    var C = rc(), pad = 12, iw = w - 2 * pad, ih = h - 2 * pad;
    var faint = cssv("--faint") || "#d8d8d8";
    // context first, so the comp always sits on top
    for (var i = 0; i < N; i++) {
      if (isComp[i]) continue;
      ctx.beginPath();
      ctx.arc(pad + px(f, i) * iw, pad + py(f, i) * ih, 1.9, 0, 7);
      ctx.fillStyle = faint; ctx.fill();
    }
    var ring = cssv("--card") || "#fff", dots = [];
    for (var k = 0; k < CI.length; k++) {
      var ci = CI[k];
      var X = pad + px(f, ci) * iw, Y = pad + py(f, ci) * ih;
      ctx.beginPath(); ctx.arc(X, Y, 5.4, 0, 7);
      ctx.fillStyle = ring; ctx.fill();
      ctx.beginPath(); ctx.arc(X, Y, 4.0, 0, 7);
      ctx.fillStyle = C[D.role[ci]]; ctx.fill();
      dots.push([X, Y, 6.5]);
    }
    // Below ~240px a cell cannot carry five names without them fusing. The
    // legend already maps colour to champion, so drop the labels rather than
    // print mush — the 390px cells are 150px wide.
    if (w >= 240) {
      ctx.font = '600 10.5px system-ui, -apple-system, "Segoe UI", sans-serif';
      var place = makePlacer(ctx, dots);
      for (var k2 = 0; k2 < CI.length; k2++) {
        var ci2 = CI[k2];
        place(D.names[ci2], pad + px(f, ci2) * iw, pad + py(f, ci2) * ih,
              cssv("--ink") || "#111");
      }
    }
  }

  /* ---- the payoff figure: the distances the table quotes, made visible ----
     The table claims Rakan sits practically on Alistar and far from Xayah.
     A number the reader cannot see on the map is a number they cannot check,
     so the final layout is redrawn here with only those four champions lit. */
  function drawPayoff(cv) {
    var w = cv.clientWidth || 400;
    var h = Math.max(200, Math.min(Math.round(w / D.aspect), 420));
    var ctx = sizeCanvas(cv, h);
    ctx.clearRect(0, 0, w, h);
    var C = rc(), pad = 16, iw = w - 2 * pad, ih = h - 2 * pad, f = nF - 1;
    var faint = cssv("--faint") || "#d8d8d8", ink = cssv("--ink") || "#111";
    var mut = cssv("--muted") || "#666", ring = cssv("--card") || "#fff";
    function X(i) { return pad + px(f, i) * iw; }
    function Y(i) { return pad + py(f, i) * ih; }
    var FOUR = ["Rakan", "Alistar", "Xayah", "Orianna"].map(function (n) {
      return D.names.indexOf(n);
    });
    var lit = {}; FOUR.forEach(function (i) { lit[i] = true; });
    for (var i = 0; i < N; i++) {
      if (lit[i]) continue;
      ctx.beginPath(); ctx.arc(X(i), Y(i), 2.2, 0, 7);
      ctx.fillStyle = faint; ctx.fill();
    }
    // connector from Rakan to each of the two contrast champions
    var R = FOUR[0];
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1.4; ctx.strokeStyle = mut;
    [FOUR[2]].forEach(function (t) {
      ctx.beginPath(); ctx.moveTo(X(R), Y(R)); ctx.lineTo(X(t), Y(t)); ctx.stroke();
    });
    ctx.setLineDash([]);
    var dots = [];
    FOUR.forEach(function (i) {
      ctx.beginPath(); ctx.arc(X(i), Y(i), 7, 0, 7);
      ctx.fillStyle = ring; ctx.fill();
      ctx.beginPath(); ctx.arc(X(i), Y(i), 5.2, 0, 7);
      ctx.fillStyle = C[D.role[i]]; ctx.fill();
      dots.push([X(i), Y(i), 8]);
    });
    ctx.font = '600 11.5px system-ui, -apple-system, "Segoe UI", sans-serif';
    var place = makePlacer(ctx, dots);
    FOUR.forEach(function (i) { place(D.names[i], X(i), Y(i), ink); });
    // distance annotation on the Rakan-Xayah span, at its midpoint
    ctx.font = '600 11px ui-monospace, "SF Mono", Menlo, monospace';
    var mx = (X(R) + X(FOUR[2])) / 2, my = (Y(R) + Y(FOUR[2])) / 2;
    place("0.238 apart", mx, my, mut);
    place("0.026 apart", X(FOUR[1]), Y(FOUR[1]) + 16, mut);
  }

  var strip = document.getElementById("p2-strip");
  var cells = [];
  KEYS.forEach(function (f, i) {
    var cell = document.createElement("div"); cell.className = "p2-cell";
    var hd = document.createElement("div"); hd.className = "p2-hd";
    hd.innerHTML = "<b>" + (i + 1) + " &middot; " + D.labels[f] + "</b>" +
                   "<span>purity " + D.purity[f].toFixed(2) + "</span>";
    var cv = document.createElement("canvas");
    cv.setAttribute("role", "img");
    cv.setAttribute("aria-label", "Snapshot " + (i + 1) + ", " + D.labels[f] +
      ", role purity " + D.purity[f].toFixed(2) + ".");
    var note = document.createElement("p"); note.className = "p2-note";
    note.innerHTML = NOTES[i];
    cell.appendChild(hd); cell.appendChild(cv); cell.appendChild(note);
    strip.appendChild(cell); cells.push({ cv: cv, f: f });
  });

  var legend = document.getElementById("p2-legend");
  var map = document.getElementById("p2-map"),
      scrub = document.getElementById("p2-scrub"),
      playBtn = document.getElementById("p2-play"),
      lab = document.getElementById("p2-label");
  var ft = 0, playing = false, rafId = null;
  scrub.max = String(nF - 1);

  function drawBig() {
    var w = map.clientWidth || 600;
    var h = Math.max(240, Math.min(Math.round(w / D.aspect), 520));
    var ctx = sizeCanvas(map, h);
    ctx.clearRect(0, 0, w, h);
    var C = rc(), pad = 14, iw = w - 2 * pad, ih = h - 2 * pad;
    var f0 = Math.floor(ft), f1 = Math.min(f0 + 1, nF - 1), t = ft - f0;
    function X(i) { return (px(f0, i) * (1 - t) + px(f1, i) * t) * iw + pad; }
    function Y(i) { return (py(f0, i) * (1 - t) + py(f1, i) * t) * ih + pad; }
    var faint = cssv("--faint") || "#d8d8d8";
    for (var i = 0; i < N; i++) {
      if (isComp[i]) continue;
      ctx.beginPath(); ctx.arc(X(i), Y(i), 2.4, 0, 7);
      ctx.fillStyle = faint; ctx.fill();
    }
    var ring = cssv("--card") || "#fff", dots = [];
    for (var k = 0; k < CI.length; k++) {
      var ci = CI[k];
      ctx.beginPath(); ctx.arc(X(ci), Y(ci), 6.4, 0, 7);
      ctx.fillStyle = ring; ctx.fill();
      ctx.beginPath(); ctx.arc(X(ci), Y(ci), 4.8, 0, 7);
      ctx.fillStyle = C[D.role[ci]]; ctx.fill();
      dots.push([X(ci), Y(ci), 7.5]);
    }
    ctx.font = '600 11.5px system-ui, -apple-system, "Segoe UI", sans-serif';
    var place = makePlacer(ctx, dots);
    for (var k2 = 0; k2 < CI.length; k2++) {
      place(D.names[CI[k2]], X(CI[k2]), Y(CI[k2]), cssv("--ink") || "#111");
    }
    var fi = Math.round(ft);
    lab.textContent = D.labels[fi] + " · purity " + D.purity[fi].toFixed(2);
    scrub.value = String(fi);
  }
  var lastTs = null;
  function tick(ts) {
    if (!playing) return;
    if (lastTs == null) lastTs = ts;
    ft += (ts - lastTs) / 150; lastTs = ts;
    if (ft >= nF - 1) { ft = nF - 1; stop(); drawBig(); return; }
    drawBig(); rafId = requestAnimationFrame(tick);
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
    var C = rc();
    legend.innerHTML = "";
    CI.forEach(function (ci) {
      var s = document.createElement("span");
      s.innerHTML = "<i style='background:" + C[D.role[ci]] + "'></i>" +
                    D.names[ci] + " &middot; " + ROLE_NAMES[D.role[ci]];
      legend.appendChild(s);
    });
    cells.forEach(function (c) { drawKey(c.cv, c.f); });
    drawBig();
    drawPayoff(document.getElementById("p2-pay"));
  }
  darkMq.addEventListener("change", renderAll);
  new MutationObserver(renderAll).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });
  window.addEventListener("resize", renderAll);
  renderAll();
})();
</script>
"""

data_json = json.dumps(payload, separators=(",", ":"))
assert "</" not in data_json
frag = (FRAG.replace("@@DATA@@", data_json)
        .replace("@@KEYS@@", json.dumps(KEYS))
        .replace("@@NOTES@@", json.dumps(NOTES))
        .replace("@@COMP@@", json.dumps(COMP))
        .replace("@@TEAM@@", GAME["team"])
        .replace("@@EVENT@@", GAME["event"])
        .replace("@@GAME@@", GAME["game"])
        .replace("@@COMPLIST@@", ", ".join(COMP))
        .replace("@@NCOMPS@@", f"{NCOMPS:,}")
        .replace("@@NOTHER@@", str(len(names) - len(COMP)))
        .replace("@@XT@@", str(P["Xayah"]["tog"]))
        .replace("@@XE@@", f'{P["Xayah"]["exp"]:.1f}')
        .replace("@@XD@@", f'{P["Xayah"]["dist"]:.3f}')
        .replace("@@AT@@", str(P["Alistar"]["tog"]))
        .replace("@@AE@@", f'{P["Alistar"]["exp"]:.1f}')
        .replace("@@AD@@", f'{P["Alistar"]["dist"]:.3f}')
        .replace("@@OT@@", str(P["Orianna"]["tog"]))
        .replace("@@OE@@", f'{P["Orianna"]["exp"]:.1f}')
        .replace("@@OD@@", f'{P["Orianna"]["dist"]:.3f}')
        .replace("@@RATIO@@", f"{RATIO:.0f}"))

shipped = (HERE / "artifact-v8.html").read_text()
head_css = shipped.split("<style>", 1)[1].split("</style>", 1)[0]


def page(theme):
    return ("<!doctype html><html" + theme + "><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Mock &middot; 2x2 strip with a real comp</title>"
            "<style>" + head_css + "</style></head><body>" + frag + "</body></html>")


(HERE / "proto_comic_v2.html").write_text(page(""))
(HERE / "proto_comic_v2_dark.html").write_text(page(' data-theme="dark"'))
print("wrote proto_comic_v2.html and proto_comic_v2_dark.html")
