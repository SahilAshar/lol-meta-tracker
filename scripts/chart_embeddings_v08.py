"""t-SNE map of the v0.8 learned champion embeddings, colored by primary role.

Roles are computed from the same multi-year data the model trained on (majority
position per champion; a champion is "flex" when its top role holds < 70% of
its games — those get a dark ring and both-role annotation). Also prints a
quantitative cluster check: 5-NN role purity in the raw embedding space vs the
20% chance floor — if the embedding didn't learn roles, the chart must not
pretend it did.

Labels are placed with a collision-aware greedy layout: candidates are scored
against precisely measured text boxes (not a length heuristic), flex picks are
always labeled, and the densest non-notable points quietly drop their label
(keeping the dot) rather than piling illegible text into cluster cores.

Input:  data/processed/champion_embeddings_v08.npz (from train_draft_model_v08.py)
Output: charts/champion_embeddings_tsne_v08.png (light, default)
        charts/champion_embeddings_tsne_v08_dark.png (with --dark)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from common import CHARTS_DIR, DATA_PROCESSED, raw_csv_path
from draft_dataset import LEAGUES, ROLES

YEARS = [2024, 2025, 2026]
FLEX_THRESHOLD = 0.70

# Overlap budget for a non-must-keep label: if the least-bad candidate offset
# still overlaps already-placed labels by more than this fraction of its own
# box, the label is dropped (dot stays). Must-keep labels (flex picks) always
# place at the least-bad candidate regardless.
DROP_THRESHOLD = 0.08

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        flex_swatch="#b9b8b2",
        role_colors={
            "top": "#2a78d6", "jng": "#008300", "mid": "#e87ba4",
            "bot": "#eda100", "sup": "#4a3aa7",
        },
    ),
    "dark": dict(
        # Matches the artifact's --card / --ink / --muted so the PNG blends
        # into the panel with no visible seam.
        surface="#1D2634",
        ink="#E8EAF0",
        ink2="#98A0B3",
        flex_swatch="#5b6472",
        # Dark-surface steps from the validated categorical palette
        # (slots 1/2/3/4/7), swapped in for contrast on the dark card.
        role_colors={
            "top": "#3987e5", "jng": "#008300", "mid": "#d55181",
            "bot": "#c98500", "sup": "#9085e9",
        },
    ),
}
ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}


def role_shares() -> pd.DataFrame:
    frames = []
    for y in YEARS:
        df = pd.read_csv(raw_csv_path(y), low_memory=False,
                         usecols=["league", "position", "champion", "gameid"])
        frames.append(df[df.league.isin(LEAGUES) & df.position.isin(ROLES)])
    games = pd.concat(frames).groupby(["champion", "position"]).gameid.nunique()
    shares = games.unstack(fill_value=0).reindex(columns=ROLES, fill_value=0)
    return shares.div(shares.sum(axis=1), axis=0)


def place_labels(fig, ax, xs, ys, texts, colors, must_keep, clearance_px, fontsize=5.6):
    """Collision-aware greedy label placement.

    Priority order: must-keep points (flex picks) first, then the rest from
    sparsest to densest local neighborhood — so natural per-cluster anchors
    get first claim on clean placements and only the genuinely crowded points
    compete for scraps. A label is dropped (not the dot) when even its best
    candidate offset still collides too much with what's already placed,
    unless the point is must-keep.
    """
    n = len(texts)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # Local crowding: neighbor count within a small radius, used only to
    # order placement (sparse points first).
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    radius = 0.03 * diag
    nn = NearestNeighbors(radius=radius).fit(np.column_stack([xs, ys]))
    crowd = np.array([len(a) for a in nn.radius_neighbors(np.column_stack([xs, ys]),
                                                            return_distance=False)])

    must_keep = np.asarray(must_keep, dtype=bool)
    order = sorted(range(n), key=lambda i: (not must_keep[i], crowd[i]))

    # Accurate per-text box size in data units via a hidden text artist.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_size_data(s):
        t = ax.text(0, 0, s, fontsize=fontsize, ha="left", va="bottom", alpha=0)
        bbox = t.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
        t.remove()
        return abs(x1 - x0), abs(y1 - y0)

    sizes = {s: text_size_data(s) for s in set(texts)}

    # Pixel gap -> data units (anisotropic-safe).
    p0 = inv.transform((0, 0))
    px = inv.transform((1, 0))
    py = inv.transform((0, 1))
    dx_per_px = px[0] - p0[0]
    dy_per_px = py[1] - p0[1]

    # (hx, vy): horizontal side (1=right, -1=left, 0=centered) and vertical
    # side (1=above, -1=below, 0=centered) relative to the marker.
    compass = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def offsets(base_px):
        # Gap tiers start beyond this point's own marker radius (+ring, for
        # flex picks) plus a small buffer, so labels never render on top of
        # their own dot.
        for gap_px in (base_px, base_px + 7, base_px + 16):
            gx, gy = gap_px * dx_per_px, gap_px * dy_per_px
            for hx, vy in compass:
                yield gap_px, hx, vy, gx, gy

    # Seed collision detection with every marker's own footprint (not just
    # already-placed labels), so a label never lands on top of a *nearby*
    # dot either — the case that matters for near-coincident points.
    placed_boxes: list[tuple[float, float, float, float]] = []
    for j in range(n):
        rx, ry = clearance_px[j] * dx_per_px, clearance_px[j] * dy_per_px
        placed_boxes.append((xs[j] - rx, ys[j] - ry, xs[j] + rx, ys[j] + ry))

    kept = [False] * n
    anchors: list[tuple[float, float]] = [None] * n  # (x0, y0) i.e. ha=left, va=bottom

    for i in order:
        w, h = sizes[texts[i]]
        best_box, best_ov, best_r = None, None, None
        for gap_px, hx, vy, gx, gy in offsets(clearance_px[i]):
            x0 = xs[i] + gx if hx == 1 else (xs[i] - gx - w if hx == -1 else xs[i] - w / 2)
            y0 = ys[i] + gy if vy == 1 else (ys[i] - gy - h if vy == -1 else ys[i] - h / 2)
            box = (x0, y0, x0 + w, y0 + h)
            ov = sum(
                max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
                * max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
                for b in placed_boxes
            )
            if best_ov is None or ov < best_ov or (ov == best_ov and gap_px < best_r):
                best_box, best_ov, best_r = box, ov, gap_px
            if ov == 0.0:
                break

        frac = best_ov / max(w * h, 1e-12)
        if must_keep[i] or frac <= DROP_THRESHOLD:
            placed_boxes.append(best_box)
            kept[i] = True
            anchors[i] = (best_box[0], best_box[1])

    for i in order:
        if kept[i]:
            ax.text(anchors[i][0], anchors[i][1], texts[i], fontsize=fontsize,
                     color=colors[i], ha="left", va="bottom", zorder=4)

    return sum(kept), n - sum(kept)


def render(theme_name: str, xy, known, primary, flex, sh, purity) -> None:
    theme = THEMES[theme_name]
    surface, ink, ink2 = theme["surface"], theme["ink"], theme["ink2"]
    role_colors = theme["role_colors"]

    fig, ax = plt.subplots(figsize=(13.5, 10.5), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # Fix the data extent before measuring/placing labels so the data<->pixel
    # transform used for label sizing doesn't shift later.
    pad_x = 0.06 * (xy[:, 0].max() - xy[:, 0].min())
    pad_y = 0.06 * (xy[:, 1].max() - xy[:, 1].min())
    ax.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
    ax.set_ylim(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)

    colors = primary.map(role_colors).to_numpy()
    flex_np = flex.to_numpy()
    # Thin surface-color ring so close/overlapping same-role dots stay
    # legible against each other, not just against the background.
    ax.scatter(xy[~flex_np, 0], xy[~flex_np, 1],
               c=colors[~flex_np], s=34, linewidths=0.8,
               edgecolors=surface, zorder=3)
    ax.scatter(xy[flex_np, 0], xy[flex_np, 1],
               c=colors[flex_np], s=42, linewidths=1.3,
               edgecolors=ink, zorder=3)

    labels = [
        f"{c} ({'/'.join(r for r in ROLES if sh.loc[c, r] >= 0.2)})" if f else c
        for c, f in zip(known, flex)
    ]
    # Marker radius in px (dpi-aware) plus its ring width, so labels clear
    # the actual drawn dot instead of a fixed guess. s is area in pt^2.
    px_per_pt = fig.dpi / 72.0
    flex_radius_px = (np.sqrt(42 / np.pi) + 1.3) * px_per_pt
    nonflex_radius_px = (np.sqrt(34 / np.pi) + 0.8) * px_per_pt
    clearance_px = np.where(flex_np, flex_radius_px, nonflex_radius_px) + 4.0

    kept, dropped = place_labels(
        fig, ax, xy[:, 0], xy[:, 1], labels,
        np.where(flex_np, ink, ink2), must_keep=flex_np,
        clearance_px=clearance_px,
    )
    print(f"[{theme_name}] labels placed: {kept}, dropped for collision: {dropped}")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=role_colors[r],
                   markersize=7, label=ROLE_LABELS[r]) for r in ROLES
    ] + [plt.Line2D([], [], marker="o", linestyle="", color=theme["flex_swatch"],
                    markeredgecolor=ink, markeredgewidth=1.3, markersize=8,
                    label="Flex (top role < 70%)")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5,
              labelcolor=ink)
    ax.text(0, 1.075, "What a draft model learns about champions",
            transform=ax.transAxes, fontsize=15, color=ink, fontweight="bold",
            va="bottom")
    ax.text(0, 1.012,
            "t-SNE of champion embeddings learned by the lol-meta-tracker v0.8 draft "
            "transformer from 2024–2026 pro drafts.\nColor = majority role. "
            "Ringed champions are flex picks — the model places them between "
            "role clusters on its own. Axes are unitless.",
            transform=ax.transAxes, fontsize=8.5, color=ink2, va="bottom")
    ax.text(1.0, -0.02, "@lolmetatracker · data: Oracle's Elixir",
            transform=ax.transAxes, fontsize=7.5, color=ink2, ha="right")
    fig.tight_layout()
    CHARTS_DIR.mkdir(exist_ok=True)
    suffix = "" if theme_name == "light" else "_dark"
    out = CHARTS_DIR / f"champion_embeddings_tsne_v08{suffix}.png"
    fig.savefig(out, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark", action="store_true",
                         help="Render the dark-theme variant instead of light.")
    args = parser.parse_args()

    data = np.load(DATA_PROCESSED / "champion_embeddings_v08.npz", allow_pickle=True)
    emb, champs = data["embeddings"], list(data["champions"])
    shares = role_shares()
    known = [c for c in champs if c in shares.index]
    emb = emb[[champs.index(c) for c in known]]
    sh = shares.loc[known]
    primary = sh.idxmax(axis=1)
    top_share = sh.max(axis=1)
    flex = top_share < FLEX_THRESHOLD

    # Quantitative check before drawing anything pretty.
    nn = NearestNeighbors(n_neighbors=6).fit(emb)
    _, idx = nn.kneighbors(emb)
    purity = np.mean([
        np.mean(primary.iloc[idx[i, 1:]] == primary.iloc[i]) for i in range(len(known))
    ])
    print(f"5-NN primary-role purity in embedding space: {purity:.3f} (chance ~0.20)")

    xy = TSNE(n_components=2, perplexity=25, init="pca", learning_rate="auto",
              random_state=16).fit_transform(emb)

    render("dark" if args.dark else "light", xy, known, primary, flex, sh, purity)


if __name__ == "__main__":
    main()
