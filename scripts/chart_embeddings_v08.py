"""t-SNE map of the v0.8 learned champion embeddings, colored by primary role.

Roles are computed from the same multi-year data the model trained on (majority
position per champion; a champion is "flex" when its top role holds < 70% of
its games — those get a dark ring and both-role annotation). Also prints a
quantitative cluster check: 5-NN role purity in the raw embedding space vs the
20% chance floor — if the embedding didn't learn roles, the chart must not
pretend it did.

Input:  data/processed/champion_embeddings_v08.npz (from train_draft_model_v08.py)
Output: charts/champion_embeddings_tsne_v08.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from common import CHARTS_DIR, DATA_PROCESSED, raw_csv_path
from draft_dataset import LEAGUES, ROLES

YEARS = [2024, 2025, 2026]
FLEX_THRESHOLD = 0.70

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
# Validated 5-slot categorical palette (all-pairs, light surface).
ROLE_COLORS = {
    "top": "#2a78d6", "jng": "#008300", "mid": "#e87ba4",
    "bot": "#eda100", "sup": "#4a3aa7",
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


def place_labels(ax, xs, ys, texts, colors):
    """Greedy 4-offset label placement to limit collisions."""
    placed: list[tuple[float, float, float, float]] = []
    xr = xs.max() - xs.min()
    yr = ys.max() - ys.min()
    offs = [(0.004, 0.004), (0.004, -0.010), (-0.004, 0.004), (-0.004, -0.010)]
    order = np.argsort(xs)  # deterministic
    for i in order:
        w, h = len(texts[i]) * 0.0042 * xr, 0.011 * yr
        best, best_overlap = None, None
        for dx, dy in offs:
            x0, y0 = xs[i] + dx * xr, ys[i] + dy * yr
            box = (x0, y0, x0 + w, y0 + h)
            ov = sum(
                max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
                * max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
                for b in placed
            )
            if best_overlap is None or ov < best_overlap:
                best, best_overlap = box, ov
            if ov == 0.0:
                break
        placed.append(best)
        ax.text(best[0], best[1], texts[i], fontsize=5.4, color=colors[i],
                ha="left", va="bottom", zorder=4)


def main() -> None:
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
    same = np.mean([
        np.mean(primary.iloc[idx[i, 1:]] == primary.iloc[i]) for i in range(len(known))
    ])
    print(f"5-NN primary-role purity in embedding space: {same:.3f} (chance ~0.20)")

    xy = TSNE(n_components=2, perplexity=25, init="pca", learning_rate="auto",
              random_state=16).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(13.5, 10.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = primary.map(ROLE_COLORS).to_numpy()
    ax.scatter(xy[~flex.to_numpy(), 0], xy[~flex.to_numpy(), 1],
               c=colors[~flex.to_numpy()], s=34, linewidths=0, zorder=3)
    ax.scatter(xy[flex.to_numpy(), 0], xy[flex.to_numpy(), 1],
               c=colors[flex.to_numpy()], s=42, linewidths=1.3,
               edgecolors=INK, zorder=3)
    labels = [
        f"{c} ({'/'.join(r for r in ROLES if sh.loc[c, r] >= 0.2)})" if f else c
        for c, f in zip(known, flex)
    ]
    place_labels(ax, xy[:, 0], xy[:, 1], labels, np.where(flex, INK, INK2))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=ROLE_COLORS[r],
                   markersize=7, label=ROLE_LABELS[r]) for r in ROLES
    ] + [plt.Line2D([], [], marker="o", linestyle="", color="#b9b8b2",
                    markeredgecolor=INK, markeredgewidth=1.3, markersize=8,
                    label="Flex (top role < 70%)")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5,
              labelcolor=INK)
    ax.set_title(
        "What a draft model learns about champions\n",
        fontsize=15, color=INK, loc="left", fontweight="bold",
    )
    ax.text(0, 1.015,
            "t-SNE of champion embeddings learned by the lol-meta-tracker v0.8 draft "
            "transformer from 2024–2026 pro drafts.\nColor = majority role. "
            "Ringed champions are flex picks — the model places them between "
            "role clusters on its own. Axes are unitless.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.text(1.0, -0.02, "@lolmetatracker · data: Oracle's Elixir",
            transform=ax.transAxes, fontsize=7.5, color=INK2, ha="right")
    fig.tight_layout()
    CHARTS_DIR.mkdir(exist_ok=True)
    out = CHARTS_DIR / "champion_embeddings_tsne_v08.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
