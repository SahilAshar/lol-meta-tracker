"""Rung-0 gate: do soloq champion co-occurrence embeddings encode role?

Follows docs/2026-07-22-soloq-data-research.md §6, Steps 1-2 (the cheap,
decisive gate). Reads the scraped soloq matches, builds champion embeddings
from unordered pick "bags" via PPMI + truncated SVD (no pick order needed),
then measures 5-NN role purity with the SAME metric as the pro model
(chart_embeddings_v08.py): pro-trained baseline = 0.704, chance ~0.20.

Gate (from §6):
  - purity >= 0.55  -> GO-eligible (Step 3, the 5-seed pro refit, is worth it)
  - purity <  0.40  -> NO-GO, falsified cheaply
  - 0.40..0.55      -> ambiguous middle; report, lean no-go on the gate alone

Primary role per champion is derived from the soloq `pos` tags themselves
(the role each champion is most often played in soloq), so this is fully
self-contained — no Data Dragon, no pro CSVs needed for the gate.

Output: data/processed/soloq_embedding_rung0.json  (metrics + provenance)
        data/processed/soloq_champ_embeddings.npz   (emb matrix + labels)
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "raw" / "soloq" / "soloq.db"
OUT_JSON = REPO / "data" / "processed" / "soloq_embedding_rung0.json"
OUT_NPZ = REPO / "data" / "processed" / "soloq_champ_embeddings.npz"

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
EMB_DIM = 128          # SVD rank for the gate (champ vocab ~170 caps the max)
MIN_GAMES = 30         # drop ultra-rare champs whose geometry is just noise


def load_games() -> list[list[tuple[str, str]]]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT payload FROM matches WHERE state='done' AND payload IS NOT NULL"
    ).fetchall()
    con.close()
    games = []
    for (payload,) in rows:
        parts = json.loads(payload)["participants"]
        bag = [(p["champ"], p.get("pos", "")) for p in parts
               if p.get("champ") and p.get("pos") in ROLES]
        if len(bag) >= 2:
            games.append(bag)
    return games


def main() -> None:
    games = load_games()
    print(f"loaded {len(games)} usable games from {DB.name}")

    # --- champion vocab + primary role from soloq play frequency ------------
    role_counts: dict[str, Counter] = defaultdict(Counter)
    for bag in games:
        for champ, pos in bag:
            role_counts[champ][pos] += 1
    champs = sorted(c for c, rc in role_counts.items() if sum(rc.values()) >= MIN_GAMES)
    idx_of = {c: i for i, c in enumerate(champs)}
    V = len(champs)
    primary = np.array([role_counts[c].most_common(1)[0][0] for c in champs])
    print(f"{V} champions with >= {MIN_GAMES} games")

    # --- full-game co-occurrence -> PPMI ------------------------------------
    # Same-role champions share co-occurrence profiles (they appear alongside
    # the same set of other-role champions), which is the signal role purity
    # reads out.
    cooc = np.zeros((V, V), dtype=np.float64)
    for bag in games:
        present = sorted({idx_of[c] for c, _ in bag if c in idx_of})
        for a_pos in range(len(present)):
            for b_pos in range(a_pos + 1, len(present)):
                a, b = present[a_pos], present[b_pos]
                cooc[a, b] += 1
                cooc[b, a] += 1

    total = cooc.sum()
    row = cooc.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooc * total) / (row * row.T))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)

    # --- embeddings via truncated SVD of PPMI -------------------------------
    dim = min(EMB_DIM, V - 1)
    svd = TruncatedSVD(n_components=dim, random_state=16)
    emb = svd.fit_transform(ppmi)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    # --- 5-NN role purity (identical metric to chart_embeddings_v08.py) -----
    nn = NearestNeighbors(n_neighbors=6).fit(emb)
    _, nbr = nn.kneighbors(emb)
    purity = float(np.mean([
        np.mean(primary[nbr[i, 1:]] == primary[i]) for i in range(V)
    ]))

    # per-role breakdown, to see which roles are crisp vs muddy
    per_role = {}
    for r in ROLES:
        mask = primary == r
        if mask.any():
            per_role[r] = float(np.mean([
                np.mean(primary[nbr[i, 1:]] == primary[i])
                for i in np.where(mask)[0]
            ]))

    if purity >= 0.55:
        verdict = "GO-eligible (purity >= 0.55; run Step 3 pro refit)"
    elif purity < 0.40:
        verdict = "NO-GO (purity < 0.40; falsified cheaply)"
    else:
        verdict = "AMBIGUOUS (0.40-0.55; gate not cleared, lean no-go)"

    result = {
        "games_used": len(games),
        "champions": V,
        "min_games": MIN_GAMES,
        "emb_dim": dim,
        "cooc_pairs_total": float(total),
        "purity_5nn": round(purity, 4),
        "pro_baseline": 0.704,
        "chance": 0.20,
        "per_role_purity": {k: round(v, 4) for k, v in per_role.items()},
        "gate": {"go_ge": 0.55, "nogo_lt": 0.40},
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    np.savez(OUT_NPZ, emb=emb, champs=np.array(champs), primary=primary)

    print(f"\n5-NN role purity: {purity:.3f}  (pro 0.704, chance 0.20)")
    for r in ROLES:
        if r in per_role:
            print(f"  {r:8s} {per_role[r]:.3f}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_NPZ.relative_to(REPO)}")


if __name__ == "__main__":
    main()
