# Debut thread — "The meta entering summer" (@lolmetatracker)

**Status: DRAFT — do not post.** Handle still being grabbed. Attach
`charts/champion_embeddings_tsne_v08.png` to tweet 8. Links to the finals
scorecard need a public home (GitHub README anchor or gist) before posting.
Every stat below traces to a source in the appendix.

---

## The thread

**1/**
The day before EWC finals, we published predictions with numbers attached.
Both series winners called — Gen.G 2-1 over T1 *exact*, Dplus Kia over
Karmine Corp (we said 3-2; they swept). Full scorecard, hits AND misses,
scored line by line: [link]

This account grades itself. Always.

**2/**
Summer split starts this week. Here's the meta entering it — built from 121
international games on patch 16.13 (MSI + EWC), Oracle's Elixir data, all
four major regions about to come back online.

**3/**
The ban quartet that defined 16.13: Poppy (78 bans), Vi (67), Jayce (64),
Orianna (60). Vi led the event at 75% presence.

When Orianna actually got through? 62% win rate over 26 games. The bans are
rational.

**4/**
Camille is a support now. Not "flex tech" — a support.

Spring domestic play: 64 games top, 23 support. International 16.13: all 34
Camille games were support... at a 44% win rate. The most contested new idea
of the summer, and so far it's *losing*. Watch who imports it anyway.

**5/**
The Nocturne paradox, extended: 60 bans — tied-4th most at the event — and a
31% win rate in the 16 games it played. The most feared champion on the
patch has the worst record of any meta jungler.

In the grand final, KC burned a ban on it against a team that never plays it.

**6/**
The quiet winners nobody talks about:
- Gnar: 30 games, 63% WR (top)
- Ezreal: 36 games, 58% WR
- Akali: 47 bans, and an 82% WR in the 11 games that slipped through

And the opposite: Ryze — 30 games, 37% WR. The default first pick that keeps
not paying rent.

**7/**
Blue side won 61% of games on 16.13 internationals.

Summer watch item #1: does that survive the patch boundary when the regional
leagues come back? Side-selection edges this large usually get patched or
drafted away. Usually.

**8/** *(attach: champion_embeddings_tsne_v08.png)*
We trained a small transformer on ~103k pro draft decisions (2024–26). We
never told it champion roles.

It learned them anyway — 5-NN role purity 0.682 vs 0.20 by chance. And the
flex picks (Poppy, Camille, Sett, Nasus, TF) literally sit *between* the
role clusters. The map drew itself.

**9/**
Honest model note: that transformer does NOT beat our boring gradient-boosted
ensemble overall. But it owns *picks* — 17.2% top-1 on exact-next-pick vs
14.8 for the GBM, blind-tested on EWC. Bans still belong to plain meta
stats. Division of labor, measured, published.

**10/**
Week 1 calendar: LPL Jul 22 · LEC Jul 24 · LCS Jul 25 · LCK Jul 29.

We'll track pick/ban shifts week over week and region by region, and score
every claim we make — first cross-league report early August. Follow along.

---

## Appendix — stat provenance

All CSV numbers computed 2026-07-20 from
`data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv`, filtered to
`patch == 16.13 & league in {MSI, EWC}` (121 unique gameids; the finals
preview's "115 games" was the same filter through July 18, before the 6
finals-day games).

| Claim | Source |
|---|---|
| Both series winners called; Gen.G 2-1 exact; DK swept 3-0 (predicted 3-2) | `reports/2026-07-19-ewc-finals-preview.md` scorecard (committed, verified vs OE/gol.gg/Liquipedia) |
| 121 games on 16.13 intl | CSV query above |
| Bans: Poppy 78, Vi 67, Jayce 64, Orianna 60, Nocturne 60 | ban1–ban5 value counts, team rows |
| Vi 75% presence (75.2) | (unique pick games + bans) / 121 |
| Orianna 26 games, 62% WR | player rows groupby champion |
| Camille intl: 34 games, all support, 44% WR; spring: 64 top / 23 sup / 3 jng | player rows by position, `date < 2026-06-01` for spring |
| Nocturne 16 games, 31% WR | player rows |
| Gnar 30g 63% (all top); Ezreal 36g 58%; Akali 11g 82% + 47 bans; Ryze 30g 37% | player rows + ban counts |
| Blue side 61% (61.2) | team rows, side == Blue, mean(result) |
| ~103k decisions (102,916), 2024–26 training | `docs/ROADMAP.md` v0.8 block / `draft_model_metrics_v08.json` lineage |
| 5-NN role purity 0.682 vs 0.20 | `docs/ROADMAP.md` v0.8 block; `charts/champion_embeddings_tsne_v08.png` |
| Picks top-1 17.2 vs 14.8 (blend vs GBM, EWC blind test) | `data/processed/draft_model_metrics_v08.json` → v0.8_blend / v0.7_refit_multi → test_ewc_main → picks.top1 |
| Split start dates | project plan (roadmap open loop 4) |

**Pending update:** if v0.8.1 (per-decision-type blend) lands with a better
picks number, tweet 9 should cite v0.8.1 instead — same provenance file.
