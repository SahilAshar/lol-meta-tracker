# Roadmap & Pickup Notes

*Last updated: 2026-07-20 (v0.8.1 per-type blend + debut draft + artifact
refresh session). This doc is the "where were we" file — read it first when
resuming work.*

## Current state (all working, all verified)

- **Pipeline**: `download → analyze → charts → report`, orchestrated by
  `scripts/daily_report.py`. Runs locally via `.venv/bin/python scripts/daily_report.py`.
- **Automation**: GitHub Actions daily at 9:23 AM ET (`.github/workflows/daily-report.yml`).
  Hash-checks the source CSV; skips recompute + commit when unchanged (verified both
  paths in CI). Monday runs archive dated snapshots to `reports/`.
- **Data**: Oracle's Elixir 2026 CSV via public Google Drive folder (the old S3
  bucket is dead — see `scripts/download_data.py` for file IDs + stale-ID rediscovery).
  Full-column 4-league parquet archive committed at `data/processed/major_leagues.parquet`.
- **Analysis window**: date-based (not split names — they're inconsistent across
  leagues). Defaults to 2026-07-22 (summer start); auto-falls-back to last 8 weeks
  until then.
- **Draft next-pick model**: `scripts/draft_dataset.py` (now multi-year:
  `--years 2024 2025 2026`, per-league-per-year fearless detection, plus a
  compact `draft_sequences*.parquet` for sequence models) +
  `scripts/train_draft_model.py` (v0.7 GBM lineage) +
  `scripts/draft_transformer.py` / `train_draft_model_v08.py` (v0.8).
  Test set is now the EWC July 2026 main event **including finals: 50 games /
  1000 decisions** (was 43/860; year guard matters — EWC 2024/2025 July games
  exist in the multi-year data).
  - **v0.7 on the grown test** (2026-only train): 13.0/32.5/42.8 vs meta
    9.9/26.6/38.0. Finals games were unusually draft-predictable; every model
    rose. Caveat: refit v0.6 (13.0/33.5/45.3) and v0.5 (13.2/32.1/43.6) single
    fits now match/beat the ensemble at top-3/5 — the ±1.5pt single-fit noise
    band in action.
  - **v0.8** = small causal transformer over the 20-slot draft sequence,
    learned weight-tied champion embeddings, d192x4L6H, 5-seed mean-prob
    ensemble, trained on 2024-2026 (102,916 decisions); config + 0.25-blend
    with the v0.7 GBM ensemble selected on val only (`experiment_v08.py`).
    Blind test: **transformer solo 10.2/22.7/30.9 — does NOT beat the
    multi-year GBM ensemble (14.9/32.8/45.2)**; the blend ties it overall
    (14.1/32.6/44.5) and clearly wins picks (17.2/35.2/48.0 vs 14.8/32.2/45.4
    top-1/3/5). Division of labor is stark: transformer owns picks, GBM owns
    bans (transformer bans 7.6 top-1 vs GBM 15.0 — it is date-blind and can't
    see current meta). Multi-year training itself lifted the v0.7 feature set
    +1.9 top-1 over 2026-only. Full lineage blocks in
    `data/processed/draft_model_metrics_v08.json`.
  - **v0.8.1** = per-decision-type blend weights (picks w=0.75 transformer,
    bans w=0.0 i.e. pure GBM), selected on the extended experiment_v08 val
    sweep (picks val top-1 19.1 at 0.75; bans 14.3 at 0). Single test look:
    **16.0/33.1/44.6 all — best top-1 of any lineage** (GBM 14.9, flat blend
    14.1); picks 17.0/32.8/44.2, bans 15.0/33.4/45.0. Honest caveats, also
    flagged in the metrics json: (a) this was the SECOND evaluation against
    the same EWC test set — frozen now, no further iteration against it;
    (b) the picks val edge did not transfer (17.0 vs the flat blend's 17.2
    top-1, and behind at top-3/5) — the whole overall gain is bans returning
    to the GBM. Split sizes from the run: train 100,836 / val 1,080 / test
    1,000 decisions (the previously quoted 102,916 is the three-way total).
    Recomputed v0.8/v0.7 blocks reproduced the stored json exactly.
  - **Embeddings learned real structure**: 5-NN role purity 0.682 vs 0.20
    chance; `charts/champion_embeddings_tsne_v08.png` (debut-thread artifact)
    shows role clusters with flex picks (Poppy, Camille, Sett, Nasus, TF)
    sitting between them, no role labels ever shown to the model. Labels are
    now collision-aware and a dark variant renders via
    `chart_embeddings_v08.py --dark` (both under 400KB for data-URI embeds).
  - Codespaces note: "standardLinux32gb" = 16GB **RAM** (32 = storage); the
    GBM stages need the slim-dtype loading in `experiment_v08.load_multi` or
    they get OOM-killed.

## Open loops (near-term, in order)

1. ~~Score the EWC finals predictions~~ **Done 2026-07-20** — scorecard appended
   to `reports/2026-07-19-ewc-finals-preview.md`, verified against the CSV +
   gol.gg/Liquipedia. Both series winners hit (Gen.G 2-1 exact; DK won but 3-0
   not 3-2); draft reads ~5/9 (ban triangle + Nocturne paradox excellent; wrong
   whenever a read required a team to fear what the market feared). 2024/2025
   CSVs came from a GitHub LFS mirror
   (`Matthew-Paoletta/The-Snowballing-Effect`) while Drive was quota-blocked;
   the 2026 quota cleared after ~5h of spaced retries.
2. ~~Draft model rung 3 (transformer)~~ **Done 2026-07-20** — see Current state.
   Honest verdict: at this data size the transformer does not beat the well-fed
   GBM ensemble; it is a picks specialist and an embeddings machine. Next rungs
   worth trying, in order of expected value:
   - ~~Per-decision-type blend weights~~ **Done 2026-07-20 as v0.8.1** — see
     Current state. Best overall top-1 (16.0); gain came from bans, not picks.
   - **Give the transformer time signal** for bans: patch embedding or the
     trailing meta-rate features injected at the output layer — its 7.6 ban
     top-1 vs GBM's 15.0 is entirely current-meta blindness.
   - **Series-prior tokens** (fearless context: which champs each side burned
     earlier in the series) — currently only enters availability masks.
3. ~~Write the "meta entering summer" debut piece~~ **Drafted 2026-07-20** —
   `drafts/2026-07-20-meta-entering-summer-debut-thread.md`: 10-tweet thread
   with full stat-provenance appendix (16.13 intl sample is now 121 games
   incl. finals; leads: Camille is 34/34 support at 44% WR, ban quartet
   Poppy/Vi/Jayce/Orianna, Nocturne 60 bans at 31% WR, blue side 61%).
   NOT posted. Blockers before posting: grab the @lolmetatracker handle and
   give the finals scorecard a public link (README anchor or gist). The
   claude.ai explainer artifact ("How the Draft Model Actually Works") was
   refreshed in place the same day: v0.7/v0.8/v0.8.1 scoreboard, finals
   scorecard section, collision-fixed t-SNE embedded light+dark.
4. **Summer split coverage begins**: LPL Jul 22, LEC Jul 24, LCS Jul 25, LCK Jul 29.
   The default window goes live automatically. First "Week 1 cross-league" report
   ~Aug 3-4 → Reddit/Twitter debut per the engagement plan.
5. **Riot personal API key** — request at developer.riotgames.com (few days'
   approval lead time). Blocks the Challenger leading-indicator feature.

## Planned features (decided order, per docs/riot-api-assessment.md)

1. **Data Dragon integration** (2-4h): champion icons + `versions.json` patch
   timeline → unlocks patch-impact analysis. Bundle with dashboard when we get there.
2. **Challenger leading-indicator** (8-12h, needs API key): daily sample of top-50
   Challenger players × 4 regions; correlate solo-queue pick spikes with later pro
   adoption. Nobody publishes this — the differentiating analytical content.
3. **Adoption-lag + sleeper-pick detection** on pro data alone (no API needed):
   which region picks a champion first, who follows, what spikes in one region only.
4. **lolesports.com unofficial API** (6-10h): schedules/standings/near-live results.
   Enhancement layer only — graceful fallback, never a dependency.
5. **HTML dashboard on GitHub Pages** — deliberately paused (2026-07-19 decision:
   analysis first, visuals later). Plotly static export regenerated by the daily
   Action; panels sketched: pick/ban explorer, WoW trends, regional fingerprints.

## Not doing

- **Own S3 mirror / Postgres** — rejected 2026-07-19. The committed parquet is the
  archive; DuckDB over parquet is the future query layer if Phase 2 (RAG/search)
  happens. Postgres only if serving a live API to others.
- **GRID** — closed to community projects (Open Access = CS2/Dota only). Recheck ~2027.
- **Official Riot API for pro match data** — confirmed dead end; it has no pro
  match endpoints. Oracle's Elixir remains the only practical source.

## Known issues

- ~~Oracle's Elixir flagged some 2026 draft/champion-select data as incorrect~~
  **Resolved**: this was the "Side-Selection Bug" (blue side no longer guaranteed to
  pick first; parser assumed it was) — fix deployed + broken games reparsed as of
  Jan 18, 2026 per OE's announcement. Remove the footnote from report templates.
  Spot-checks vs. Leaguepedia remain cheap insurance for draft work. Modeling rule
  that survives the fix: derive pick order from draft columns, never infer from side.
  Full stat glossary now at `docs/oracle-elixir-definitions.md`.
- The ~50MB download happens every daily run even when data is unchanged
  (unavoidable — hash requires the file; Drive gives no usable checksum header).
- LPL 2025 shows 8 champion overlaps across 797 same-series game pairs, so
  per-year fearless detection marks it non-fearless even though most of the
  season was fearless (format changed mid-year / a few irregular series). Cost
  is a few phantom candidates in those games' candidate sets — acceptable;
  per-split detection would fix it if it ever matters.
