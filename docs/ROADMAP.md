# Roadmap & Pickup Notes

*Last updated: 2026-07-19 (draft model v0.7 session). This doc is the "where
were we" file — read it first when resuming work.*

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
- **Draft next-pick model**: `scripts/draft_dataset.py` + `scripts/train_draft_model.py`.
  v0.7 (pair synergy/counter features + a 10-model ensemble: 5 seeds x
  {HistGBM classifier, LGBM LambdaRank}, equal-weight rank-averaged) blind-tested
  on the EWC July main event: top-1/3/5 = 11.2/32.0/41.4 vs meta 8.4/25.9/36.4
  (v0.6: 10.7/31.7/42.8; v0.5: 11.7/29.3/40.9). Picks 12.3/33.0/42.3; bans
  10.0/30.9/40.5 keep beating meta bans (9.1/30.2/40.7) at top-1/3, tie top-5.
  Config chosen on val only (`scripts/experiment_v07*.py`): ranker family wins
  top-1, classifier family wins top-5, blend dominates both; ensembling kills the
  ±1.5pt top-1 seed/platform noise any single fit shows. Comparison in
  `data/processed/draft_model_metrics_v07.json` (v0.6/v0.5/v0 blocks; auto-refits
  older feature sets if the test set grows). Runs in Codespaces via
  `.devcontainer/` (lightgbm needs libomp locally on macOS — decided to keep the
  Mac clean).

## Open loops (near-term, in order)

1. **Score the EWC finals predictions** — `reports/2026-07-19-ewc-finals-preview.md`
   predicted Gen.G 2-1 over T1 and Dplus Kia 3-2 over Karmine Corp, plus draft
   reads (Nocturne paradox, expected bans). Finals were July 19; results land in
   the CSV ~July 20. Append an honest scorecard section to the preview.
   NOTE 2026-07-19: Drive download is returning "Quota exceeded" for the 2026 CSV
   (transient, popular-file limit — not a stale ID; still failing as of the v0.6
   session, evening of 7/19). Retry the download before scoring; also re-run the
   draft blind test once finals games land (test set grows past 43 games —
   `train_draft_model.py` now auto-refits the v0.5 feature set for comparability
   when that happens).
2. **Draft model rung 3** — v0.7 (pair features + clf/ranker seed ensemble) done;
   GBM ceiling now well established. Next per the ladder is the small transformer
   with learned champion embeddings (t-SNE of embeddings = flex/role clusters).
   Pair it with pulling the 2024-2025 CSVs (file IDs already in download_data.py)
   — 31k decisions from 2026 alone is thin for sequence models. Observed v0.7
   lesson: val gains compress on test; picks top-5 dipped vs v0.6 (42.3 vs 45.1)
   while top-1/3 improved — report both when comparing. Data-quality note: OE
   scrambled position labels in one Gen.G game (2026-07-17) — roster inference
   guards against this via majority-role assignment, but spot-checks vs
   Leaguepedia stay cheap insurance.
3. **Write the "meta entering summer" debut piece** — full EWC + MSI patch 16.13
   sample (115 international games), framed for @lolmetatracker's first thread
   (handle being grabbed). Prediction scorecard = credibility receipt.
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
