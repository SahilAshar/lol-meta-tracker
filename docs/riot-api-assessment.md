# Riot API Value-Add Assessment for lol-meta-tracker

*Researched 2026-07-19. Question: does access to Riot's APIs add anything valuable
beyond what Oracle's Elixir already provides?*

## 1. Official Riot Games API (developer.riotgames.com)

**Does it expose pro/esports match data?** No. This is the single biggest misconception to clear up. Pro games (LCK/LPL/LEC/LCS) are played on Riot's internal tournament infrastructure, and the public API has no read endpoint for arbitrary esports matches. The **Tournament API** (`tournament-v5`) sounds relevant but isn't — it's for *creating your own* custom tournaments (you generate tournament codes, players join lobbies you spin up, you get results back for *those* games). It cannot retrieve LCK/LPL games that already happened. `match-v5` only returns matches tied to a `puuid` — ranked, normal, ARAM, and your own tournament-code games. There is no "give me all LEC games from patch 26.13" endpoint.

**Rate limits:**
- **Development key** (auto-issued, no approval): expires every 24h, prototyping only.
- **Personal key** (approved, ongoing): 20 req/1s, 100 req/2min, per region. **Cannot access the Tournament API.** No public-facing product allowed.
- **Production key** (requires a working demo + approval): 500 req/10s, 30,000 req/10min, per region. Gets Tournament API access (still only for self-created tournaments) and can be raised further for apps in good standing.

**Verdict:** For pulling *pro* match data, this API adds **nothing** — Oracle's Elixir is already the better (and only practical) source. Skip entirely for that purpose.

## 2. Unofficial lolesports.com API (esports-api.lolesports.com / feed.lolesports.com)

This is the API that lolesports.com's own website uses, reverse-engineered and documented by the community for years. It uses a long-standing shared public key (`x-api-key: 0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z`) that's been stable and widely reused across dozens of open-source wrappers (`rigelifland/lolesports_api`, `Pupix/lol-esports-api`, etc.) for several years — a good stability signal despite being unofficial.

**Endpoints of interest:**
- `getSchedule` / `getLive` — upcoming and in-progress matches, best-of format, stream links. **This is genuinely useful**: it tells you exactly when LCK/LPL/LEC/LCS games are happening, so your automation can trigger right after a match ends instead of guessing or polling OE's daily CSV blind.
- `getWindow` / `getDetails` — near-real-time in-game stats (gold, kills, objectives, items) while a game is live or immediately after. This is **faster than Oracle's Elixir**, which is a daily-batch aggregate — OE itself is sourced partly from this same lolesports.com pipeline plus Leaguepedia/lpl.qq.com, so it's authoritative but inherently lagged by up to a day.
- `getStandings`, `getTeams`, `getLeagues` — league/tournament metadata, team rosters, standings tables. OE's CSVs don't include standings or roster/team metadata at all — this fills a real gap for any "league standings" widget on a dashboard.
- `getEventDetails` — VOD links, which OE doesn't have.

**Value-add beyond OE:** (a) precise match-timing signal for scheduling automation, (b) same-day/near-live results instead of next-day, (c) standings/VODs/team metadata OE simply doesn't carry.

**Risk:** Unofficial and reverse-engineered — no ToS, no SLA, Riot could rotate the key or change response shapes without notice (it has changed schema before across multi-year history). Treat as a **nice-to-have enhancement, not a dependency** — wrap calls defensively and fall back to OE-only mode if it breaks.

**Effort:** 6–10 hours (well-documented, several reference Python implementations to crib from; mainly wiring `getSchedule` for automation timing + `getStandings` for a dashboard widget).

## 3. Data Dragon / Community Dragon

**Data Dragon** (`ddragon.leagueoflegends.com`) is Riot's official, sanctioned static-data CDN — champion JSON, splash art, square icons, and a `versions.json` listing every patch version and its release ordering. No auth, no rate limit concerns, cacheable indefinitely per patch, explicitly intended for third-party use like this.

**Value-add:**
- Champion square icons for charts/tables (pick/ban reports, dashboard) — turns a bare champion-name list into something visually parseable at a glance.
- `versions.json` gives a clean patch-version timeline to join against OE's per-match dates for **patch impact analysis** (a planned feature; Data Dragon is the natural source for "when did patch X ship" ground truth).

**Community Dragon** is a community-maintained superset (higher-res assets, more granular data) — useful if Data Dragon's assets feel too coarse for a polished HTML dashboard, but it's an unofficial mirror so treat Data Dragon as primary.

**Risk:** Essentially none. Official, stable, free, no key required.

**Effort:** 2–4 hours (fetch versions.json, cache champion.json + icon URLs, map champion names/IDs to OE's naming).

## 4. Solo-queue-as-leading-indicator (Challenger pick trends)

**Feasibility:** High, and cheaper than it sounds. Pull `league-v4`'s Challenger endpoint once per region (4 calls total for KR/CN/EUW/NA) to get the ~200–300 highest-elo players, then for a sampled subset (say top 50/region) pull recent match IDs via `match-v5` by-puuid and dedupe match details. A daily pull of ~50 players × 4 regions, ~10 new games each, is on the order of 2,000–2,500 API calls — trivially inside the personal-key budget of 100 req/2min sustained (≈72k/day theoretical ceiling). This is a well-supported, officially-sanctioned use case (personal key, non-commercial, no Tournament API needed) — no ToS risk.

**Does this already exist?** Raw Challenger stats exist all over (op.gg, Mobalytics, u.gg, lolalytics) but none of them expose a public API, and none frame it as an **adoption-lag / leading-indicator signal against pro play**. That correlation — "champion X spiked in Korean Challenger, pro adoption followed N days later" — is not something any existing public tool surfaces. Combined with OE pro-pick data, this is genuinely differentiating content and maps directly onto two features already on the roadmap (adoption-lag analysis, sleeper-pick detection). This is arguably the **highest-value single integration** on this list, because it's not just "nice dashboard polish" — it's new analytical content nothing else provides.

**Effort:** 8–12 hours (key request/approval lead time + building the sampling/pull/cache pipeline + the lag-correlation logic itself, which is really where the project value lives, not the API call).

**Risk:** Low — official, sanctioned, well within personal-key limits if you sample rather than exhaustively pull every Challenger player's full history.

## 5. GRID / official esports data portal

Confirmed still restricted for this project's purposes. GRID's "Open Access" free-tier program (real-time stats, no cost, targeted at indie devs/students/researchers) currently covers **CS2 and Dota 2 only** — League of Legends is explicitly not yet included ("more titles coming soon," no ETA given). Separately, GRID's League of Legends Data Portal (LDP) exists but is scoped to: free access for pro teams (ERL1+, for their own scrim/match data) and paid commercial access for betting/fantasy/fan-engagement partners. Riot has stated an intention to roll LDP out to the broader community for non-commercial use, but that has not happened yet — no application path, no free community tier exists today.

**Verdict:** Not accessible for a personal/community project right now. Worth re-checking LDP's community rollout status in 6–12 months, but not actionable today.

## Ranked Recommendation

1. **Data Dragon** (2–4 hrs, near-zero risk) — do this first. Immediate visual payoff for the dashboard and directly unlocks the patch-impact-analysis feature.
2. **Solo-queue leading-indicator via official Riot API, personal key** (8–12 hrs, low risk) — highest strategic value. The one integration that produces genuinely new analytical content (adoption-lag / sleeper-pick detection) rather than polishing what OE already gives.
3. **Unofficial lolesports.com API** (6–10 hrs, moderate/unofficial risk) — match-timing automation and standings/live-results widget. Build as an optional enhancement layer with graceful fallback to OE, not a dependency.
4. **GRID** — skip; nothing to integrate today. Revisit if/when LDP opens to community/non-commercial devs.
5. **Official Riot API for pro match data** — dead end; documented here so it isn't accidentally pursued.

## Sources

[Riot Developer Portal](https://developer.riotgames.com/apis) · [Portal docs](https://developer.riotgames.com/docs/portal) · [Production Key Applications](https://support-developer.riotgames.com/hc/en-us/articles/22801383038867-Production-Key-Applications) · [Tournament API](https://developer.riotgames.com/tournament-api.html) · [Unofficial Lolesports API docs](https://vickz84259.github.io/lolesports-api-docs/) · [lolesports API gist](https://gist.github.com/levi/e7e5e808ac0119e154ce) · [rigelifland/lolesports_api](https://github.com/rigelifland/lolesports_api) · [Data Dragon docs](https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html) · [HexTechDocs Data Dragon](https://hextechdocs.dev/data-dragon/) · [GRID Open Access](https://grid.gg/open-access/) · [GRID LoL Data Portal](https://grid.gg/get-league-of-legends/) · [GRID/Riot partnership](https://grid.gg/riot-games-and-grid-announce-exclusive-global-esports-data-partnership/) · [Oracle's Elixir FAQ](https://oracleselixir.com/faq)
