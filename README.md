# lol-meta-tracker

Automated cross-league meta tracking for pro League of Legends. Every day, this
pipeline pulls fresh match data from [Oracle's Elixir](https://oracleselixir.com),
computes champion pick/ban rates across the four major leagues — **LCK, LPL, LEC,
LCS** — and publishes charts plus a markdown report (skipping days when the
source data hasn't changed).

Nobody does consistent cross-league comparative analysis. This repo does.

**Latest report:** <!-- LATEST-REPORT -->[Latest meta report (2026-07-23)](reports/latest.md)<!-- /LATEST-REPORT -->

## What it tracks

- Champion pick/ban rates per league, week over week
- Cross-league comparison: which regions favor which champions
- Week-over-week pick-rate movers (meta shift detection)
- Planned: adoption-lag analysis, sleeper-pick alerts, patch impact, regional playstyle fingerprints

## Setup

Requires Python 3.10+.

```bash
# with uv (recommended)
uv sync

# or plain pip
python -m venv .venv && source .venv/bin/activate
pip install pandas matplotlib seaborn requests plotly
```

## Usage

```bash
python scripts/daily_report.py                  # full run: download → analyze → charts → report
python scripts/daily_report.py --skip-download  # reuse the existing raw CSV
python scripts/daily_report.py --force          # recompute even if source data unchanged
python scripts/daily_report.py --snapshot       # also archive a dated copy of the report
```

Individual steps:

| Script | What it does |
|---|---|
| `scripts/download_data.py` | Downloads the current-year Oracle's Elixir CSV (public Google Drive) to `data/raw/` |
| `scripts/analyze_meta.py` | Computes pick/ban rates per league + cross-league comparison → `data/processed/` |
| `scripts/generate_charts.py` | Renders pick-rate, ban-rate, and week-over-week charts → `charts/` |
| `scripts/daily_report.py` | Orchestrates all of the above and writes `reports/latest.md` (skips recompute when the source CSV hash is unchanged) |

The analysis window defaults to **2026-07-22** (summer split start: LPL Jul 22,
LEC Jul 24, LCS Jul 25, LCK Jul 29). Until games exist in that window, the
pipeline automatically falls back to the most recent 8 weeks of data.

## Data

- **Source:** Oracle's Elixir yearly CSVs (~165 columns, 12 rows/game: 10 player + 2 team), updated daily
- Raw CSVs (~50MB) are gitignored and re-downloaded on each run; processed aggregates and charts are committed
- Known upstream issue: some 2026 draft/champion-select data is incorrect due to format changes, pending fix

## Automation

GitHub Actions runs the pipeline **daily** at 9:23 AM ET
(`.github/workflows/daily-report.yml`): download → hash check → (if the CSV
changed) analyze → charts → `reports/latest.md` → commit + push. Oracle's
Elixir updates roughly daily, so most runs produce a fresh report; unchanged
days exit in seconds without committing. Monday runs also archive a dated
snapshot to `reports/` for week-over-week history.
