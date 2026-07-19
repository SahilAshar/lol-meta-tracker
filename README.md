# lol-meta-tracker

Automated cross-league meta tracking for pro League of Legends. Every week, this
pipeline downloads fresh match data from [Oracle's Elixir](https://oracleselixir.com),
computes champion pick/ban rates across the four major leagues — **LCK, LPL, LEC,
LCS** — and publishes charts plus a markdown report.

Nobody does consistent cross-league comparative analysis. This repo does.

**Latest report:** <!-- LATEST-REPORT -->[2026-07-18 meta report](reports/2026-07-18-meta-report.md)<!-- /LATEST-REPORT -->

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
python scripts/weekly_report.py                  # full run: download → analyze → charts → report
python scripts/weekly_report.py --skip-download  # reuse the existing raw CSV
python scripts/weekly_report.py --since 2026-07-22  # explicit analysis window
```

Individual steps:

| Script | What it does |
|---|---|
| `scripts/download_data.py` | Downloads the current-year Oracle's Elixir CSV (public Google Drive) to `data/raw/` |
| `scripts/analyze_meta.py` | Computes pick/ban rates per league + cross-league comparison → `data/processed/` |
| `scripts/generate_charts.py` | Renders pick-rate, ban-rate, and week-over-week charts → `charts/` |
| `scripts/weekly_report.py` | Orchestrates all of the above and writes `reports/YYYY-MM-DD-meta-report.md` |

The analysis window defaults to **2026-07-22** (summer split start: LPL Jul 22,
LEC Jul 24, LCS Jul 25, LCK Jul 29). Until games exist in that window, the
pipeline automatically falls back to the most recent 8 weeks of data.

## Data

- **Source:** Oracle's Elixir yearly CSVs (~165 columns, 12 rows/game: 10 player + 2 team), updated daily
- Raw CSVs (~50MB) are gitignored and re-downloaded on each run; processed aggregates and charts are committed
- Known upstream issue: some 2026 draft/champion-select data is incorrect due to format changes, pending fix

## Weekly automation

Option A — **Claude Code cloud schedule** (runs with laptop closed): in a Claude
Code session, `/schedule` a weekly job (cron `0 9 * * 1`, Mondays 9am) with the
prompt: *"cd ~/Documents/repos/lol-meta-tracker && run python scripts/weekly_report.py,
then commit and push the new report, charts, and processed data."*

Option B — **system crontab** (laptop must be awake):

```cron
0 9 * * 1 cd ~/Documents/repos/lol-meta-tracker && .venv/bin/python scripts/weekly_report.py && git add -A && git commit -m "Weekly meta report $(date +\%F)" && git push
```

Option C — **GitHub Actions**: standard weekly cron YAML running the same
pipeline; free for public repos. (Not yet set up.)
