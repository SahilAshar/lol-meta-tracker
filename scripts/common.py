"""Shared constants and paths for the lol-meta-tracker pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CHARTS_DIR = REPO_ROOT / "charts"
REPORTS_DIR = REPO_ROOT / "reports"

MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCS"]

# Summer 2026: LPL opens July 22, LEC July 24, LCS July 25, LCK July 29.
DEFAULT_SINCE = "2026-07-22"

CURRENT_YEAR = 2026


def raw_csv_path(year: int = CURRENT_YEAR) -> Path:
    return DATA_RAW / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
