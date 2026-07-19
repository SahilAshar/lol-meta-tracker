"""Download Oracle's Elixir match data CSV.

Oracle's Elixir hosts yearly CSVs in a public Google Drive folder (linked from
oracleselixir.com/tools/downloads — the old S3 bucket is dead). Known file IDs
are hardcoded below; if a year is missing or an ID goes stale, the folder page
is scraped to rediscover IDs.
"""

import argparse
import re
import sys

import requests

from common import CURRENT_YEAR, DATA_RAW, raw_csv_path

DRIVE_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"

FILE_IDS = {
    2020: "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
    2021: "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def csv_filename(year: int) -> str:
    return f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def discover_file_id(year: int) -> str:
    """Scrape the public Drive folder listing to find the file ID for a year."""
    resp = requests.get(DRIVE_FOLDER_URL, headers=UA, timeout=60)
    resp.raise_for_status()
    html = resp.text
    idx = html.find(csv_filename(year))
    if idx == -1:
        raise FileNotFoundError(
            f"{csv_filename(year)} not found in Drive folder {DRIVE_FOLDER_URL}"
        )
    ids = re.findall(r'"([A-Za-z0-9_-]{28,44})"', html[max(0, idx - 2000) : idx])
    if not ids:
        raise RuntimeError(f"Could not extract a file ID for {year} from folder HTML")
    return ids[-1]


def download(year: int, force: bool = False) -> None:
    dest = raw_csv_path(year)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    file_id = FILE_IDS.get(year)
    if file_id is None:
        print(f"No known file ID for {year}, scraping Drive folder...")
        file_id = discover_file_id(year)
        print(f"Discovered file ID: {file_id}")

    params = {"id": file_id, "export": "download", "confirm": "t"}
    print(f"Downloading {csv_filename(year)} ...")
    resp = requests.get(DOWNLOAD_URL, params=params, headers=UA, timeout=300, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        # Drive returned an interstitial page instead of the file — the
        # hardcoded ID is likely stale. Retry once with a freshly scraped ID.
        print("Got HTML instead of CSV (stale file ID?), rescraping folder...")
        file_id = discover_file_id(year)
        params["id"] = file_id
        resp = requests.get(DOWNLOAD_URL, params=params, headers=UA, timeout=300, stream=True)
        resp.raise_for_status()
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError("Drive keeps returning HTML — download flow may have changed")

    tmp = dest.with_suffix(".csv.part")
    size = 0
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            size += len(chunk)

    # Sanity-check before replacing any existing good file
    with open(tmp, errors="ignore") as f:
        header = f.readline()
    if "gameid" not in header or "league" not in header:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file does not look like Oracle's Elixir match data")

    tmp.replace(dest)
    print(f"Saved {dest} ({size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=CURRENT_YEAR)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    if raw_csv_path(args.year).exists() and not args.force:
        print(f"{raw_csv_path(args.year)} already exists; re-downloading to get latest games.")
    download(args.year, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
