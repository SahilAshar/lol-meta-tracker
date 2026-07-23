"""Download Oracle's Elixir match data CSV.

Oracle's Elixir hosts yearly CSVs in a public Google Drive folder (linked from
oracleselixir.com/tools/downloads — the old S3 bucket is dead). Known file IDs
are hardcoded below; if a year is missing or an ID goes stale, the folder page
is scraped to rediscover IDs.

Google Drive blocks direct downloads from cloud/CI IPs. We try multiple
URL patterns with a persistent session (cookies) and fall back across them.
"""

import argparse
import re
import sys
import time

import requests

from common import CURRENT_YEAR, DATA_RAW, raw_csv_path

DRIVE_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

FILE_IDS = {
    2020: "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
    2021: "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
}

DOWNLOAD_URLS = [
    # Pattern 1: usercontent endpoint (current default)
    lambda fid: (
        "https://drive.usercontent.google.com/download",
        {"id": fid, "export": "download", "confirm": "t"},
    ),
    # Pattern 2: /uc?export=download (older, sometimes bypasses blocks)
    lambda fid: (
        f"https://drive.google.com/uc",
        {"export": "download", "id": fid, "confirm": "t"},
    ),
    # Pattern 3: direct file download via open link
    lambda fid: (
        f"https://drive.google.com/uc",
        {"export": "download", "id": fid},
    ),
]


def csv_filename(year: int) -> str:
    return f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def _make_session() -> requests.Session:
    """Create a session with browser-like headers for cookie persistence."""
    s = requests.Session()
    s.headers.update(UA)
    s.headers["Accept"] = "text/html,application/xhtml+xml,*/*"
    s.headers["Accept-Language"] = "en-US,en;q=0.9"
    return s


def _extract_confirm_url(html: str) -> str | None:
    """Parse Drive's virus-scan interstitial for the real download link."""
    # Drive sometimes shows a form with a confirm action
    m = re.search(r'href="(/uc\?export=download[^"]+)"', html)
    if m:
        return "https://drive.google.com" + m.group(1).replace("&amp;", "&")
    m = re.search(r'action="([^"]+)"', html)
    if m:
        url = m.group(1).replace("&amp;", "&")
        if "download" in url or "uc" in url:
            if not url.startswith("http"):
                url = "https://drive.google.com" + url
            return url
    return None


def discover_file_id(year: int, session: requests.Session | None = None) -> str:
    """Scrape the public Drive folder listing to find the file ID for a year."""
    s = session or requests
    resp = s.get(DRIVE_FOLDER_URL, headers=UA, timeout=60)
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


def _try_download(session: requests.Session, file_id: str) -> requests.Response | None:
    """Try all URL patterns, returning the first that yields non-HTML content."""
    for i, make_url in enumerate(DOWNLOAD_URLS):
        url, params = make_url(file_id)
        label = f"pattern {i + 1}"
        print(f"  Trying {label}: {url.split('/')[-1]}...")
        resp = session.get(url, params=params, timeout=300, stream=True)
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct:
            print(f"  {label} succeeded (Content-Type: {ct})")
            return resp

        # Got HTML — check for a virus-scan interstitial with a confirm link
        body = resp.text
        confirm_url = _extract_confirm_url(body)
        if confirm_url:
            print(f"  {label} returned interstitial, following confirm link...")
            resp2 = session.get(confirm_url, timeout=300, stream=True)
            resp2.raise_for_status()
            if "text/html" not in resp2.headers.get("Content-Type", ""):
                print(f"  Confirm link succeeded")
                return resp2

        print(f"  {label} returned HTML, trying next...")

    return None


def download(year: int, force: bool = False) -> None:
    dest = raw_csv_path(year)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    file_id = FILE_IDS.get(year)
    if file_id is None:
        print(f"No known file ID for {year}, scraping Drive folder...")
        file_id = discover_file_id(year)
        print(f"Discovered file ID: {file_id}")

    session = _make_session()

    # Warm the session by visiting the folder page first (gets cookies)
    print(f"Downloading {csv_filename(year)} ...")
    try:
        session.get(DRIVE_FOLDER_URL, timeout=30)
    except Exception:
        pass  # non-fatal; we just want cookies if available

    resp = _try_download(session, file_id)

    # If all patterns failed, rescrape for a new file ID and retry
    if resp is None:
        print("All URL patterns returned HTML. Rescraping folder for new file ID...")
        file_id = discover_file_id(year, session)
        print(f"Discovered file ID: {file_id}")
        time.sleep(2)
        resp = _try_download(session, file_id)

    if resp is None:
        raise RuntimeError(
            "Drive keeps returning HTML for all URL patterns — "
            "download may require manual intervention or a different approach"
        )

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
