"""Daily pipeline orchestrator: download → analyze → charts → markdown report.

Runs daily; skips all computation if the source CSV is unchanged since the
last run (Oracle's Elixir updates roughly once a day, but not on a fixed
schedule). Always writes reports/latest.md; with --snapshot (Mondays in CI)
also writes a dated reports/YYYY-MM-DD-meta-report.md for the archive.

Usage:
  python scripts/daily_report.py                   # full run
  python scripts/daily_report.py --skip-download   # reuse existing raw CSV
  python scripts/daily_report.py --force           # recompute even if unchanged
  python scripts/daily_report.py --snapshot        # also write dated snapshot
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from common import DATA_PROCESSED, MAJOR_LEAGUES, REPO_ROOT, REPORTS_DIR, raw_csv_path

HASH_FILE = DATA_PROCESSED / "source.sha256"

SCRIPTS = Path(__file__).resolve().parent


def source_hash() -> str:
    h = hashlib.sha256()
    with open(raw_csv_path(), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_step(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print(f"\n=== {script} ===")
    subprocess.run(cmd, check=True)


def md_rate_table(df: pd.DataFrame, rate_col: str, count_col: str, top: int = 10) -> str:
    lines = ["| Champion | Rate | Games |", "|---|---|---|"]
    for _, row in df.head(top).iterrows():
        lines.append(f"| {row['champion']} | {row[rate_col]:.0%} | {int(row[count_col])} |")
    return "\n".join(lines)


def build_report(report_date: str, snapshot: bool = False) -> Path:
    meta = json.loads((DATA_PROCESSED / "meta.json").read_text())
    pick_rates = pd.read_csv(DATA_PROCESSED / "pick_rates.csv")
    ban_rates = pd.read_csv(DATA_PROCESSED / "ban_rates.csv")
    comparison = pd.read_csv(DATA_PROCESSED / "pick_rate_comparison.csv")

    parts = [f"# Cross-League Meta Report — {report_date}", ""]
    window_desc = f"Games from **{meta['since']}** through **{meta.get('window_end') or 'n/a'}**."
    if meta.get("fallback"):
        window_desc += f"\n\n> ⚠️ {meta['fallback_reason']}. Summer split coverage begins once leagues start (LPL Jul 22 → LCK Jul 29)."
    parts += [window_desc, "", "## Games analyzed", ""]
    parts += ["| League | Games |", "|---|---|"]
    for lg in MAJOR_LEAGUES:
        n = meta["games_per_league"].get(lg, 0)
        parts.append(f"| {lg} | {n if n else '— (not started)'} |")

    parts += ["", "## Charts", "",
              "![Pick rates by region](../charts/pick_rate_by_region.png)", "",
              "![Ban rates by region](../charts/ban_rate_by_region.png)", ""]
    if (REPO_ROOT / "charts" / "week_over_week_delta.png").exists():
        parts += ["![Week-over-week movers](../charts/week_over_week_delta.png)", ""]

    parts += ["## Cross-league pick rates (top 15 by mean)", ""]
    header = "| Champion | " + " | ".join(MAJOR_LEAGUES) + " | Mean |"
    parts += [header, "|" + "---|" * (len(MAJOR_LEAGUES) + 2)]
    for _, row in comparison.head(15).iterrows():
        cells = [
            f"{row[lg]:.0%}" if pd.notna(row[lg]) else "—" for lg in MAJOR_LEAGUES
        ]
        parts.append(f"| {row['champion']} | " + " | ".join(cells) + f" | {row['mean_pick_rate']:.0%} |")

    for lg in MAJOR_LEAGUES:
        lg_picks = pick_rates[pick_rates["league"] == lg]
        lg_bans = ban_rates[ban_rates["league"] == lg]
        parts += ["", f"## {lg}", ""]
        if lg_picks.empty:
            parts.append("_No games in window yet._")
            continue
        parts += ["**Top picks**", "", md_rate_table(lg_picks, "pick_rate", "games_picked"), ""]
        parts += ["**Top bans**", "", md_rate_table(lg_bans, "ban_rate", "games_banned")]

    parts += ["", "---",
              "_Data: [Oracle's Elixir](https://oracleselixir.com). "
              "Note: some 2026 draft/champion-select data has known issues pending upstream fix._", ""]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "latest.md"
    out.write_text("\n".join(parts))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    if snapshot:
        dated = REPORTS_DIR / f"{report_date}-meta-report.md"
        dated.write_text("\n".join(parts))
        print(f"Wrote snapshot {dated.relative_to(REPO_ROOT)}")
    return out


def update_readme_latest(report_path: Path, report_date: str) -> None:
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return
    rel = report_path.relative_to(REPO_ROOT)
    link = f"[Latest meta report ({report_date})]({rel})"
    text = readme.read_text()
    new = re.sub(
        r"(<!-- LATEST-REPORT -->).*?(<!-- /LATEST-REPORT -->)",
        rf"\g<1>{link}\g<2>",
        text,
        flags=re.DOTALL,
    )
    if new != text:
        readme.write_text(new)
        print("Updated Latest Report link in README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--since", default=None, help="Override analysis window start")
    parser.add_argument("--force", action="store_true", help="Recompute even if source CSV unchanged")
    parser.add_argument("--snapshot", action="store_true", help="Also write dated report snapshot")
    args = parser.parse_args()

    if not args.skip_download:
        run_step("download_data.py")

    new_hash = source_hash()
    old_hash = HASH_FILE.read_text().strip() if HASH_FILE.exists() else None
    if new_hash == old_hash and not args.force:
        print("Source CSV unchanged since last run — nothing to recompute.")
        return

    analyze_args = ["--since", args.since] if args.since else []
    run_step("analyze_meta.py", *analyze_args)
    run_step("generate_charts.py")

    report_date = date.today().isoformat()
    report = build_report(report_date, snapshot=args.snapshot)
    update_readme_latest(report, report_date)
    HASH_FILE.write_text(new_hash + "\n")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
