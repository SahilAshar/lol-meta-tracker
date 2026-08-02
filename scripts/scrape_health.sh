#!/bin/bash
# Per-region health check for a running soloq scrape.
# Usage: bash scripts/scrape_health.sh [seconds_between_samples]
# Catches the failure mode where one region wedges at 0 new matches while the
# total still looks healthy. Restart is lossless — the SQLite state machine
# resumes exactly where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
DB=data/raw/soloq/soloq.db
GAP="${1:-120}"

q() { sqlite3 "$DB" "$1"; }
snap() { q "SELECT region||' '||COUNT(*) FROM matches WHERE state='done' GROUP BY region"; }

echo "=== process ==="
ps -eo pid,etime,command | grep "[r]iot_soloq_scrape.py" || echo "!! NO SCRAPER RUNNING"
echo "   (kill by explicit PID only — 'pkill -f python3 ...' matches nothing)"

echo "=== power ==="
pmset -g batt | head -2

LOG=$(ls -t logs/*.log 2>/dev/null | head -1 || true)
if [ -n "$LOG" ]; then
  echo "=== log: $LOG ==="
  echo "regions started: $(grep -c 'ladder players' "$LOG") lines"
  grep 'ladder players' "$LOG" | tail -3
  echo "FATAL count: $(grep -c FATAL "$LOG" || true)"
  grep FATAL "$LOG" | tail -3 || true
fi

echo "=== coverage ==="
q "SELECT 'floor  '||MIN(date(game_creation/1000,'unixepoch')) FROM matches WHERE state='done' AND game_creation>0"
q "SELECT 'pre-Jun08 '||COUNT(*) FROM matches WHERE state='done' AND game_creation>0
     AND game_creation/1000 < CAST(strftime('%s','2026-06-08') AS INTEGER)"
q "SELECT state||' '||COUNT(*) FROM matches GROUP BY state"

echo "=== per-region rate over ${GAP}s ==="
before=$(snap); sleep "$GAP"; after=$(snap)
paste <(echo "$before") <(echo "$after") | awk '{d=$4-$2;
  printf "  %-4s %8d -> %8d  (+%d)%s\n", $1, $2, $4, d, (d==0?"   << WEDGED":"")}'
