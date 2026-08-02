#!/usr/bin/env bash
# Three-floor soloq coverage sensitivity gate
# (docs/2026-08-01-soloq-backfill-handoff.md, "Step 1" + the sweep it asks for).
#
# WHY THIS IS A SWEEP AND NOT A CONSTANT FLIP
# -------------------------------------------
# The 07-31/08-01 backfill moved the db floor from 2026-06-08 to 2026-03-03 and
# took live train decisions from 1,360 to 19,277 (14.2x). But the lift tables
# are patch-BLIND, and the games the backfill added are OLD: the 06-08 window
# spans 3 patches (16.11-16.13), the 03-03 window spans 10 (16.4-16.13). So
# lowering the floor buys more live rows and pays in staler champion-strength
# evidence. Those two effects push in opposite directions and a single re-run
# at 03-03 cannot separate them. Hence three arms.
#
# Read the result as: if the effect is real it should be monotone-ish or peak at
# the dense floor. If 03-03 underperforms 05-01, the answer is a density- or
# recency-weighted table, not more calendar.
#
# THE ONE THING YOU MUST NOT CHANGE
# ---------------------------------
# All three arms read ONE frozen snapshot. The scraper adds ~200k games/day; if
# the arms were built from the live db hours apart, "floor" would be confounded
# with "how far the scrape had got", and the sweep would measure nothing. The
# snapshot is taken once, up front, and every arm points at it.
#
# Usage:
#   bash scripts/sweep_soloq_floor.sh [snapshot.db]
# With no argument it takes a fresh snapshot from the live db via VACUUM INTO,
# which is safe to run while the scraper is writing.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
FLOORS=(2026-06-08 2026-05-01 2026-03-03)   # control, mid, full
PROC=data/processed
STAMP=$(date +%Y%m%dT%H%M%S)
SNAP=${1:-data/raw/soloq/sweep-snapshot-${STAMP}.db}

if [[ ! -f $SNAP ]]; then
  echo "==> freezing a snapshot for all arms: $SNAP"
  # VACUUM INTO takes a consistent copy without blocking the writer.
  sqlite3 "file:data/raw/soloq/soloq.db?mode=ro" "VACUUM INTO '$SNAP';"
fi
echo "==> snapshot: $SNAP ($(du -h "$SNAP" | cut -f1))"
sqlite3 "$SNAP" "select 'games ' || count(*) || ', floor ' ||
  min(date(game_creation/1000,'unixepoch')) || ', ceil ' ||
  max(date(game_creation/1000,'unixepoch'))
  from matches where state='done' and game_creation > 0;"

for FLOOR in "${FLOORS[@]}"; do
  KEY="floor$(echo "$FLOOR" | tr -d -)"
  TABLES="$PROC/soloq_lift_tables_${KEY}.json"
  RESULT="$PROC/rung2_gbm_${KEY}.json"

  echo
  echo "=============================================================="
  echo "ARM $FLOOR"
  echo "=============================================================="

  echo "--> 1/3 lift tables"
  $PY scripts/soloq_lift_tables.py --floor "$FLOOR" --db "$SNAP" --out "$TABLES"

  echo "--> 2/3 decision dataset"
  $PY scripts/draft_dataset.py --soloq-floor "$FLOOR" \
      --soloq-tables "$TABLES" --out-suffix "_${KEY}"

  echo "--> 3/3 rung-2 GBM gate"
  $PY scripts/experiment_v12_rung2_gbm.py \
      --dataset "$PROC/draft_decisions_${KEY}.parquet" \
      --soloq-tables "$TABLES" --out "$RESULT"
done

echo
echo "=============================================================="
echo "SWEEP COMPLETE — arms written to $PROC/rung2_gbm_floor*.json"
echo "=============================================================="
$PY - <<'EOF'
import json, pathlib
rows = []
for p in sorted(pathlib.Path("data/processed").glob("rung2_gbm_floor*.json")):
    d = json.loads(p.read_text())
    t = d["provenance"]["soloq_table"]
    base = d["models"]["baseline"]["ensemble"]
    best = max(("V1", "V2", "V3"),
               key=lambda k: d["models"][k]["ensemble"]["top3"])
    rows.append((t.get("soloq_coverage_start"), t.get("clean_games"),
                 len(t.get("patch_mix") or {}), base["top3"],
                 d["models"][best]["ensemble"]["top3"], best,
                 d["gate"]["pass"]))
print(f"{'floor':<12}{'games':>9}{'patches':>9}{'base t3':>9}"
      f"{'best t3':>9}{'arm':>6}{'gate':>7}")
for r in sorted(rows):
    print(f"{r[0]:<12}{r[1]:>9}{r[2]:>9}{r[3]:>9.4f}{r[4]:>9.4f}"
          f"{r[5]:>6}{'PASS' if r[6] else 'null':>7}")
print("\nMore games is not automatically better — compare across floors, and "
      "remember the 1,080-decision val set cannot resolve a sub-1pt top-3 "
      "effect. State the power limit whatever the result.")
EOF
