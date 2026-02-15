#!/usr/bin/env bash
# Daily collection daemon — runs daily_collect.py once per day at ~6 AM ET (11:00 UTC).
# Start in background: nohup bash scripts/daily_cron.sh &
#
# How it works:
#   - Sleeps until 11:00 UTC, runs the collector, then sleeps until the next 11:00 UTC.
#   - If started after 11:00 UTC, runs immediately for today, then resumes the schedule.
#   - Logs to logs/cron.log. PID written to logs/cron.pid for easy stopping.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
COLLECTOR="$SCRIPT_DIR/scripts/daily_collect.py"
LOGFILE="$SCRIPT_DIR/logs/cron.log"
PIDFILE="$SCRIPT_DIR/logs/cron.pid"
TARGET_HOUR=11  # 11:00 UTC = 6:00 AM ET

mkdir -p "$SCRIPT_DIR/logs"
echo $$ > "$PIDFILE"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') [cron] Started (PID $$)" >> "$LOGFILE"

seconds_until_target() {
    local now_s=$(date -u '+%s')
    local today_target=$(date -u -d "today ${TARGET_HOUR}:00:00" '+%s' 2>/dev/null || \
                         date -u -j -f '%Y-%m-%d %H:%M:%S' "$(date -u '+%Y-%m-%d') ${TARGET_HOUR}:00:00" '+%s')
    local diff=$((today_target - now_s))
    if [ "$diff" -le 0 ]; then
        # Already past target time today, aim for tomorrow
        diff=$((diff + 86400))
    fi
    echo "$diff"
}

while true; do
    # Run the collector
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') [cron] Running daily_collect.py" >> "$LOGFILE"
    "$PYTHON" "$COLLECTOR" >> "$LOGFILE" 2>&1 || \
        echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') [cron] daily_collect.py FAILED (exit $?)" >> "$LOGFILE"

    # Sleep until next target time
    WAIT=$(seconds_until_target)
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') [cron] Sleeping ${WAIT}s until next run" >> "$LOGFILE"
    sleep "$WAIT"
done
