#!/usr/bin/env bash
# Standalone progress logger -- detached from the chat session on purpose, so it never
# sends notifications. Tail data/processed/progress.log yourself to check on long jobs.
LOG="/root/Issue-Assignee-Recommender/data/processed/progress.log"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  jobs=$(pgrep -af "scripts/0[0-9]_" 2>/dev/null | grep -v progress_logger)
  if [ -z "$jobs" ]; then
    echo "[$ts] no tracked pipeline jobs running" >> "$LOG"
  else
    while IFS= read -r line; do
      pid=$(echo "$line" | awk '{print $1}')
      etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
      script=$(echo "$line" | grep -oE 'scripts/[0-9]+_[a-zA-Z0-9_]+\.py')
      logfile=$(ls -t /root/Issue-Assignee-Recommender/data/processed/*.log 2>/dev/null | grep -v /progress.log | head -1)
      lastline=""
      [ -n "$logfile" ] && lastline=$(tail -1 "$logfile" 2>/dev/null)
      echo "[$ts] pid=$pid script=$script elapsed=$etime last=\"$lastline\"" >> "$LOG"
    done <<< "$jobs"
  fi
  sleep 60
done
