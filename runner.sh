#!/usr/bin/env bash
# Job runner: trains every line of jobs.txt in order, re-reading the file after
# each job. Finished jobs (joint/decoder-student.pt exists) are skipped; failed
# ones are logged to failed.txt and skipped until that line is removed.
# Re-running resumes mid-stage (resume.pt). ./runner.sh start|status|log|stop
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; REPO="$ROOT/sanoTTS"; PY="$REPO/.venv/bin/python"
UNIT=sanotts-runner

next_job() {
  grep -v '^\s*#' "$ROOT/jobs.txt" | grep -v '^\s*$' | while IFS= read -r line; do
    IFS='|' read -r voice text profile name extra <<<"$line"
    [[ -f "$REPO/artifacts/voices/$name/joint/decoder-student.pt" ]] && continue
    grep -qxF "$name" "$ROOT/failed.txt" 2>/dev/null && continue
    echo "$line"; break
  done
}

run_all() {
  cd "$REPO"; touch "$ROOT/failed.txt"
  while line="$(next_job)"; [[ -n "$line" ]]; do
    IFS='|' read -r voice text profile name extra <<<"$line"
    echo "== $(date '+%F %T') START $name ($profile)"
    # shellcheck disable=SC2086
    # One GPU job at a time (shared with teachers/pipeline.sh): 8 GB is not enough for both.
    flock "$ROOT/.gpu.lock" "$PY" tools/train_voice_from_piper.py --voice "$voice" \
      --teacher-onnx "$REPO/models/teachers/$voice/$voice.onnx" \
      --text "$ROOT/data/text/$text" --profile "$profile" --device cuda \
      --out-dir "$REPO/artifacts/voices/$name" --stop-after s9 $extra
    rc=$?
    [[ $rc -eq 143 || $rc -eq 130 ]] && { echo "== stopped by signal"; exit $rc; }
    [[ $rc -ne 0 ]] && echo "$name" >> "$ROOT/failed.txt"
    echo "== $(date '+%F %T') END $name rc=$rc"
    "$ROOT/render_samples.sh" >/dev/null 2>&1 || true
  done
  echo "== $(date '+%F %T') ALL JOBS DONE"
}

case "${1:-}" in
  _run) run_all ;;
  start)
    # While the multilingual model trains it has GPU priority (multilingual/phases.sh
    # creates this file and removes it, then restarts the runner, when it finishes).
    [[ -f "$ROOT/.runner-hold" ]] && { echo "runner on hold (.runner-hold): multilingual has GPU priority"; exit 0; }
    systemctl --user is-active --quiet $UNIT && { echo "already running"; exit 0; }
    systemctl --user reset-failed $UNIT 2>/dev/null || true
    systemd-run --user --unit=$UNIT --collect \
      --property=KillSignal=SIGTERM --property=TimeoutStopSec=180 --setenv=PYTHONUNBUFFERED=1 \
      systemd-inhibit --what=sleep:idle --who=sanotts --why="TTS training" --mode=block \
      "$ROOT/runner.sh" _run
    echo "started — ./runner.sh log to watch" ;;
  status)
    systemctl --user is-active $UNIT
    journalctl --user -u $UNIT --no-pager | grep -E "== |ERROR" | tail -12
    echo "failed: $(tr '\n' ' ' < "$ROOT/failed.txt" 2>/dev/null)" ;;
  log) journalctl --user -u $UNIT -f -n 30 ;;
  stop) systemctl --user stop $UNIT && echo "stopped (progress saved)" ;;
  *) sed -n '2,6p' "$0" ;;
esac
