#!/usr/bin/env bash
# Overnight queue: distil each voice in turn. Re-running resumes: finished
# stages are skipped and an interrupted stage continues from its resume.pt.
#   ./queue.sh start | status | log | stop
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; REPO="$ROOT/sanoTTS"; PY="$REPO/.venv/bin/python"
UNIT=sanotts-queue
# voice | text corpus | profile | out-dir name
JOBS=(
  "hi_IN-priyamvada-medium|hi_IN.jsonl|shipped|hi_IN-priyamvada-medium"
  "te_IN-padmavathi-medium|te_IN.jsonl|shipped|te_IN-padmavathi-medium"
  "ml_IN-meera-medium|ml_IN.jsonl|shipped|ml_IN-meera-medium"
  "ur_PK-aegis_female-medium|ur_PK.jsonl|shipped|ur_PK-aegis_female-medium"
  "hi_IN-priyamvada-medium|hi_IN.jsonl|long|hi_IN-priyamvada-medium-long"
)

run_queue() {
  cd "$REPO"
  for job in "${JOBS[@]}"; do
    IFS='|' read -r voice text profile name <<<"$job"
    out="$REPO/artifacts/voices/$name"
    if [[ -f "$out/joint/decoder-student.pt" ]]; then echo "== $name already done"; continue; fi
    echo "== $(date '+%F %T') START $name ($profile)"
    "$PY" tools/train_voice_from_piper.py --voice "$voice" \
      --teacher-onnx "$REPO/models/teachers/$voice/$voice.onnx" \
      --text "$ROOT/data/text/$text" --profile "$profile" --device cuda \
      --out-dir "$out" --stop-after s9
    rc=$?
    [[ $rc -eq 143 || $rc -eq 130 ]] && { echo "== stopped by signal; resume with ./queue.sh start"; exit $rc; }
    echo "== $(date '+%F %T') END $name rc=$rc"
  done
  echo "== $(date '+%F %T') QUEUE COMPLETE"
}

case "${1:-}" in
  _run) run_queue ;;
  start)
    systemctl --user is-active --quiet $UNIT && { echo "already running"; exit 0; }
    systemctl --user reset-failed $UNIT 2>/dev/null || true
    # Keep the machine from idle-suspending while it trains (closing the lid still suspends;
    # training just pauses and carries on when it wakes).
    systemd-run --user --unit=$UNIT --collect \
      --property=KillSignal=SIGTERM --property=TimeoutStopSec=180 --setenv=PYTHONUNBUFFERED=1 \
      systemd-inhibit --what=sleep:idle --who=sanotts --why="TTS training" --mode=block \
      "$ROOT/queue.sh" _run
    echo "started — ./queue.sh log to watch" ;;
  status)
    systemctl --user status $UNIT --no-pager -n 3 | head -5
    journalctl --user -u $UNIT --no-pager | grep -E "^.*== |START|DONE|SKIP|ERROR" | tail -15
    find "$REPO/artifacts/voices" -name resume.pt -printf "in progress: %h (saved %TH:%TM)\n" 2>/dev/null ;;
  log) journalctl --user -u $UNIT -f -n 30 ;;
  stop) systemctl --user stop $UNIT && echo "stopped (progress saved)" ;;
  *) sed -n '2,5p' "$0" ;;
esac
