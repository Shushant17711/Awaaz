#!/usr/bin/env bash
# Run / resume / watch a sanoTTS distillation run as a background user service.
#
#   ./train.sh start <voice> [extra driver args]   e.g. ./train.sh start hi_IN-pratham-medium
#   ./train.sh status <voice>
#   ./train.sh log <voice>          follow the current stage log
#   ./train.sh stop <voice>         save a checkpoint and stop
#
# `start` after a shutdown/reboot/stop resumes: finished stages are skipped and
# the interrupted stage continues from its resume.pt (see sanoTTS/tools/resume_state.py).
# The service gets SIGTERM on stop or poweroff; trainers save before exiting,
# and systemd waits up to 3 minutes for that.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$ROOT/sanoTTS"
PY="$REPO/.venv/bin/python"

cmd="${1:-}"; voice="${2:-}"
[[ -n "$cmd" && -n "$voice" ]] || { sed -n '2,12p' "$0"; exit 2; }
shift 2
unit="sanotts-${voice//[^A-Za-z0-9]/-}"
run_dir="$REPO/artifacts/voices/$voice"

case "$cmd" in
  start)
    if systemctl --user is-active --quiet "$unit"; then
      echo "$unit is already running"; exit 0
    fi
    systemctl --user reset-failed "$unit" 2>/dev/null || true
    lang="${voice%%-*}"
    text_args=()
    corpus="$REPO/data/textsets/multilang-distill-v1/$lang.train.jsonl"
    [[ -f "$corpus" ]] && text_args=(--text "$corpus")
    # Use the local teacher when present, so a resume works offline.
    teacher_args=()
    onnx="$REPO/models/teachers/$voice/$voice.onnx"
    [[ -f "$onnx" ]] && teacher_args=(--teacher-onnx "$onnx")
    systemd-run --user --unit="$unit" --collect \
      --property=KillSignal=SIGTERM --property=TimeoutStopSec=180 \
      --property=WorkingDirectory="$REPO" \
      --setenv=PYTHONUNBUFFERED=1 \
      "$PY" tools/train_voice_from_piper.py --voice "$voice" "${teacher_args[@]}" \
        --teacher-dir "$REPO/models/teachers" --device cuda \
        "${text_args[@]}" "$@"
    echo "started $unit — ./train.sh log $voice to watch"
    ;;
  status)
    systemctl --user status "$unit" --no-pager -n 5 || true
    echo; echo "finished stages / checkpoints:"
    ls -1 "$run_dir"/logs 2>/dev/null | sed 's/\.log$//' || echo "  (none yet)"
    find "$run_dir" -name resume.pt -printf "  in progress: %h (saved %TY-%Tm-%Td %TH:%TM)\n" 2>/dev/null || true
    ;;
  log)
    journalctl --user -u "$unit" -n 20 --no-pager || true
    latest="$(ls -t "$run_dir"/logs/*.log 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] && { echo "== $latest"; tail -n 20 -f "$latest"; }
    ;;
  stop)
    systemctl --user stop "$unit" && echo "stopped $unit (checkpoint saved)"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
