#!/usr/bin/env bash
# Multilingual track, fully unattended. Each phase is skipped when its output exists,
# so re-running resumes (and each trainer resumes mid-phase from resume.pt).
#   vocoder  - once >= 8 languages are fully rendered (it is nearly language-agnostic)
#   acoustic - once all 12 are rendered
#   joint -> eval -> export + parity -> model card -> release the per-language runner
# GPU phases take .gpu.lock; ../.runner-hold keeps the per-language queue off the GPU meanwhile.
#   phases.sh start|status|log|stop
set -uo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$M/.." && pwd)"; PY="$ROOT/sanoTTS/.venv/bin/python"; UNIT=sanotts-multilingual
VOC_STEPS=${VOC_STEPS:-60000}; AC_STEPS=${AC_STEPS:-150000}; JOINT_STEPS=${JOINT_STEPS:-20000}

rendered() {  # languages whose manifest covers the full text file
  local n=0
  for t in "$M"/data/text/*.jsonl; do
    l=$(basename "$t" .jsonl); m="$M/data/wav/$l/manifest.jsonl"
    [[ -f "$m" ]] && [[ $(wc -l < "$m") -ge $(( $(wc -l < "$t") - 50 )) ]] && n=$((n + 1))
  done
  echo $n
}

gpu() {  # gpu NAME cmd...   (exit on signal, fail loudly otherwise)
  local name=$1; shift
  echo "== $(date '+%F %T') $name"
  flock "$ROOT/.gpu.lock" "$@"; local rc=$?
  [[ $rc -eq 143 || $rc -eq 130 ]] && exit $rc
  [[ $rc -ne 0 ]] && { echo "== $name failed rc=$rc"; exit 1; }
}

run() {
  # (no GPU priority hold: time is not a constraint; per-language baselines finish too)
  until [[ $(rendered) -ge 12 ]]; do echo "== $(date '+%F %T') waiting: $(rendered)/12 rendered"; sleep 900; done
  [[ -f "$M/runs/vocoder/vocoder.pt" ]] || gpu "phase 3: vocoder ($VOC_STEPS steps)" "$PY" "$M/train_vocoder.py" --steps "$VOC_STEPS"

  [[ -f "$M/runs/acoustic/acoustic.pt" ]] || gpu "phase 4: acoustic ($AC_STEPS steps)" "$PY" "$M/train_acoustic.py" --steps "$AC_STEPS"
  [[ -f "$M/runs/joint/vocoder.pt" ]] || gpu "phase 5: joint ($JOINT_STEPS steps)" "$PY" "$M/train_joint.py" --steps "$JOINT_STEPS"
  J="$M/runs/joint"
  if [[ ! -f "$ROOT/eval/results-eval32-ml.md" ]]; then
    echo "== $(date '+%F %T') phase 5: eval (32 held-out sentences x 12 languages)"
    "$PY" "$M/synth.py" --eval32 --acoustic "$J/acoustic.pt" --vocoder "$J/vocoder.pt" \
      && (cd "$ROOT" && nice -n 5 "$PY" eval_voices.py large-v3-turbo eval32-ml) || { echo "== eval failed"; exit 1; }
  fi
  if [[ ! -f "$ROOT/release/multilingual/manifest.json" ]]; then
    echo "== $(date '+%F %T') phase 6: export + parity"
    "$PY" "$M/export_runtime.py" --acoustic "$J/acoustic.pt" --vocoder "$J/vocoder.pt" --out "$ROOT/release/multilingual" \
      && "$PY" "$M/test_runtime.py" --acoustic "$J/acoustic.pt" --vocoder "$J/vocoder.pt" || { echo "== export/parity failed"; exit 1; }
  fi
  [[ -x "$M/finalize.sh" ]] && "$M/finalize.sh"
  echo "== $(date '+%F %T') MULTILINGUAL DONE: eval/results-eval32-ml.md, release/multilingual/"
  rm -f "$ROOT/.runner-hold" && "$ROOT/runner.sh" start
}

case "${1:-}" in
  _run) run ;;
  start)
    systemctl --user is-active --quiet $UNIT && { echo "already running"; exit 0; }
    systemctl --user reset-failed $UNIT 2>/dev/null || true
    systemd-run --user --unit=$UNIT --collect --property=KillSignal=SIGTERM --property=TimeoutStopSec=180 \
      --setenv=PYTHONUNBUFFERED=1 --setenv=VOC_STEPS="$VOC_STEPS" --setenv=AC_STEPS="$AC_STEPS" \
      --setenv=JOINT_STEPS="$JOINT_STEPS" --property=WorkingDirectory="$M" \
      systemd-inhibit --what=sleep:idle --who=sanotts-ml --why="multilingual TTS" --mode=block "$M/phases.sh" _run
    echo started ;;
  status) systemctl --user is-active $UNIT; echo "rendered: $(rendered)/12"
    journalctl --user -u $UNIT -o cat --no-pager | grep -E '^== |"step"' | tail -3 ;;
  log) journalctl --user -u $UNIT -f -n 20 ;;
  stop) systemctl --user stop $UNIT ;;
esac
