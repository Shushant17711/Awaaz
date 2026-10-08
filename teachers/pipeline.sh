#!/usr/bin/env bash
# Teacher track for languages with no Piper voice:
#   IndicTTS studio data (female speaker) -> fine-tune Piper teacher -> export ONNX
#   -> queue sanoTTS distillation (shipped + long) in ../jobs.txt.
# Each step is skipped when its output exists, so re-running resumes.
#   pipeline.sh start|status|log|stop        (data prep runs separately: prep.sh)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; T="$ROOT/teachers"; S="$ROOT/sanoTTS"
UNIT=sanotts-teachers
STEPS=${STEPS:-15000}   # fine-tuning optimizer steps per teacher
BATCH=12
# lang | espeak voice | base checkpoint | locale
LANGS=("ta|ta|maya|ta_IN" "kn|kn|maya|kn_IN" "gu|gu|rohan|gu_IN" "pa|pa|rohan|pa_IN" "or|or|bengali|or_IN" "as|as|bengali|as_IN")

one() {
  IFS='|' read -r lang espeak base loc <<<"$1"
  name="${loc}-indictts-medium"; d="$S/models/teachers/$name"
  until grep -q "done," "$T/$lang/prep.log" 2>/dev/null; do
    echo "== $(date '+%F %T') $lang: waiting for data prep"; sleep 300
  done
  if [[ ! -f "$T/$lang/final.onnx" ]]; then
    n=$(( $(wc -l < "$T/$lang/metadata.csv") * 70 / 100 ))  # ~70% survive the 1-12 s filter
    epochs=$(( (STEPS * BATCH + n - 1) / n ))
    echo "== $(date '+%F %T') $lang: teacher ($n utts, $epochs epochs ~ $STEPS steps, base $base)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock "$ROOT/.gpu.lock" \
      "$T/train_teacher.sh" "$lang" "$espeak" "$base" "$epochs" "$BATCH"
    rc=$?; [[ $rc -eq 143 || $rc -eq 130 ]] && exit $rc
    [[ $rc -ne 0 ]] && { echo "== $lang teacher training failed rc=$rc"; return 1; }
    last=$(ls -t "$T/$lang"/lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [[ -z "$last" ]] && last=$(ls -t "$T/$lang"/lightning_logs/*/checkpoints/*.ckpt | head -1)
    "$ROOT/train-venv/bin/python" -m piper.train.export_onnx --checkpoint "$last" --output-file "$T/$lang/final.onnx.tmp" \
      && mv "$T/$lang/final.onnx.tmp" "$T/$lang/final.onnx" || { echo "== $lang export failed"; return 1; }
    # Disk: Piper's preprocessing cache is ~15 GB per language and only needed while
    # training; keep last.ckpt (for any further fine-tuning) and drop the other top-k checkpoints.
    rm -rf "$T/$lang/cache"
    find "$T/$lang/lightning_logs" -name "*.ckpt" ! -name "last.ckpt" -delete
  fi
  if [[ ! -f "$d/$name.onnx" ]]; then
    mkdir -p "$d"
    # Restore dec.* weight names if the export anonymised them (see canonicalize_piper_onnx.py).
    "$S/.venv/bin/python" "$S/tools/canonicalize_piper_onnx.py" "$T/$lang/final.onnx" "$d/$name.onnx" \
      || { echo "== $lang canonicalize failed"; return 1; }
    cp "$T/$lang/config.json" "$d/$name.onnx.json"
  fi
  for prof in shipped long; do
    out="$name"; [[ $prof == long ]] && out="$name-long"
    grep -q "|$out|" "$ROOT/jobs.txt" || echo "$name|$loc.jsonl|$prof|$out|" >> "$ROOT/jobs.txt"
  done
  systemctl --user is-active --quiet sanotts-runner || "$ROOT/runner.sh" start
  echo "== $(date '+%F %T') $lang: queued for distillation"
}

case "${1:-}" in
  _run) for x in "${LANGS[@]}"; do one "$x"; done; echo "== $(date '+%F %T') TEACHERS DONE" ;;
  start)
    systemctl --user is-active --quiet $UNIT && { echo "already running"; exit 0; }
    systemctl --user reset-failed $UNIT 2>/dev/null || true
    systemd-run --user --unit=$UNIT --collect --property=KillSignal=SIGTERM --property=TimeoutStopSec=120 \
      --setenv=PYTHONUNBUFFERED=1 \
      systemd-inhibit --what=sleep:idle --who=sanotts-teachers --why="TTS teacher training" --mode=block \
      "$T/pipeline.sh" _run
    echo started ;;
  status)
    systemctl --user is-active $UNIT; journalctl --user -u $UNIT --no-pager | grep "== " | tail -6
    for x in "${LANGS[@]}"; do l=${x%%|*}
      e=$(ls -t "$T/$l"/lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1)
      echo "$l: prep=$(tail -1 "$T/$l/prep.log" 2>/dev/null | cut -c1-60) | ckpt=${e:+$(basename "$(dirname "$(dirname "$e")")")} final=$([[ -f $T/$l/final.onnx ]] && echo yes || echo no)"
    done ;;
  log) journalctl --user -u $UNIT -f -n 20 ;;
  stop) systemctl --user stop $UNIT ;;
esac
