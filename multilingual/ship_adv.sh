#!/usr/bin/env bash
# Ship the de-smoothed acoustic model (runs/acoustic-adv) + phase-3 vocoder to release/multilingual.
set -uo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$M/.." && pwd)"; PY="$ROOT/sanoTTS/.venv/bin/python"
A="$M/runs/acoustic-adv/acoustic.pt"; V="$M/runs/vocoder/vocoder.pt"; R="$ROOT/release/multilingual"
cp -r "$R" "$ROOT/release/multilingual.prev-$(date +%Y%m%d-%H%M)"
rm -rf "$R" && "$PY" "$M/export_runtime.py" --acoustic "$A" --vocoder "$V" --out "$R" || exit 1
"$PY" "$M/test_runtime.py" --acoustic "$A" --vocoder "$V" || exit 1
"$PY" "$M/synth.py" --eval32 --tag mladv --acoustic "$A" --vocoder "$V" || exit 1
(cd "$ROOT" && nice -n 5 "$PY" eval_voices.py large-v3-turbo eval32-mladv) || exit 1
ML_RESULTS="$ROOT/eval/results-eval32-mladv.json" "$M/finalize.sh" || exit 1
echo "SHIPPED de-smoothed model to $R"
