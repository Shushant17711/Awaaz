#!/usr/bin/env bash
# Ship de-smoothed acoustic + 260k-step vocoder as fp16 (release/multilingual) and int8 (release/multilingual-int8).
set -uo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$M/.." && pwd)"; PY="$ROOT/sanoTTS/.venv/bin/python"
A="$M/runs/acoustic-adv/acoustic.pt"; V="$M/runs/vocoder-long/vocoder.pt"
R="$ROOT/release/multilingual"; R8="$ROOT/release/multilingual-int8"
[[ -d "$R" ]] && cp -r "$R" "$ROOT/release/multilingual.prev-$(date +%Y%m%d-%H%M)"
rm -rf "$R" "$R8"
"$PY" "$M/export_runtime.py" --acoustic "$A" --vocoder "$V" --out "$R" || exit 1
"$PY" "$M/export_runtime.py" --acoustic "$A" --vocoder "$V" --out "$R8" --int8 || exit 1
"$PY" "$M/test_runtime.py" --acoustic "$A" --vocoder "$V" || exit 1
"$PY" "$M/synth.py" --eval32 --tag mlfinal --acoustic "$A" --vocoder "$V" || exit 1
(cd "$ROOT" && nice -n 5 "$PY" eval_voices.py large-v3-turbo eval32-mlfinal) || exit 1
ML_RESULTS="$ROOT/eval/results-eval32-mlfinal.json" "$M/finalize.sh" || exit 1
RELEASE_DIR="$R8" ML_RESULTS="$ROOT/eval/results-eval32-mlfinal.json" "$M/finalize.sh" || exit 1
sed -i '1a\
\
> **int8 build:** weights stored as int8 with per-output-channel scales (2.5 MB instead of 4.8 MB). Measured against fp16 with identical timing: acoustic mel-L1 0.02, vocoder waveform corr 0.9993. Scores in the table were measured on the fp16 build.' "$R8/README.md"
echo "SHIPPED fp16 -> $R ; int8 -> $R8"
