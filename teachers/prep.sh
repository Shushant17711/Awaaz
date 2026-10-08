#!/usr/bin/env bash
# Download + prepare IndicTTS female-speaker data for each teacher language, in order.
# Resumable (finished shards are skipped). Runs as user unit sanotts-prep.
T="$(cd "$(dirname "$0")" && pwd)"; PY="$T/../sanoTTS/.venv/bin/python"
for x in "ta Tamil" "kn Kannada" "gu Gujarati" "pa Punjabi" "or Odia" "as Assamese"; do
  set -- $x; mkdir -p "$T/$1"
  grep -q "done," "$T/$1/prep.log" 2>/dev/null && continue
  for try in 1 2 3 4 5; do "$PY" "$T/prepare_indictts.py" "$1" "$2" >> "$T/$1/prep.log" 2>&1 && break; sleep 30; done
done
