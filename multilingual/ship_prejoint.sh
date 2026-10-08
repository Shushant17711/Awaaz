#!/usr/bin/env bash
# The joint fine-tune scored worse than the pre-joint checkpoints (A/B, 8 langs x 8 held-out
# sentences: mean CER 0.176 pre-joint vs 0.190 joint; pre-joint better in 6/8). Ship pre-joint:
# wait for phases.sh (which exports the joint model) and the pre-joint 32-sentence eval, then
# overwrite release/multilingual with the pre-joint checkpoints, re-run parity and the model card.
set -uo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$M/.." && pwd)"; PY="$ROOT/sanoTTS/.venv/bin/python"
until [[ -f "$ROOT/eval/results-eval32-mlpre.md" ]] && ! systemctl --user is-active --quiet sanotts-multilingual; do sleep 120; done
R="$ROOT/release/multilingual"
rm -rf "$R" && "$PY" "$M/export_runtime.py" --acoustic "$M/runs/acoustic/acoustic.pt" --vocoder "$M/runs/vocoder/vocoder.pt" --out "$R" || exit 1
"$PY" "$M/test_runtime.py" --acoustic "$M/runs/acoustic/acoustic.pt" --vocoder "$M/runs/vocoder/vocoder.pt" || exit 1
ML_RESULTS="$ROOT/eval/results-eval32-mlpre.json" "$M/finalize.sh" || exit 1
echo "SHIPPED pre-joint model to $R"
