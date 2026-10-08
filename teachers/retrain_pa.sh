#!/usr/bin/env bash
# After asr_filter.py pa finishes and the teacher pipeline is idle: discard the
# teacher trained on mispaired data and retrain it on the filtered rows.
T="$(cd "$(dirname "$0")" && pwd)"; S="$T/../sanoTTS"
while systemctl --user is-active --quiet sanotts-asrfilter-pa || systemctl --user is-active --quiet sanotts-teachers; do sleep 120; done
grep -q "kept" <(journalctl --user -u sanotts-asrfilter-pa -o cat --no-pager) || { echo "asr filter did not finish"; exit 1; }
rm -rf "$T/pa/final.onnx" "$T/pa/lightning_logs" "$T/pa/cache" "$T/pa/train.out" "$S/models/teachers/pa_IN-indictts-medium.hold"
echo "retraining pa on $(wc -l < "$T/pa/metadata.train.csv") ASR-verified rows"
"$T/pipeline.sh" start
