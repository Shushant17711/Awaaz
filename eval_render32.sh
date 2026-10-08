#!/usr/bin/env bash
# Render 32 held-out sentences (corpus rows 13-44; rows 1-140 are never trained on)
# for every finished voice into eval32/<voice>/{teacher,student}/ for eval_voices.py.
ROOT="$(cd "$(dirname "$0")" && pwd)"; REPO="$ROOT/sanoTTS"; PY="$REPO/.venv/bin/python"
cd "$REPO"
for V in artifacts/voices/*/; do
  name=$(basename "$V"); [[ -f "$V/joint/decoder-student.pt" ]] || continue
  [[ -f "$ROOT/eval32/$name/student/manifest.jsonl" ]] && continue
  teacher=${name%-long}; loc=${name%%-*}; T="models/teachers/$teacher/$teacher"
  sed -n 13,44p "$ROOT/data/text/$loc.jsonl" > "/tmp/eval32-$loc.jsonl"
  "$PY" tools/render_eval_dirs.py --acoustic "$V/joint/latent-student.pt" --decoder-student "$V/joint/decoder-student.pt" \
    --duration "$V/duration/duration-student.pt" --piper-model "$T.onnx" --piper-config "$T.onnx.json" \
    --textset "/tmp/eval32-$loc.jsonl" --out-dir "$ROOT/eval32/$name" --length-scale 1.0 >/dev/null 2>&1 \
    && echo "rendered $name" || echo "render failed $name"
done
