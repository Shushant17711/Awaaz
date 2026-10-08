#!/usr/bin/env bash
# Render 8 held-out sentences (teacher vs student) for every finished voice
# into samples/<voice>/{teacher,student}/0000N.wav
ROOT="$(cd "$(dirname "$0")" && pwd)"; REPO="$ROOT/sanoTTS"; PY="$REPO/.venv/bin/python"
cd "$REPO"
"$ROOT/package_all.sh" || true
cd "$REPO"
for V in artifacts/voices/*/; do
  name=$(basename "$V"); [[ -f "$V/joint/decoder-student.pt" ]] || continue
  [[ -f "$ROOT/samples/$name/student/manifest.jsonl" ]] && continue
  teacher=${name%-long}; loc=${name%%-*}
  T="models/teachers/$teacher/$teacher"
  sed -n 15,22p "$ROOT/data/text/$loc.jsonl" > "/tmp/listen-$loc.jsonl"
  "$PY" tools/render_eval_dirs.py --acoustic "$V/joint/latent-student.pt" \
    --decoder-student "$V/joint/decoder-student.pt" --duration "$V/duration/duration-student.pt" \
    --piper-model "$T.onnx" --piper-config "$T.onnx.json" --textset "/tmp/listen-$loc.jsonl" \
    --out-dir "$ROOT/samples/$name" --length-scale 1.0 || echo "render failed for $name"
done

# Disk hygiene: finished voices don't need their training packs (re-renderable,
# deterministic, ~4 GB each). Keep checkpoints, logs, eval128.
for V in artifacts/voices/*/; do
  [[ -f "$V/joint/decoder-student.pt" ]] || continue
  rm -rf "$V/train-acoustic" "$V/train512-decoder" "$V/signatures"
done

# Score any newly finished voices on the 32-sentence held-out set and refresh release/best/.
# Detached, so the runner's next GPU job isn't held up by this CPU work.
if ! pgrep -f "eval_voices.py|eval_render32.sh" >/dev/null; then
  ( "$ROOT/eval_render32.sh" && nice -n 5 "$PY" "$ROOT/eval_voices.py" large-v3-turbo eval32 \
      && "$PY" "$ROOT/select_best.py" ) >> "$ROOT/eval/run32.log" 2>&1 < /dev/null &
  disown
fi
