#!/usr/bin/env bash
# Fine-tune a multi-speaker Piper teacher for one language; resumable.
#   train_teacher.sh <lang> <espeak> <base: maya|rohan> <max_epochs> [batch]
# First run warm-starts from the base single-speaker checkpoint (all
# shape-matching weights; new speaker layers fresh). Later runs resume from
# lightning_logs/*/checkpoints/last.ckpt (saved every epoch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="$ROOT/train-venv/bin/python"
lang=$1; espeak=$2; base=$3; epochs=$4; batch=${5:-16}
dir="$ROOT/teachers/$lang"; cd "$dir"
# Studio clips run to ~60 s; a batch holding a few of those overflows 8 GB. Train on 1-12 s clips.
if [[ ! -f metadata.train.csv ]]; then
  "$ROOT/sanoTTS/.venv/bin/python" - "$(cat audio_dir)" <<'PY'
import sys, soundfile as sf
from pathlib import Path
import statistics
audio = Path(sys.argv[1]); rows = []
for line in open("metadata.csv", encoding="utf-8"):
    f, text = line.rstrip("\n").split("|", 1)[0], line.rstrip("\n").split("|")[-1]
    try: d = sf.info(audio / f).duration
    except Exception: continue
    if 1.0 <= d <= 12.0: rows.append((line, len(text) / d))
# Speaking-rate sanity: a transcript far faster/slower than the language's median rate does
# not belong to its audio (misaligned pairs exist in IndicTTS Punjabi: 33-41 chars/s vs a
# 10 chars/s median). They teach wrong pronunciations and blow up attention memory.
med = statistics.median(r for _, r in rows)
kept = [l for l, r in rows if 0.4 * med <= r <= 2.0 * med]
# Piper reads metadata as CSV: a stray " opens a quoted field that swallows the following
# rows (IndicTTS Punjabi/Odia have ~200 such quotes -> 9,600-phoneme "sentences" -> OOM).
kept = [l.replace('"', "") for l in kept]
open("metadata.train.csv", "w", encoding="utf-8").writelines(kept)
print(f"kept {len(kept)} clips of 1-12 s at sane speaking rate (median {med:.1f} chars/s; dropped {len(rows) - len(kept)})")
PY
fi
# 3 columns (file|speaker|text) = multi-speaker; 2 columns (file|text) = single speaker.
if [[ $(head -1 metadata.train.csv | tr -cd '|' | wc -c) -ge 2 ]]; then
  nspk=$(cut -d'|' -f2 metadata.train.csv | sort -u | wc -l)
else
  nspk=1
fi
last=$(ls -t lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
[[ -z "$last" ]] && last=$(ls -t lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
if [[ -n "$last" ]]; then start=(--ckpt_path "$last"); echo "resuming from $last"
else start=(--model.warmstart_ckpt "$ROOT/teachers/base/$base/base.ckpt"); fi
# Retry with half the batch on CUDA OOM (long sentences, e.g. Punjabi, overflow 8 GB at 12).
# Each retry resumes from the newest checkpoint, so nothing already trained is lost.
b=$batch
while :; do
  last=$(ls -t lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
  [[ -z "$last" ]] && last=$(ls -t lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
  if [[ -n "$last" ]]; then start=(--ckpt_path "$last"); else start=(--model.warmstart_ckpt "$ROOT/teachers/base/$base/base.ckpt"); fi
  set +e
  "$PY" -m piper.train fit \
  --data.voice_name "${lang}_IN-openslr-medium" \
  --data.csv_path "$dir/metadata.train.csv" --data.audio_dir "$(cat audio_dir)" \
  --model.sample_rate 22050 --data.espeak_voice "$espeak" \
  --data.cache_dir "$dir/cache" --data.config_path "$dir/config.json" \
  --data.batch_size "$b" --model.num_speakers "$nspk" \
  --trainer.max_epochs "$epochs" --trainer.accelerator gpu --trainer.devices 1 \
  --trainer.precision "${PRECISION:-bf16-mixed}" \
  --trainer.default_root_dir "$dir" "${start[@]}" 2>&1 | tee train.out
  rc=${PIPESTATUS[0]}
  set -e
  [[ $rc -eq 0 ]] && exit 0
  if grep -q "OutOfMemoryError" train.out && [[ $b -gt 4 ]]; then
    b=$(( b / 2 )); echo "== OOM: retrying $lang with batch $b"; continue
  fi
  exit $rc
done
