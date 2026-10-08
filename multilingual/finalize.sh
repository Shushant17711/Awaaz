#!/usr/bin/env bash
# Make release/multilingual/ publish-ready (nothing is uploaded):
#   samples/<lang>.wav  - one held-out sentence per language, rendered by the NumPy runtime itself
#   README.md           - model card: what it is, size, per-language scores vs baselines, usage, licenses
set -euo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$M/.." && pwd)"; PY="$ROOT/sanoTTS/.venv/bin/python"
R="${RELEASE_DIR:-$ROOT/release/multilingual}"
mkdir -p "$R/samples"
"$PY" - "$R" "$M" "$ROOT" <<'PY'
import json, sys, time, wave
from pathlib import Path
import numpy as np
R, M, ROOT = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(R))
from indicml import Voice
v = Voice(R)
rows, rtf = [], []
for lang in v.languages:
    text = json.loads((M / "data/text" / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines()[14])["text"]
    t0 = time.time(); a = v.synthesize(text, lang); dt = time.time() - t0
    rtf.append(dt / (len(a) / v.sample_rate))
    with wave.open(str(R / "samples" / f"{lang}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(v.sample_rate); w.writeframes((a * 32767).astype(np.int16).tobytes())
    rows.append((lang, text))
import os; ml = json.loads(Path(os.environ.get("ML_RESULTS", ROOT / "eval/results-eval32-ml.json")).read_text())
base = json.loads((ROOT / "eval/results-eval32.json").read_text()) if (ROOT / "eval/results-eval32.json").is_file() else {}
best = {}
for name, r in base.items():                       # best per-language baseline (standard or long)
    loc = name.split("-")[0]
    if loc not in best or r["student_cer"] < best[loc]:
        best[loc] = r["student_cer"]
names = {"hi_IN": "Hindi", "mr_IN": "Marathi", "bn_BD": "Bengali", "as_IN": "Assamese", "pa_IN": "Punjabi",
         "gu_IN": "Gujarati", "or_IN": "Odia", "ta_IN": "Tamil", "te_IN": "Telugu", "kn_IN": "Kannada",
         "ml_IN": "Malayalam", "ur_PK": "Urdu"}
table = ["| Language | Teacher CER | This model CER | Per-language sanoTTS baseline CER |", "|---|---|---|---|"]
gaps = []
for lang in v.languages:
    r = ml.get(lang)
    b = best.get(lang)
    if r and b is not None:
        gaps.append(r["student_cer"] - b)
    table.append(f"| {names[lang]} | {r['teacher_cer']:.3f} | {r['student_cer']:.3f} | {b:.3f} |" if r and b is not None else
                 f"| {names[lang]} | {r['teacher_cer']:.3f} | {r['student_cer']:.3f} | — |" if r else
                 f"| {names[lang]} | — | not auto-scorable (see note) | — |")
man = v.manifest
mb = man["weights_size_bytes"] / 1e6
card = f"""# indicml: one tiny TTS for 12 Indian languages

A single **{man['total_parameters'] / 1e6:.2f}M-parameter** text-to-speech model ({mb:.1f} MB fp16) that speaks
{', '.join(names[l] for l in v.languages)}.
It runs on CPU with **only NumPy** (no PyTorch, no eSpeak); measured real-time factor {np.median(rtf):.2f} on a laptop CPU.

The 12 separate sanoTTS-style voices it replaces total ~19M parameters / ~37 MB.

## Use
```
pip install numpy
python -m indicml . hi_IN "नमस्ते, आप कैसे हैं?" out.wav
```
```python
from indicml import Voice
voice = Voice(".")
audio = voice.synthesize("ਪੰਜਾਬੀ ਬੋਲੀ", "pa_IN")      # float32, 22,050 Hz
```
Languages: {', '.join(f'`{l}`' for l in v.languages)}. Each language speaks in the voice of the teacher it was distilled from (all female voices).

## How it was made
1. **Teachers:** 6 Piper voices (hi, te, ml, ur, plus one frozen speaker each from the multi-speaker bn and mr voices) and 6 Piper voices fine-tuned here on IIT Madras IndicTTS studio recordings (ta, kn, gu, pa, or, as).
2. **Synthetic corpus:** each teacher read ~10,000 Wikipedia sentences (Leipzig Corpora).
3. **Student:** a unified Indic grapheme front end (the 9 Brahmic scripts share one token set via their common Unicode layout; Urdu has its own range) → acoustic model with a language embedding, monotonic-alignment-search durations and language FiLM → 80-bin mel → a shared HiFi-GAN-style vocoder (sanoTTS's piperlite architecture, initialised from a Piper generator).
4. **Joint fine-tune** of acoustic + vocoder on the model's own predicted mels.

## Quality
Whisper large-v3-turbo character error rate (CER) on 32 held-out sentences per language (never trained on). Lower is better. The teacher's CER is the ceiling for that language; the baseline is the best separately trained ~1.57M per-language sanoTTS student.

{chr(10).join(table)}

Mean CER difference vs the per-language baselines: **{np.mean(gaps):+.3f}** over {len(gaps)} scored languages.
Whisper is weak on Malayalam (teacher CER is high), so read that row relatively. **Odia and Assamese are proxy-scored:** Whisper has no usable model for either (0.91 CER on real Assamese recordings), so their speech is transcribed as Bengali and compared with the reference transliterated into Bengali script (validated on real recordings: Odia 20/20, Assamese 19/20 clips under 0.6). These rows compare student vs teacher within the language, not across languages.
**Listen to `samples/`** before trusting any number.

## Limitations
- Numbers: digits are read through the grapheme tokens and are not reliably expanded. Write numbers as words.
- Urdu script omits most short vowels; Urdu is expected to be the weakest language.
- One voice per language; no speaker or style control.
- Kangri, Bhojpuri and Konkani are not included (no usable training data yet).

## Licenses: check before publishing
- Runtime code: vocoder ops vendored from sanoTTS (MIT, `indicml/LICENSE.sanotts`); the rest was written for this project.
- **Model weights** are derived from the teachers' outputs, so they inherit the teachers' and their datasets' terms. See `DATASETS.md` in the project. Kannada, Gujarati, Odia and Assamese IndicTTS data have **no license stated** on their Hugging Face cards: confirm with IIT Madras before a public release. Each Piper voice's MODEL_CARD names its own license.
"""
(R / "README.md").write_text(card, encoding="utf-8")
print(f"model card + {len(rows)} samples written; median RTF {np.median(rtf):.2f}")
PY
[[ -f "$M/CARD_NOTES.md" ]] && cat "$M/CARD_NOTES.md" >> "$R/README.md"
cp "$ROOT/DATASETS.md" "$R/DATASETS.md"
