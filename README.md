# Aaवाz

Small text-to-speech for Indian languages. One **2.4M-parameter** model (4.8 MB fp16) speaks 12 languages and runs on a CPU with **only NumPy**: no PyTorch, no eSpeak, no GPU. On a laptop CPU it runs at about 0.09× real time.

| Code | Language | Code | Language |
|---|---|---|---|
| `hi_IN` | Hindi | `ta_IN` | Tamil |
| `mr_IN` | Marathi | `te_IN` | Telugu |
| `bn_BD` | Bengali | `kn_IN` | Kannada |
| `as_IN` | Assamese | `ml_IN` | Malayalam |
| `pa_IN` | Punjabi | `gu_IN` | Gujarati |
| `or_IN` | Odia | `ur_PK` | Urdu |

Samples for every language are in [`release/multilingual/samples/`](release/multilingual/samples/).

## Quick start

```bash
cd release/multilingual
pip install numpy
python -m indicml . hi_IN "नमस्ते, आप कैसे हैं?" out.wav
```

```python
from indicml import Voice

voice = Voice("release/multilingual")
audio = voice.synthesize("ਪੰਜਾਬੀ ਬੋਲੀ", "pa_IN")   # float32 mono, 22,050 Hz
```

Write numbers as words. The model reads digits as plain characters and does not expand them.

## How it works

The model is **distilled**: full-size Piper/VITS "teacher" voices read about 10,000 Wikipedia sentences per language, and a small student learns to copy them.

1. **Teachers.** Six existing Piper voices (hi, te, ml, ur, plus one speaker each from the multi-speaker bn and mr voices). Six more (ta, kn, gu, pa, or, as) were fine-tuned in this repo on IIT Madras IndicTTS studio recordings.
2. **Front end.** One grapheme tokenizer covers all 9 Brahmic scripts through their shared Unicode layout. Urdu gets its own range. There is no phonemizer.
3. **Acoustic model.** Uses a language embedding, monotonic-alignment-search durations and per-language FiLM, and outputs an 80-bin mel spectrogram.
4. **Vocoder.** A shared HiFi-GAN-style generator, initialised from a Piper decoder.
5. **Fine-tuning.** An adversarial mel-patch stage reduces over-smoothing, followed by a long vocoder run on the model's own predicted mels.

## Quality

Character error rate (CER) from Whisper large-v3-turbo on 32 held-out sentences per language. Lower is better. The teacher's CER is the best the student can reach.

| Language | Teacher | Aaवाz |
|---|---|---|
| Hindi | 0.119 | 0.105 |
| Marathi | 0.207 | 0.217 |
| Bengali | 0.296 | 0.305 |
| Assamese* | 0.471 | 0.501 |
| Punjabi | 0.399 | 0.416 |
| Gujarati | 0.278 | 0.279 |
| Odia* | 0.361 | 0.335 |
| Tamil | 0.106 | 0.151 |
| Telugu | 0.111 | 0.132 |
| Kannada | 0.135 | 0.190 |
| Malayalam | 0.548 | 0.547 |
| Urdu | 0.067 | 0.092 |

\*Whisper cannot transcribe Assamese or Odia. Those rows were scored by transliterating the text to Bengali script. Whisper is also unreliable on Malayalam, so listen to that voice rather than trusting its number. The full write-up and known weak spots are in [`release/multilingual/README.md`](release/multilingual/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `release/multilingual/` | The shipped model: `weights.fp16.bin`, `manifest.json`, the NumPy runtime `indicml/` and samples |
| `multilingual/` | Training code for the multilingual student (front end, acoustic model, vocoder, export, evaluation) |
| `multilingual/v2/` | A second training run that was **not** promoted, with its report |
| `teachers/` | Data preparation and Piper teacher fine-tuning (IndicTTS, ASR mispair filter) |
| `eval/`, `eval_voices.py`, `combine/` | Whisper CER evaluation and checkpoint-combination experiments |
| `diag/`, `review/` | Spectral diagnostics and listening-review sheets |
| `kangri/` | Recording script for a future Kangri voice (no speech data yet) |
| `train.sh`, `runner.sh`, `queue*.sh`, `package_all.sh` | Earlier per-language distillation pipeline |
| `DATASETS.md` | Where all the data came from and its license status |

Training needs the third-party checkouts listed in `.gitignore` (`sanoTTS/`, `piper1-gpl/`) and the datasets in `DATASETS.md`. Those are not included here.

## Not yet supported

- **Konkani, Bhojpuri, Kangri:** there is no usable training data yet.
- Each language has one voice, with no control over speaker or style.
- Prosody is flatter than in the 61M-parameter teacher voices.

## Licenses

- **Runtime:** the vocoder ops are vendored from sanoTTS under MIT (see `release/multilingual/indicml/LICENSE.sanotts`). The rest of the code was written for this project.
- **Model weights** are derived from the teachers' outputs, so they inherit the terms of the teachers and their datasets. The IndicTTS Kannada, Gujarati, Odia and Assamese sets have **no license stated**. Treat the weights as research-only until those terms are confirmed. See [`DATASETS.md`](DATASETS.md).

## Acknowledgements

- [Piper](https://github.com/OHF-Voice/piper1-gpl) teacher voices
- IIT Madras IndicTTS recordings
- Leipzig Corpora Collection text
- The sanoTTS distillation recipe
