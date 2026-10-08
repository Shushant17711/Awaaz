# indicml: one tiny TTS for 12 Indian languages

A single **2.40M-parameter** text-to-speech model (4.8 MB fp16) that speaks
Hindi, Marathi, Bengali, Assamese, Punjabi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Urdu.
It runs on CPU with **only NumPy** (no PyTorch, no eSpeak); measured real-time factor 0.09 on a laptop CPU.

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
Languages: `hi_IN`, `mr_IN`, `bn_BD`, `as_IN`, `pa_IN`, `gu_IN`, `or_IN`, `ta_IN`, `te_IN`, `kn_IN`, `ml_IN`, `ur_PK`. Each language speaks in the voice of the teacher it was distilled from (all female voices).

## How it was made
1. **Teachers:** 6 Piper voices (hi, te, ml, ur, plus one frozen speaker each from the multi-speaker bn and mr voices) and 6 Piper voices fine-tuned here on IIT Madras IndicTTS studio recordings (ta, kn, gu, pa, or, as).
2. **Synthetic corpus:** each teacher read ~10,000 Wikipedia sentences (Leipzig Corpora).
3. **Student:** a unified Indic grapheme front end (the 9 Brahmic scripts share one token set via their common Unicode layout; Urdu has its own range) → acoustic model with a language embedding, monotonic-alignment-search durations and language FiLM → 80-bin mel → a shared HiFi-GAN-style vocoder (sanoTTS's piperlite architecture, initialised from a Piper generator).
4. **Joint fine-tune** of acoustic + vocoder on the model's own predicted mels.

## Quality
Whisper large-v3-turbo character error rate (CER) on 32 held-out sentences per language (never trained on). Lower is better. The teacher's CER is the ceiling for that language; the baseline is the best separately trained ~1.57M per-language sanoTTS student.

| Language | Teacher CER | This model CER | Per-language sanoTTS baseline CER |
|---|---|---|---|
| Hindi | 0.119 | 0.105 | 0.114 |
| Marathi | 0.207 | 0.217 | 0.260 |
| Bengali | 0.296 | 0.305 | 0.318 |
| Assamese | 0.471 | 0.501 | 0.486 |
| Punjabi | 0.399 | 0.416 | 0.351 |
| Gujarati | 0.278 | 0.279 | 0.297 |
| Odia | 0.361 | 0.335 | 0.368 |
| Tamil | 0.106 | 0.151 | 0.118 |
| Telugu | 0.111 | 0.132 | 0.111 |
| Kannada | 0.135 | 0.190 | 0.148 |
| Malayalam | 0.548 | 0.547 | 0.489 |
| Urdu | 0.067 | 0.092 | 0.072 |

Mean CER difference vs the per-language baselines: **+0.012** over 12 scored languages.
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

## How the shipped checkpoint was chosen
- **De-smoothing (shipped):** the acoustic model was first trained with L1/L2 losses only, which gave over-smoothed spectrograms (~20% less frame-to-frame detail than real speech) and a robotic, buzzy sound. A further 20k steps with a training-only mel-patch discriminator closed the detail gap (0.53 vs 0.54 real), cut the extra 4–11 kHz buzz by more than half, and kept intelligibility (8-language A/B mean CER 0.175, vs 0.176 before).
- **Longer vocoder training (shipped):** the vocoder was continued from 60k to 260k steps with cosine LR decay (2e-4 → 1e-5). Resynthesis error on real teacher audio fell 15% (mel-L1 0.278 → 0.236); excess energy above 4 kHz went from +0.99 dB to −0.05 dB (now matches the teacher). 8-language A/B mean CER 0.175 → 0.166.
- **Rejected:** joint acoustic+vocoder fine-tune (A/B mean CER 0.190), vocoder-only joint fine-tune (0.193), and fine-tuning the long-trained vocoder on predicted mels (0.170 vs 0.166).
- **Output loudness:** the runtime normalises speech to −16 dBFS with a soft-knee limiter at −1 dBFS (`synthesize(..., normalize=False)` returns the raw output).

## Known weak spots
- **Punjabi** (+6.6 points vs its per-language baseline, over the +5 target) and **Kannada** (+4.2). Punjabi's teacher was trained on only 1,232 usable clips after ~60% of the IndicTTS Punjabi rows turned out mispaired, and Whisper is weak on Punjabi; listen before relying on it.
- **Malayalam** scores are unreliable: Whisper's CER on the *same* teacher audio has ranged from 0.50 to 0.62 across runs. Judge Malayalam by listening.
- Intelligibility of this release vs the previous one is a statistical tie (8-language A/B: 0.166 vs 0.175 in favour; full 12×32 eval: 0.273 vs 0.268 against). It was shipped for its clearly better sound.
- It is still a 2.4M-parameter model covering 12 languages: expect it to sound less natural than the 61 MB single-language Piper teachers, especially in prosody.
