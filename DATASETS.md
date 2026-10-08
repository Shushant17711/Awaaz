# Data provenance and licenses

Check every license again before redistributing any model. Voices inherit obligations from their teacher and from the data the teacher was trained on.

## Teachers (Piper voices, used as-is)

From `huggingface.co/rhasspy/piper-voices`. Each voice's `MODEL_CARD` names its dataset and license. Read it before release.

| Voice | Used for | Notes |
|---|---|---|
| hi_IN-priyamvada-medium | Hindi | |
| te_IN-padmavathi-medium | Telugu | |
| ml_IN-meera-medium | Malayalam | |
| ur_PK-aegis_female-medium | Urdu | Weight names were anonymised in the export. Fixed locally with `canonicalize_piper_onnx.py` (output bit-identical). |
| bn_BD-google-medium, speaker 12 ("4811") | Bengali | 16-speaker voice. One speaker frozen with `make_single_speaker_teacher.py`. |
| mr_IN-google-medium, speaker 0 ("mrt_01523") | Marathi | 9-speaker voice. One speaker frozen. |

## Teachers trained here

Fine-tuned from Piper training checkpoints (`huggingface.co/datasets/rhasspy/piper-checkpoints`):
- te_IN maya: base for Tamil and Kannada
- hi_IN rohan: base for Gujarati and Punjabi
- bn_BD google: base for Odia and Assamese

Audio: **IIT Madras IndicTTS**, via the SPRINGLab mirror (`huggingface.co/datasets/SPRINGLab/IndicTTS_<Language>`). Only the female speaker is used, resampled to 22.05 kHz.

| Language | Dataset | License on HF card |
|---|---|---|
| Tamil | SPRINGLab/IndicTTS_Tamil | CC-BY-4.0 |
| Punjabi | SPRINGLab/IndicTTS_Punjabi | CC-BY-4.0 |
| Kannada | SPRINGLab/IndicTTS_Kannada | **not stated**; confirm with IITM before release |
| Gujarati | SPRINGLab/IndicTTS_Gujarati | **not stated** |
| Odia | SPRINGLab/IndicTTS_Odia | **not stated** |
| Assamese | SPRINGLab/IndicTTS_Assamese | **not stated** |

The original IndicTTS database (iitm.ac.in/donlab/indictts) historically required a license agreement. The SPRINGLab mirror is ungated. Treat the unstated ones as research-only until confirmed.

Assamese text: `য + ় (U+09AF U+09BC)` is rewritten to precomposed `য় (U+09DF)`, because eSpeak's Assamese voice otherwise spells the letters out. The runtime needs the same rewrite at inference.

Not used (downloaded, then replaced by the studio data above): OpenSLR 65/78/79, the Google crowdsourced ta/gu/kn female sets (CC-BY-SA-4.0).

## Distillation text

Leipzig Corpora Collection, `<lang>_wikipedia_2021_10K` (downloads.wortschatz-leipzig.de), CC-BY. Filtered to 25–170 characters, native script ≥ 90%, no Latin letters. 3,200 sentences per language, shuffled with seed 0. Rows 1–140 are held out (never trained on); evaluation uses rows 13–44.

## Evaluation

Whisper large-v3-turbo (faster-whisper, int8, CPU), MIT license. Transcribes teacher and student audio; character error rate is computed against the source text.

## Not yet available (needs the user)

- **Konkani**: AI4Bharat IndicVoices-R / Rasa (HF, gated: requires login and accepting terms).
- **Bhojpuri**: RESPIN via Bhashini/ULCA (registration).
- **Kangri**: no speech corpus exists. Text: AI4Bharat Kangri corpus. Needs recordings.

## Data-quality findings (2026-10-04)
- **IndicTTS Punjabi (SPRINGLab mirror): about half the rows pair a transcript with a different sentence's audio.** In a 40-clip Whisper sample, 16 matched and 20 clearly did not. It is not a fixed offset: no row within ±40 matches, so the rows cannot be realigned. A teacher trained on it babbled. Fix: `teachers/asr_filter.py pa` keeps only rows whose Whisper CER is ≤ 0.6 (scores in `teachers/pa/asr_scores.tsv`), and the teacher is retrained on those.
- Punjabi and Odia transcripts contain stray `"` characters that broke Piper's CSV reader: one "sentence" swallowed the following rows, giving ~9,600 phonemes vs a median of 75, and out-of-memory crashes. These are stripped before training.
- Punjabi also had transcripts at 33–41 characters/s against a 10/s median (more mispairs). A speaking-rate filter (0.4–2× the language median) drops them, for every trained teacher.
- Odia and Assamese alignment, checked via Bengali Whisper on transliterated text: clean (20/20 and 19/20 clips match).
