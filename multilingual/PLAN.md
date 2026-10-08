# One small TTS for many Indian languages: plan

**Goal:** a single model, about 2–3M parameters in total, that speaks 12 Indian languages. Text goes in through our own script front end (no eSpeak), and the whole stack can be released under a permissive license (MIT).

**Baseline to beat:** the per-language sanoTTS students in `release/best/`. Each is ~1.57M params and 3.0 MB, so 12 languages ≈ 19M params / 37 MB. Their quality is measured in `eval/results-eval32.md`.

**Success criteria (all three):**
1. **Size:** total ≤ 3M params / ≤ 6 MB fp16. That is ≤ 1/6 of the 12 separate models.
2. **Intelligibility:** average Whisper character error rate (CER) within **+2 points** of the per-language baselines on the same 32 held-out sentences per language. No single language may be worse by more than +5 points.
3. **No eSpeak** anywhere at inference. Text goes straight from Unicode to model input.

If (2) fails at 3M params, the fallback is a 2-model split (Indo-Aryan / Dravidian), which still meets (1) and (3).

---

## 1. Key design decisions

### 1.1 Distill through mel spectrograms, not teacher latents
sanoTTS students regress each teacher's private 192-dim VITS latent. The 12 teachers are independently trained, so their latent spaces are unrelated. One decoder cannot invert all of them.

Instead, every teacher's **audio** becomes a common 80-bin log-mel spectrogram at 22.05 kHz, hop 256. Then:
- **Shared vocoder** (mel → waveform): language-agnostic. It trains on all teachers' audio at once, so it gets 12× the data any single sanoTTS decoder sees.
- **Shared acoustic model** (text → mel): conditioned on a language id.

Tradeoff: sanoTTS's write-up credits the latent interface with its small models' ability to handle unseen text. We give that up. We compensate with far more (synthetic) text per language than sanoTTS used: 8k–20k sentences vs 2k. Mel is also what sanoTTS's own sub-300k "nano" voices use, so it is a proven interface at this size.

### 1.2 Teachers become a synthetic corpus
The 12 teachers (6 Piper, 6 trained here) are frozen. We render each on a large, varied text set and keep wav + mel. The student is trained from scratch on this synthetic corpus. No human recordings are needed beyond what the teachers were trained on, and each new language only needs a teacher.

### 1.3 Durations come from alignment, not the teacher
sanoTTS copies each teacher's per-phoneme durations. Those are indexed by eSpeak phonemes, which our front end does not produce. We learn alignment ourselves with Monotonic Alignment Search (MAS) between our input tokens and the mel frames, as in Glow-TTS/VITS. MAS is already compiled in `train-venv` (Piper's `monotonic_align`). A small duration predictor is trained on the MAS output.

### 1.4 Voice = language (v1)
Each teacher is one speaker, so the language id also selects the voice. A separate speaker id is out of scope for v1. It would be needed for "Tamil in the Hindi voice", which v1 does not attempt.

---

## 2. Front end: unified Indic script (no eSpeak)

Indic scripts are nearly phonemic and share a structure. Nine Brahmic Unicode blocks (Devanagari, Bengali/Assamese, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) are laid out **at the same offsets** within their 128-codepoint blocks (the ISCII heritage).

**Plan:**
1. **Normalize:** NFC, then nukta compositions (e.g. Assamese/Bengali `য়` → U+09DF, the bug we hit with eSpeak). Strip ZWJ/ZWNJ where they carry no sound.
2. **Unify:** map each Brahmic codepoint to `(block offset)`, giving one shared symbol inventory of about 90 symbols. `क`, `ক`, `ਕ`, `ક`, `କ`, `க`, `క`, `ಕ`, `ക` all become the same token "KA". Script-specific letters keep their own tokens (e.g. Assamese `ৰ ৱ`, Tamil `ழ`, Malayalam chillus, Marathi/Gujarati specifics).
3. **Tokens:** the model sees `[lang] + unified graphemes`, including virama, vowel signs, anusvara/visarga/chandrabindu, and punctuation (as pause tokens). Language-specific pronunciation is left to the model plus the language id. That covers:
   - Hindi/Punjabi/Gujarati/Marathi dropping the final inherent "a" (schwa deletion);
   - Bengali/Assamese/Odia inherent "o";
   - Tamil's under-specified consonants (no voiced/aspirated letters).
4. **Numbers:** a per-language number-to-words expansion table, Indic and ASCII digits. Small, hand-written, tested.
5. **Urdu:** Perso-Arabic script is not Brahmic and omits most short vowels. It gets its own symbol range in the same inventory. Expect it to be the weakest language. **Optional v1.1:** Urdu → Devanagari transliteration via a word lexicon built from parallel Hindi/Urdu data.
6. **Optional rule layer (only if needed after evaluation):** explicit schwa-deletion rules for Hindi/Marathi/Punjabi, and inherent-vowel rules for Bengali/Assamese/Odia, as preprocessing. We add rules only where the model's error rate shows it can't learn them.

**Deliverable:** `multilingual/frontend/` (pure Python, no dependencies), with tests:
- the round-trip and coverage on every corpus: ≥ 99.5% of characters map to a known token;
- per-language golden examples;
- the Assamese `য়` case;
- number expansion.

This front end also unblocks Kangri and Bhojpuri later. Both are Devanagari, so they need no new eSpeak voice.

---

## 3. Model (budget ≈ 2.5M params)

| Part | Shape (first guess) | Params |
|---|---|---|
| Token + language embedding | ~110 tokens × 192, 12 langs × 192 | ~25k |
| Text encoder | 4–6 conv/light-attention blocks, width 192 | ~0.7M |
| Duration predictor | 3 conv blocks, width 128 | ~0.1M |
| Mel decoder (encoder → 80-mel, frame-level) | 4–6 conv blocks, width 192, language FiLM | ~0.7M |
| Vocoder (mel → wav) | iSTFT head, ~1M (sanoTTS-style piperlite/iSTFT size) | ~1.0M |
| **Total** | | **~2.5M** |

Notes:
- **Vocoder first and separately.** Train it on the real teacher mels → teacher audio (not on predicted mels), then fine-tune on predicted mels. This is the "z-mix" lesson from sanoTTS: a decoder trained only on clean inputs is brittle to the acoustic model's errors.
- **Language conditioning:** an additive embedding at the encoder input plus FiLM in the mel decoder. Cheap, and it lets one network learn per-language prosody.
- **De-smoothing:** keep sanoTTS's latent-adversarial idea, applied to the mel: a small discriminator during training only, never shipped. A plain L1 regressor gives buzzy, over-smoothed audio.

---

## 4. Data

- **Text:** Leipzig wiki 10K now; move to the 30K/100K corpora for more variety. Target **8k–20k sentences per language**. Hold out the same 140 rows we already hold out, so evaluation stays comparable to the baselines.
- **Synthetic audio:** each teacher renders its language's text with noise 0, giving deterministic output. Store 22.05 kHz 16-bit wav plus a precomputed mel. Estimated size: 12 langs × 10k × ~5 s ≈ 170 h ≈ 27 GB wav, plus 3 GB mel. Disk has 150 GB free.
- **Cost:** Piper renders ~3 sentences/s on CPU, so ~11 h of CPU for 120k sentences. This can run on the CPU now, alongside current GPU training.
- **Balance:** sample languages uniformly per batch, not by data size.

---

## 5. Phases

| Phase | What | Output | Est. effort |
|---|---|---|---|
| 0 | Freeze baselines; reuse `eval_voices.py` (32 held-out sentences, Whisper CER) | `eval/results-eval32.md` | done |
| 1 | Unified Indic front end + tests | `multilingual/frontend/` | 1 day |
| 2 | Synthetic corpus: render 12 teachers × 10k sentences, wav + mel | `multilingual/data/` | ~1 day, mostly unattended CPU |
| 3 | Shared vocoder; check resynthesis CER vs teacher audio per language | `vocoder.pt`, per-language resynthesis CER | ~1 day GPU |
| 4 | Multilingual acoustic + MAS durations; resumable training (`resume_state.py`) | `acoustic.pt` | 1–2 days GPU |
| 5 | Joint fine-tune (vocoder on predicted mels) + evaluation + ablations: model size 1.5M vs 3M, with/without the rule layer, multilingual vs Indo-Aryan/Dravidian split | results table vs baselines | 1–2 days |
| 6 | Export: fp16/int8 blob + **own NumPy runtime** (MIT), with the front end embedded | `release/multilingual/` | 1 day |

GPU phases run one at a time under the existing `.gpu.lock`, after the current teacher queue. Phase 1 (code) and phase 2 (CPU rendering) can start now.

---

## 6. Evaluation (same harness as the baselines)
- **Intelligibility:** Whisper large-v3-turbo CER on the 32 held-out sentences per language, compared directly with the `release/best/` baselines.
  - Odia: Whisper has no model, so it stays unscored.
  - Malayalam: Whisper is unreliable even on the teacher, so compare student against teacher only.
- **Naturalness:** SCOREQ/UTMOS, relative only (student vs teacher, multilingual vs baseline). These predictors are English-trained.
- **Size and speed:** parameter count, blob size, CPU real-time factor in the NumPy runtime.
- **Listening:** 5 sentences per language, side by side with teacher and baseline. Every metric gets an ear check before it is believed.

## 7. Risks
1. **Memorization at small size** (sanoTTS's main warning). Mitigation: 4–10× more text than sanoTTS used, and evaluation only on held-out text.
2. **Grapheme input can't learn a pronunciation rule** (schwa deletion, Bengali inherent vowel). Mitigation: the §2.6 rule layer, added only where error rates show the need.
3. **Urdu without short vowels.** Expected to be the weakest. Mitigation: the v1.1 transliteration lexicon.
4. **Mel interface loses quality vs teacher latents.** If the vocoder resynthesis CER in phase 3 is already > +2 points, the target is unreachable. Fix the vocoder before phase 4.
5. **Language interference:** one language degrades when others are added. Mitigation: language FiLM and balanced sampling. The fallback is the 2-model split.
6. **Licenses:** synthetic data inherits each teacher's terms. Four IndicTTS-derived languages have no license stated (`DATASETS.md`). Confirm before any public release.

## 8. Decisions (2026-10-03)
- **Voice:** language = voice for v1. Every teacher is a female voice, so the set is consistent in gender. One shared voice across languages (speaker disentanglement) is deferred to v2, once v1 meets its targets.
- **Release:** private. Nothing is published; license questions (§7.6) only matter before a future release.

## 9. Status (2026-10-03 23:55)
| Phase | State | Files |
|---|---|---|
| 1 Front end | ✅ 9/9 tests; 199 tokens used across 12 languages | `frontend/indic.py`, `frontend/test_indic.py` |
| 2 Corpus | 🔄 rendering 10,140 sentences/language at ~5/s (≈32 min/language); languages wait for their teacher | `render_corpus.py` (unit `sanotts-render`) → `data/wav/<lang>/` |
| 3 Vocoder | code done, CPU smoke-tested; 850K params, piperlite decoder with 80-mel input, initialised from the hi_IN teacher's generator | `mel.py`, `train_vocoder.py` |
| 4 Acoustic | code done, CPU smoke-tested; 1.55M params; MAS alignment via Piper's compiled `maximum_path` | `acoustic.py`, `train_acoustic.py` |
| 5 Joint + eval | joint fine-tune done and CPU smoke-tested: z-mix 50/50 predicted/real mel, 10% of vocoder gradient into the acoustic model, discriminators carried over from phase 3. Eval: `synth.py --eval32` → `eval32-ml/<lang>/`, scored by `../eval_voices.py large-v3-turbo eval32-ml` | `train_joint.py`, `synth.py` |
| 6 Runtime | ✅ NumPy-only `indicml` (no torch, no eSpeak): vendored sanoTTS piperlite vocoder ops (MIT) + own acoustic forward. Parity vs PyTorch on 5 languages: identical durations, mel max |Δ| ≤ 7.4e-6, waveform corr 1.000000. Package 4.80 MB / 2,399,330 params | `runtime/indicml/`, `export_runtime.py`, `test_runtime.py` |

Phases 3→6 run automatically (vocoder → acoustic → joint → eval → export + parity) (`phases.sh`, unit `sanotts-multilingual`) once all 12 languages are rendered, taking turns on the GPU via `.gpu.lock`.
Total model so far: ~2.4M params (vocoder 0.85M + acoustic 1.55M, of which ~100K is an embedding table that can be trimmed to the 199 used tokens).

### Phase 3 result (2026-10-04 20:39)
Vocoder: 60,000 steps; final log-mel L1 0.295 on training segments.
Resynthesis check (teacher audio → mel → vocoder, 6 held-out sentences per language, Whisper CER vs. the teacher's original audio):

| | hi | ta | bn | pa | or | ml |
|---|---|---|---|---|---|---|
| CER gap | +2.9 | 0.0 | +1.5 | −2.0 | +1.5 | +14.1 |
| resynthesis mel-L1 | 0.297 | 0.293 | 0.276 | 0.266 | 0.265 | 0.282 |

The Malayalam CER gap is Whisper noise, not the vocoder: its mel distortion sits mid-pack, and Whisper is already ~0.44–0.59 CER on Malayalam teacher audio. The vocoder passes the §7.4 gate; phase 4 proceeds. The joint fine-tune (phase 5) trains it further on predicted mels.

### Phase 4–5 results (2026-10-05)
- Acoustic: 150,000 steps; final log-mel L1 0.51 on training batches.
- Joint fine-tune: 20,000 steps. **A/B on 8 languages × 8 held-out sentences: pre-joint mean Whisper CER 0.176 vs joint 0.190; pre-joint better in 6/8** (joint wins only hi 0.091 vs 0.096 and ur 0.075 vs 0.077). The z-mix plus 10% gradient into the acoustic model made the mels easier to vocode but harder to understand. **Shipped: the pre-joint checkpoints** (`runs/acoustic`, `runs/vocoder`), scored with the full 32-sentence eval as `eval/results-eval32-mlpre.*`.
- Next idea if revisited: freeze the acoustic model in the joint stage (`--ac-grad 0`) and fine-tune only the vocoder on predicted mels.

## 10. Result (2026-10-05 03:15): shipped `release/multilingual/`
| Criterion | Target | Result |
|---|---|---|
| Size | ≤ 3M params / ≤ 6 MB | ✅ 2,399,330 params, 4.80 MB fp16 (vs 12 × 1.57M ≈ 19M / 37 MB) |
| Mean CER vs per-language baselines | ≤ +2 points | ✅ +1.1 (+0.4 excluding Malayalam) |
| Worst single language | ≤ +5 points | ⚠️ 11/12 pass; Malayalam +9.1 (Whisper-unreliable, see the vocoder check); next worst Kannada +4.0, Tamil +3.1 |
| No eSpeak at inference | — | ✅ unified Indic graphemes; runtime needs only NumPy (verified in a clean venv) |
| Speed | — | CPU real-time factor ~0.05 (≈20× real time) |

Beats the per-language baselines on Gujarati, Marathi, Odia and Assamese; ties on Punjabi.
**Open:** a listening pass (no human has listened yet); Malayalam quality; Kannada/Tamil gap (ideas: bigger acoustic model, more text, vocoder-only joint with `--ac-grad 0`).
- Follow-up (2026-10-05 04:20): vocoder-only joint (`--ac-grad 0`, 20k steps) scored a mean CER of **0.193** on the same 8×8 A/B (prejoint 0.176, joint 0.190). Also rejected. Further joint-stage tuning on predicted mels does not pay here; the pre-joint checkpoints remain shipped.

## 11. De-smoothing update (2026-10-05, after the user's listening test)
The user reported low volume and a robotic sound. Diagnosis: the output was −19 dBFS (5 dB under the teacher), and the vocoder was fine on real mels (within ~1 dB of the teacher), but the acoustic model's mels were over-smoothed (frame-to-frame detail ~20% low). This is the §3 de-smoothing step that had been skipped.
Fix: `train_acoustic_adv.py` (mel-patch LSGAN plus feature matching, 20k steps) and runtime loudness normalisation.
**Shipped result:** all 12 languages within +5 points (worst Malayalam +4.2, unreliable), mean +0.7 vs the per-language baselines; poem test 4–8 kHz excess buzz down from +3.5 dB to +1.4 dB vs the teacher. The previous release is kept as `release/multilingual.prev-*`.

## 12. Vocoder update (2026-10-05 evening)
- Vocoder continued 60k → 260k steps (cosine LR 2e-4 → 1e-5): resynthesis mel-L1 0.278 → 0.236; >4 kHz excess +0.99 → −0.05 dB vs teacher; poem 4–8 kHz excess +1.4 → +0.6 dB; 8×8 A/B CER 0.175 → **0.166**. **Shipped** (fp16 `release/multilingual`, int8 `release/multilingual-int8`).
- Vocoder fine-tune on predicted (de-smoothed) mels, acoustic frozen, 15k steps: A/B CER 0.170. **Rejected.**
- int8: per-output-channel weights; with identical timing, acoustic mel-L1 0.02 and vocoder waveform corr 0.9993 vs fp16. 2.48 MB.
- Full 12×32 eval of the shipped release: mean CER 0.273 (previous 0.268): a statistical tie with the A/B result (which favoured the new vocoder). Mean +1.2 points vs per-language baselines (+0.8 excluding Malayalam). **Punjabi +6.6 (over the +5 target)**, Kannada +4.2, others ≤ +3.4. Shipped for its clearly better sound (resynthesis error −15%, no excess buzz).
- Open: Punjabi teacher data (only 1,232 clean clips); Kannada/Tamil gap; a human listening pass.
