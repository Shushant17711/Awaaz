
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
