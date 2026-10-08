# v2 unattended run: fixes for the listening-review feedback

Runs as the user service `sanotts-v2` (auto-starts at login, resumes after reboot/crash). Model size is unchanged (2.4M params, 4.8 MB / 2.5 MB int8).

| Feedback | What v2 does |
|---|---|
| Sibilant harshness / hiss | mel range 0–8 kHz → full 0–11 kHz, so the vocoder no longer guesses the top band |
| Metallic Odia/Assamese/Kannada; dataset disparity | those teachers (+ Punjabi) trained +10k steps; each kept only if Whisper CER doesn't worsen |
| Mispronunciations, smeared conjuncts, micro-pauses | corpus doubled (+10k new sentences per language) |
| Flat questions | ~520–880 question sentences/language whose teacher audio gets a +5-semitone terminal rise |
| Robotic | vocoder 150k + acoustic 150k + de-smoothing 20k steps on the new data |

**Check progress:** `journalctl --user -u sanotts-v2 -o cat | grep "^=="`
**Result:** `cat multilingual/v2/REPORT-v2.md` (written at the end)
**Listen:** `review/indic_tts_review_v2.mp3` (each language ends with a question)
**Stop:** `systemctl --user stop sanotts-v2` (saves progress) · **resume:** `systemctl --user start sanotts-v2`
**Remove the auto-start when finished:** `systemctl --user disable sanotts-v2`

Safety: `release/multilingual` is replaced **only** if v2's 12-language Whisper score is not worse than the current release (previous release kept as `release/multilingual.prev-*`); otherwise v2 goes to `release/candidate-v2` for listening.
Estimated duration: ~25–30 h (teachers ~6 h ‖ rendering, then ~3–4 h rendering, vocoder ~8 h, acoustic ~4 h, de-smoothing ~1 h, evaluation ~2 h).
