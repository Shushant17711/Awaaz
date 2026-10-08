# Kangri (Himachali): recording kit

No Kangri speech dataset exists anywhere, so this voice needs recordings. Everything else is ready.

## What to record
`recording_script.tsv` has 1,500 sentences (about 2.4 hours read aloud). They come from the CC0 Kangri corpus (github.com/chauhanshweta/Kangri_corpus), selected to cover 90% of the sound combinations in the whole corpus.

- The first 300 sentences already cover most of the sounds. **Around 500 sentences (~45 min) is enough for a first voice.** More helps.
- One speaker only, a native Kangri speaker. Read naturally, at a steady pace.

## How to record
- A quiet room with soft furnishings. Same mic, same distance (~20 cm), same volume every session.
- WAV, mono, 22,050 Hz or higher (44.1/48 kHz is fine), 16-bit.
- One file per sentence, named by its id: `kangri_00001.wav`, `kangri_00002.wav`, …
- About 0.3 s of silence before and after. Re-record any mistakes; don't correct yourself mid-sentence.
- Phone apps work if nothing better is available (e.g. a voice recorder set to WAV). Consistency matters more than equipment.

Put the files in `kangri/wavs/`.

## Then
```
teachers/kangri.sh   # run from the repo root
```
This builds `metadata.csv` from whichever ids have a WAV, fine-tunes a teacher from the Hindi base checkpoint, exports it, and queues distillation, the same way as the other languages. Text goes through the Hindi eSpeak voice, the closest available front end.

Expect a proof-of-concept voice, not studio quality (see plan.md §6).
