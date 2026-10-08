#!/usr/bin/env python3
"""Drop mispaired text/audio rows from a teacher dataset using Whisper.

    asr_filter.py <lang> [--whisper-lang pa] [--max-cer 0.6] [--to-bengali-base 0x0B00]

~50% of IndicTTS Punjabi rows pair a transcript with a different sentence's
audio (40-clip sample: 16 match, 20 clearly don't; not a fixed offset, so they
can't be realigned). Training on them gave a babbling teacher. This transcribes
every row of teachers/<lang>/metadata.train.csv and keeps rows whose
character error rate is <= --max-cer. Whisper is weak on these languages, so the
threshold is loose: it only needs to separate "roughly this sentence" (0.3-0.6)
from "a different sentence" (>0.8).

Writes metadata.train.csv (filtered) and keeps the original as
metadata.train.unfiltered.csv; per-row scores go to asr_scores.tsv. Resumable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval_voices import cer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--whisper-lang")
    ap.add_argument("--max-cer", type=float, default=0.6)
    ap.add_argument("--to-bengali-base", type=lambda s: int(s, 0), default=0,
                    help="transliterate text from this Brahmic block to Bengali before scoring (for languages Whisper lacks)")
    args = ap.parse_args()
    d = HERE / args.lang
    src = d / "metadata.train.unfiltered.csv"
    if not src.is_file():
        (d / "metadata.train.csv").rename(src)
    rows = [l.rstrip("\n") for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    audio = Path((d / "audio_dir").read_text().strip())
    scores_path = d / "asr_scores.tsv"
    done = {}
    if scores_path.is_file():
        for l in scores_path.read_text(encoding="utf-8").splitlines():
            f, c = l.split("\t")[:2]
            done[f] = float(c)

    from faster_whisper import WhisperModel  # noqa: PLC0415
    import os  # noqa: PLC0415
    if os.environ.get("ASR_DEVICE") == "cuda":   # ~1 GB VRAM; 10x the shared-CPU speed
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    else:
        model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8",
                             cpu_threads=int(os.environ.get("ASR_THREADS", "8")))
    base = args.to_bengali_base

    def bengali(t: str) -> str:
        return "".join(chr(ord(c) - base + 0x0980) if base <= ord(c) < base + 0x80 else c for c in t) if base else t

    with scores_path.open("a", encoding="utf-8") as out:
        for n, row in enumerate(rows, 1):
            f, text = row.split("|", 1)[0], row.split("|")[-1]
            if f in done:
                continue
            hyp = "".join(s.text for s in model.transcribe(str(audio / f), language=args.whisper_lang or args.lang,
                                                           beam_size=5)[0])
            done[f] = cer(bengali(text), hyp)
            out.write(f"{f}\t{done[f]:.4f}\t{hyp}\n")
            out.flush()
            if n % 200 == 0:
                print(f"{args.lang}: {n}/{len(rows)} scored", flush=True)
    kept = [r for r in rows if done.get(r.split("|", 1)[0], 9) <= args.max_cer]
    (d / "metadata.train.csv").write_text("".join(r + "\n" for r in kept), encoding="utf-8")
    print(f"{args.lang}: kept {len(kept)}/{len(rows)} rows with Whisper CER <= {args.max_cer}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
