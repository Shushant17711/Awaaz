#!/usr/bin/env python3
"""OpenSLR Google crowdsourced Indic set -> Piper multi-speaker training CSV.

Input:  data/audio/<lang>_in_female/{line_index.tsv, *.wav}  (48 kHz, many speakers)
Output: teachers/<lang>/metadata.csv  "<file>.wav|<speaker>|<text>"
        teachers/<lang>/speakers.json {speaker: utterance count}
Audio is left where it is; Piper resamples to 22.05 kHz while caching.
Speakers with fewer than --min-utts are dropped: the speaker embedding needs
enough examples, and the frozen speaker is picked from the best-covered ones.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("--min-utts", type=int, default=40)
    args = parser.parse_args()

    src = ROOT / "data" / "audio" / f"{args.lang}_in_female"
    tsv = next(src.rglob("line_index.tsv"))
    wav_dir = tsv.parent
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        utt, text = line.split("\t", 1)
        text = " ".join(text.split())
        if not text or not (wav_dir / f"{utt}.wav").is_file():
            continue
        rows.append((utt, utt.split("_")[1], text))
    counts = collections.Counter(speaker for _, speaker, _ in rows)
    keep = {s for s, n in counts.items() if n >= args.min_utts}
    out = ROOT / "teachers" / args.lang
    out.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if r[1] in keep]
    with (out / "metadata.csv").open("w", encoding="utf-8") as handle:
        for utt, speaker, text in kept:
            handle.write(f"{utt}.wav|{speaker}|{text}\n")
    (out / "speakers.json").write_text(json.dumps(dict(counts.most_common()), indent=1))
    (out / "audio_dir").write_text(str(wav_dir))
    print(f"{args.lang}: {len(kept)}/{len(rows)} utterances, {len(keep)} speakers (min {args.min_utts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
