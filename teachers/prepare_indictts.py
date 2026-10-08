#!/usr/bin/env python3
"""IIT Madras IndicTTS (SPRINGLab mirror on Hugging Face) -> Piper single-speaker set.

    prepare_indictts.py <lang code> <HF dataset suffix>    e.g.  ta Tamil

Streams the parquet shards one at a time (the full sets are 3.5-8.7 GB of
48 kHz audio): keeps only the female speaker, resamples to 22.05 kHz 16-bit,
writes teachers/<lang>/wavs/*.wav + metadata.csv ("file.wav|text"), then deletes
the shard. Resumable: finished shards are recorded in shards.done.
"""

from __future__ import annotations

import io
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
SR = 22050


def main() -> int:
    lang, suffix = sys.argv[1], sys.argv[2]
    repo = f"SPRINGLab/IndicTTS_{suffix}"
    out = ROOT / "teachers" / lang
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    done_path = out / "shards.done"
    done = set(done_path.read_text().split()) if done_path.is_file() else set()

    api = json.loads(subprocess.run(["curl", "-sfL", "--retry", "5", f"https://huggingface.co/api/datasets/{repo}"],
                                    capture_output=True, text=True, check=True).stdout)
    shards = sorted(s["rfilename"] for s in api["siblings"] if s["rfilename"].endswith(".parquet"))
    cap = int(os.environ.get("MAX_UTTS", "3500"))  # ~5-6 h of studio speech: plenty to fine-tune a teacher
    for shard in shards:
        if shard in done:
            continue
        have = sum(1 for _ in (out / "metadata.csv").open(encoding="utf-8")) if (out / "metadata.csv").is_file() else 0
        if have >= cap:
            break
        local = out / "shard.parquet"
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard}"
        for attempt in range(8):
            if subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "10", "--retry-delay", "5",
                               "-o", str(local), url]).returncode == 0:
                break
        else:
            print(f"download failed: {shard}", file=sys.stderr)
            return 1
        table = pq.read_table(local, columns=["audio", "text", "gender"])
        rows = []
        for i, (audio, text, gender) in enumerate(zip(table.column("audio").to_pylist(),
                                                      table.column("text").to_pylist(),
                                                      table.column("gender").to_pylist())):
            if gender != 0 or not text or not str(text).strip():
                continue
            data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            data = data.mean(axis=1)
            if sr != SR:
                data = soxr.resample(data, sr, SR)
            peak = float(np.abs(data).max()) or 1.0
            if peak > 0.99:
                data = data * (0.99 / peak)
            name = f"{Path(shard).stem}-{i:05d}.wav"
            sf.write(wav_dir / name, data, SR, subtype="PCM_16")
            text = " ".join(str(text).replace('"', "").split())  # no CSV quote chars (see train_teacher.sh)
            if lang == "as":
                # eSpeak's Assamese voice spells out য + nukta letter by letter; the
                # precomposed U+09DF reads correctly (ɔχəmia for অসমীয়া).
                text = text.replace("\u09af\u09bc", "\u09df")
            rows.append(f"{name}|{text}")
        with (out / "metadata.csv").open("a", encoding="utf-8") as handle:
            handle.write("".join(r + "\n" for r in rows))
        local.unlink()
        with done_path.open("a") as handle:
            handle.write(shard + "\n")
        print(f"{lang}: {shard} -> {len(rows)} female utterances", flush=True)
    (out / "audio_dir").write_text(str(wav_dir))
    n = sum(1 for _ in (out / "metadata.csv").open(encoding="utf-8"))
    print(f"{lang}: done, {n} utterances", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
