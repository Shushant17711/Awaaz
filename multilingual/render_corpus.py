#!/usr/bin/env python3
"""Phase 2: render the synthetic multilingual corpus from the frozen teachers.

For each language, the teacher reads multilingual/data/text/<loc>.jsonl
(140 held-out + 10k train rows) into multilingual/data/wav/<loc>/<id>.wav
(22.05 kHz, 16-bit mono), deterministically (noise 0, as sanoTTS packs do),
and appends {"id","text","split","wav","seconds"} to <loc>/manifest.jsonl.

Resumable: rows whose wav exists are skipped. Languages whose teacher is not
trained yet are skipped and picked up on the next pass; run with --wait to keep
polling until all 12 are rendered.

Mels are not stored here; they are computed in phase 3 from the wavs, so the
mel config can change without re-rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = ROOT / "sanoTTS" / "models" / "teachers"
DATA = Path(os.environ.get("ML_DATA", str(ROOT / "multilingual" / "data")))
# locale -> teacher voice (see DATASETS.md / progress.md for provenance)
TEACHER = {
    "hi_IN": "hi_IN-priyamvada-medium", "te_IN": "te_IN-padmavathi-medium", "ml_IN": "ml_IN-meera-medium",
    "ur_PK": "ur_PK-aegis_female-medium", "bn_BD": "bn_BD-google4811-medium", "mr_IN": "mr_IN-google01523-medium",
    "ta_IN": "ta_IN-indictts-medium", "kn_IN": "kn_IN-indictts-medium", "gu_IN": "gu_IN-indictts-medium",
    "pa_IN": "pa_IN-indictts-medium", "or_IN": "or_IN-indictts-medium", "as_IN": "as_IN-indictts-medium",
}


def question_rise(path: Path, semitones: float = 5.0, tail_s: float = 0.45) -> None:
    """Give a rendered sentence a rising terminal pitch (the teachers don't do it themselves).

    The last tail_s seconds are pitch-shifted by an amount that ramps 0 -> `semitones`:
    four shifted copies of the tail are cross-faded over time, then the tail is joined back
    with a short cross-fade. Used only for corpus rows marked "question".
    """
    import librosa  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    y, sr = sf.read(path, dtype="float32")
    n = int(tail_s * sr)
    if len(y) < 2 * n:
        return
    tail = y[-n:]
    steps = [0.0, semitones / 3, 2 * semitones / 3, semitones]
    shifted = [tail] + [librosa.effects.pitch_shift(tail, sr=sr, n_steps=k) for k in steps[1:]]
    pos = np.linspace(0, len(steps) - 1, n)                      # where along the ramp each sample is
    out = np.zeros(n, dtype=np.float32)
    for i, sh in enumerate(shifted):
        w = np.clip(1 - np.abs(pos - i), 0, 1)                   # triangular cross-fade weights
        out += w * sh[:n]
    fade = int(0.02 * sr)
    out[:fade] = out[:fade] * np.linspace(0, 1, fade) + tail[:fade] * np.linspace(1, 0, fade)
    y[-n:] = out
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")


def render_language(loc: str, threads: int) -> str:
    voice_name = TEACHER[loc]
    onnx_path = TEACHERS / voice_name / f"{voice_name}.onnx"
    if not onnx_path.is_file():
        return "waiting"
    from piper import PiperVoice, SynthesisConfig  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    voice = PiperVoice.load(str(onnx_path))
    voice.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    syn = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0, length_scale=1.0)

    out_dir = DATA / "wav" / loc
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.is_file():
        done = {json.loads(l)["id"] for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = [json.loads(l) for l in (DATA / "text" / f"{loc}.jsonl").read_text(encoding="utf-8").splitlines()]
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return "done"
    started, n = time.time(), 0
    with manifest.open("a", encoding="utf-8") as man:
        for row in todo:
            path = out_dir / f"{row['id']}.wav"
            try:
                with wave.open(str(path), "wb") as wav:
                    voice.synthesize_wav(row["text"], wav, syn_config=syn)
                with wave.open(str(path), "rb") as wav:
                    seconds = wav.getnframes() / wav.getframerate()
            except Exception as exc:  # noqa: BLE001 - one bad sentence must not stop the corpus
                print(f"{loc} {row['id']}: render failed: {exc}", file=sys.stderr, flush=True)
                path.unlink(missing_ok=True)
                continue
            if seconds < 0.3:
                path.unlink(missing_ok=True)
                continue
            if row.get("question"):
                question_rise(path)
            man.write(json.dumps({**row, "wav": str(path.relative_to(DATA)), "seconds": round(seconds, 3)},
                                 ensure_ascii=False) + "\n")
            man.flush()
            n += 1
            if n % 500 == 0:
                print(f"{loc}: {len(done) + n}/{len(rows)} ({n / (time.time() - started):.1f} rows/s)", flush=True)
    return "done"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="keep polling until every teacher's language is rendered")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--langs", default="", help="comma list of locales to render (default: all)")
    args = ap.parse_args()
    while True:
        states = {}
        for loc in [l for l in TEACHER if not args.langs or l in args.langs.split(",")]:
            states[loc] = render_language(loc, args.threads)
            if states[loc] == "done":
                print(f"{loc}: rendered", flush=True)
        if not args.wait or all(s == "done" for s in states.values()):
            break
        waiting = [k for k, v in states.items() if v == "waiting"]
        print(f"waiting for teachers: {waiting}", flush=True)
        time.sleep(600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
