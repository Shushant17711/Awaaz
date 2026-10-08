"""python -m indicml PACKAGE LANG "text" out.wav [--length-scale 1.0]"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np

from .model import Voice


def main() -> int:
    ap = argparse.ArgumentParser(prog="indicml")
    ap.add_argument("package"); ap.add_argument("lang"); ap.add_argument("text"); ap.add_argument("out")
    ap.add_argument("--length-scale", type=float, default=1.0)
    args = ap.parse_args()
    voice = Voice(args.package)
    t0 = time.time()
    audio = voice.synthesize(args.text, args.lang, args.length_scale)
    dt = time.time() - t0
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(args.out, "wb") as w:            # stdlib only: no soundfile needed
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(voice.sample_rate); w.writeframes(pcm.tobytes())
    sec = len(audio) / voice.sample_rate
    print(f"{args.out}: {sec:.2f}s audio in {dt:.2f}s (RTF {dt / max(sec, 1e-9):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
