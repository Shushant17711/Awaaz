#!/usr/bin/env python3
"""Compare an old and a new teacher ONNX on 10 held-out sentences (Whisper CER).

    teacher_gate.py <loc> <old.onnx> <new.onnx> <config.json>
Prints JSON {"old": cer, "new": cer, "accept": bool}. Accept when new <= old + 0.01.
Odia/Assamese are scored through Bengali Whisper on Bengali-transliterated text
(the same proxy eval_voices.py uses).
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_voices import LANG, TO_BENGALI, cer, to_bengali  # noqa: E402


def main() -> int:
    loc, old, new, cfg = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    from faster_whisper import WhisperModel  # noqa: PLC0415
    from piper import PiperVoice  # noqa: PLC0415
    w = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=8)
    texts = [json.loads(l)["text"] for l in (ROOT / "multilingual/data/text" / f"{loc}.jsonl").read_text(encoding="utf-8").splitlines()][44:54]
    res = {}
    for name, onnx in (("old", old), ("new", new)):
        v = PiperVoice.load(str(onnx), config_path=str(cfg))
        tot = 0.0
        for t in texts:
            p = f"/tmp/awaaz-gate_{loc}.wav"
            with wave.open(p, "wb") as f:
                v.synthesize_wav(t, f)
            ref = to_bengali(t, TO_BENGALI[loc[:2]]) if loc[:2] in TO_BENGALI else t
            tot += cer(ref, "".join(s.text for s in w.transcribe(p, language=LANG[loc[:2]], beam_size=5)[0]))
        res[name] = round(tot / len(texts), 4)
    res["accept"] = res["new"] <= res["old"] + 0.01
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
