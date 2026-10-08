#!/usr/bin/env python3
"""Intelligibility gate for distilled voices (local stand-in for upstream s10).

For every voice with rendered samples (samples/<voice>/{teacher,student}/ from
render_samples.sh: 8 held-out sentences), transcribe teacher and student audio
with Whisper and score character error rate against the source text.
The number that matters is the GAP: student CER - teacher CER. The teacher's
own CER is the ceiling Whisper can hear for that language; a student can't do
better than its teacher, only lose less.

Writes eval/results.json and eval/results.md. Re-running skips voices already scored.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANG = {"hi": "hi", "te": "te", "ml": "ml", "ur": "ur", "bn": "bn", "mr": "mr", "ta": "ta", "kn": "kn",
        "gu": "gu", "pa": "pa",
        # Proxy scoring: Whisper has no usable Odia/Assamese model (Assamese real-recording CER 0.91),
        # but both share Bengali's Brahmic letter layout. Transcribe as Bengali and compare against the
        # reference transliterated into the Bengali block. Validated on real IndicTTS recordings:
        # Odia 20/20 and Assamese 19/20 clips under 0.6 CER. Comparable within a language, not across.
        "or": "bn", "as": "bn"}
TO_BENGALI = {"or": 0x0B00, "as": 0x0980}  # source Brahmic block per proxy-scored language


def to_bengali(text: str, base: int) -> str:
    return "".join(chr(ord(c) - base + 0x0980) if base <= ord(c) < base + 0x80 else c for c in text)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥.,!?;:\"'()\[\]\-–—،۔؟]", " ", text)
    return re.sub(r"\s+", "", text).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def main() -> int:
    from faster_whisper import WhisperModel  # noqa: PLC0415

    out_dir = ROOT / "eval"
    out_dir.mkdir(exist_ok=True)
    results_path = out_dir / (f"results-{sys.argv[2]}.json" if len(sys.argv) > 2 else "results.json")
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    model = WhisperModel(sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo", device="cpu", compute_type="int8")

    sample_root = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "samples")
    for sample_dir in sorted(sample_root.iterdir()):
        name = sample_dir.name
        manifest = sample_dir / "student" / "manifest.jsonl"
        if name in results or not manifest.is_file():
            continue
        lang = LANG.get(name[:2])
        if lang is None:
            print(f"{name}: no Whisper model for this language; unscored", flush=True)
            continue
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        per = {"teacher": [], "student": []}
        for row in rows:
            if not row.get("ok", True):
                continue
            for lane in per:
                segs, _ = model.transcribe(str(sample_dir / lane / row["wav"]), language=lang, beam_size=5)
                hyp = "".join(s.text for s in segs)
                ref = to_bengali(row["text"], TO_BENGALI[name[:2]]) if name[:2] in TO_BENGALI else row["text"]
                per[lane].append({"text": row["text"], "hyp": hyp, "cer": cer(ref, hyp)})
        t = sum(x["cer"] for x in per["teacher"]) / max(len(per["teacher"]), 1)
        s = sum(x["cer"] for x in per["student"]) / max(len(per["student"]), 1)
        results[name] = {"teacher_cer": t, "student_cer": s, "gap": s - t, "n": len(per["student"]), "detail": per}
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:34s} teacher CER {t:.3f}  student CER {s:.3f}  gap {s - t:+.3f}", flush=True)

    lines = ["| voice | teacher CER | student CER | gap (lower=better) |", "|---|---|---|---|"]
    for name, r in sorted(results.items()):
        lines.append(f"| {name} | {r['teacher_cer']:.3f} | {r['student_cer']:.3f} | {r['gap']:+.3f} |")
    (results_path.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
