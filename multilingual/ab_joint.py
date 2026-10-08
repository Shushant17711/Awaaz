#!/usr/bin/env python3
"""A/B: pre-joint (phase 3+4) vs joint (phase 5) multilingual model.

Renders the same held-out sentences (rows 13-20) with both, scores Whisper CER,
prints per-language and mean, and writes eval/ab-joint.json. The joint stage's
vocoder loss drifted up slightly during training (0.43 -> ~0.46-0.50), so the
joint result is not assumed to be better.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from synth import Synthesizer  # noqa: E402
from eval_voices import cer, LANG, TO_BENGALI, to_bengali  # noqa: E402

LANGS = ["hi_IN", "ta_IN", "te_IN", "bn_BD", "kn_IN", "gu_IN", "mr_IN", "ur_PK"]
VARIANTS = {"prejoint": (HERE / "runs/acoustic/acoustic.pt", HERE / "runs/vocoder/vocoder.pt"),
            "joint": (HERE / "runs/joint/acoustic.pt", HERE / "runs/joint/vocoder.pt"),
            "jointvoc": (HERE / "runs/joint-voc/acoustic.pt", HERE / "runs/joint-voc/vocoder.pt"),
            "adv": (HERE / "runs/acoustic-adv/acoustic.pt", HERE / "runs/vocoder/vocoder.pt"),
            "advlong": (HERE / "runs/acoustic-adv/acoustic.pt", HERE / "runs/vocoder-long/vocoder.pt"),
            "advlongft": (HERE / "runs/acoustic-adv/acoustic.pt", HERE / "runs/vocoder-ft/vocoder.pt")}
import os
VARIANTS = {k: VARIANTS[k] for k in os.environ.get("AB_VARIANTS", "prejoint,joint").split(",")}


def main() -> int:
    from faster_whisper import WhisperModel  # noqa: PLC0415
    w = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=8)
    out = Path("/tmp/awaaz-ab"); out.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict[str, float]] = {}
    for name, (a, v) in VARIANTS.items():
        syn = Synthesizer(a, v, "cpu")
        for lang in LANGS:
            rows = [json.loads(l)["text"] for l in (HERE / "data/text" / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines()][12:20]
            tot = 0.0
            for i, t in enumerate(rows):
                p = out / f"{name}_{lang}_{i}.wav"
                sf.write(p, syn(t, lang), 22050)
                ref = to_bengali(t, TO_BENGALI[lang[:2]]) if lang[:2] in TO_BENGALI else t
                tot += cer(ref, "".join(s.text for s in w.transcribe(str(p), language=LANG[lang[:2]], beam_size=5)[0]))
            res.setdefault(lang, {})[name] = tot / len(rows)
            print(f"{name:9s} {lang}: CER {res[lang][name]:.3f}", flush=True)
    for name in VARIANTS:
        print(f"MEAN {name}: {sum(r[name] for r in res.values()) / len(res):.3f}")
    (ROOT / "eval" / f"ab-{'-'.join(VARIANTS)}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
