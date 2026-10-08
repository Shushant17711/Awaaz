#!/usr/bin/env python3
"""Synthesize with the multilingual model, and lay out an evaluation set.

    python synth.py --lang hi_IN --text "..." --out x.wav          # one sentence
    python synth.py --eval32                                         # all languages

--eval32 renders held-out rows 13-44 (the same 32 sentences the per-language
baselines are scored on) into ../eval32-ml/<lang>/{student,teacher}/ with a
manifest per lane; score with `../eval_voices.py large-v3-turbo eval32-ml`.
The teacher lane is the teacher's own rendering from the synthetic corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "frontend"))
sys.path.insert(0, str(ROOT / "sanoTTS" / "tools"))
import indic  # noqa: E402
import mel as melmod  # noqa: E402
import train_roota_piper_decoder_student as dec  # noqa: E402
from acoustic import MultilingualAcoustic  # noqa: E402


class Synthesizer:
    def __init__(self, acoustic: Path, vocoder: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        a = torch.load(acoustic, map_location="cpu", weights_only=False)
        self.acoustic = MultilingualAcoustic(**a["config"])
        self.acoustic.load_state_dict(a["model_state_dict"])
        self.acoustic.to(self.device).eval()
        v = torch.load(vocoder, map_location="cpu", weights_only=False)
        c = v["config"]
        self.vocoder = dec.DecoderStudent(in_channels=c["in_channels"], channels=tuple(c["channels"]),
                                          res_layers=c["res_layers"], variant=c["variant"], activation=c["activation"])
        self.vocoder.load_state_dict(v["model_state_dict"])
        self.vocoder.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, text: str, lang: str, length_scale: float = 1.0) -> np.ndarray:
        ids = torch.tensor([indic.encode(text)], device=self.device)
        mel = self.acoustic.infer(ids, torch.tensor([indic.LANG_ID[lang]], device=self.device), length_scale)
        audio = self.vocoder(mel)
        audio = audio[0] if isinstance(audio, tuple) else audio
        return audio.reshape(-1).clamp(-1, 1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", type=Path, default=HERE / "runs/acoustic/acoustic.pt")
    ap.add_argument("--vocoder", type=Path, default=HERE / "runs/vocoder/vocoder.pt")
    ap.add_argument("--lang"); ap.add_argument("--text"); ap.add_argument("--out", type=Path)
    ap.add_argument("--eval32", action="store_true")
    ap.add_argument("--tag", default="ml", help="writes to ../eval32-<tag>/")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    syn = Synthesizer(args.acoustic, args.vocoder, args.device)
    if not args.eval32:
        sf.write(args.out, syn(args.text, args.lang), melmod.SR)
        return 0
    for lang in indic.LANGS:
        data = Path(os.environ.get("ML_DATA", str(HERE / "data")))
        man = data / "wav" / lang / "manifest.jsonl"
        if not man.is_file():
            continue
        rows = {json.loads(l)["id"]: json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()}
        held = [json.loads(l) for l in (data / "text" / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines()][12:44]
        out = ROOT / f"eval32-{args.tag}" / lang     # separate root: never mixed into the baseline results
        for lane in ("student", "teacher"):
            (out / lane).mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, r in enumerate(held):
            wav = f"{i:05d}.wav"
            ok = r["id"] in rows
            if ok:
                shutil.copy(data / rows[r["id"]]["wav"], out / "teacher" / wav)
                sf.write(out / "student" / wav, syn(r["text"], lang), melmod.SR)
            manifest.append({"index": i, "wav": wav, "text": r["text"], "ok": ok})
        for lane in ("student", "teacher"):
            (out / lane / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n")
        print(f"{lang}: {sum(m['ok'] for m in manifest)} sentences -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
