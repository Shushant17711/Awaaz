#!/usr/bin/env python3
"""Parity: NumPy runtime (runtime/indicml) vs the PyTorch models.

    python test_runtime.py [--acoustic A.pt --vocoder V.pt]
Without checkpoints, trains throwaway ones for a few CPU steps so the weights
are non-trivial. Both sides use the same fp16-rounded weights, so any gap is
the runtime's arithmetic, not quantization. Gates: identical durations,
mel max-abs < 1e-2, waveform correlation > 0.999.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "frontend"))
sys.path.insert(0, str(HERE / "runtime"))
import indic  # noqa: E402
from synth import Synthesizer  # noqa: E402
from indicml import Voice  # noqa: E402

SAMPLES = {"hi_IN": "नमस्ते, आज मौसम अच्छा है।", "ta_IN": "தமிழ் ஒரு அழகான மொழி.",
           "bn_BD": "আমি বাংলায় কথা বলি।", "ur_PK": "یہ ایک امتحان ہے؟", "as_IN": "অসমীয়া ভাষা।"}


def round_fp16(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in list(module.parameters()) + list(module.buffers()):
            if p.is_floating_point():
                p.copy_(p.half().float())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", type=Path); ap.add_argument("--vocoder", type=Path)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="indicml-parity-"))
    if args.acoustic is None:
        langs = ",".join(sorted(p.parent.name for p in (HERE / "data/wav").glob("*/manifest.jsonl"))[:1])
        subprocess.run([PY, str(HERE / "train_vocoder.py"), "--device", "cpu", "--steps", "3", "--batch", "2",
                        "--adv-start", "99", "--out-dir", str(tmp / "voc"), "--langs", langs], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([PY, str(HERE / "train_acoustic.py"), "--device", "cpu", "--steps", "30", "--batch", "2",
                        "--warmup", "1", "--out-dir", str(tmp / "ac"), "--langs", langs], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args.acoustic, args.vocoder = tmp / "ac/acoustic.pt", tmp / "voc/vocoder.pt"
    subprocess.run([PY, str(HERE / "export_runtime.py"), "--acoustic", str(args.acoustic), "--vocoder",
                    str(args.vocoder), "--out", str(tmp / "pkg")], check=True, stdout=subprocess.DEVNULL)

    syn = Synthesizer(args.acoustic, args.vocoder, "cpu")
    round_fp16(syn.acoustic); round_fp16(syn.vocoder)
    voice = Voice(tmp / "pkg")
    ok = True
    for lang, text in SAMPLES.items():
        ids = torch.tensor([indic.encode(text)])
        with torch.no_grad():
            mel_t = syn.acoustic.infer(ids, torch.tensor([indic.LANG_ID[lang]]))[0].numpy()
            wav_t = syn.vocoder(torch.from_numpy(mel_t)[None])
            wav_t = (wav_t[0] if isinstance(wav_t, tuple) else wav_t).reshape(-1).clamp(-1, 1).numpy()
        mel_n = voice.mel(text, lang)
        wav_n = voice.synthesize(text, lang, normalize=False)   # parity is on the raw model output; normalization is separate
        same_len = mel_t.shape == mel_n.shape
        mel_err = float(np.abs(mel_t - mel_n).max()) if same_len else float("inf")
        n = min(len(wav_t), len(wav_n))
        corr = float(np.corrcoef(wav_t[:n], wav_n[:n])[0, 1]) if n > 1 and wav_t[:n].std() > 0 else float("nan")
        passed = same_len and mel_err < 1e-2 and corr > 0.999
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {lang}: frames {mel_t.shape[1]} vs {mel_n.shape[1]}, "
              f"mel max|Δ| {mel_err:.2e}, wav corr {corr:.6f}")
    pkg_bytes = sum(f.stat().st_size for f in (tmp / "pkg").glob("weights.fp16.bin"))
    print(f"package weights: {pkg_bytes / 1e6:.2f} MB, parameters: {voice.manifest['total_parameters']:,}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
