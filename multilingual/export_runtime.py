#!/usr/bin/env python3
"""Export acoustic.pt + vocoder.pt as an indicml voice package (fp16 blob + manifest).

    python export_runtime.py --acoustic runs/acoustic/acoustic.pt --vocoder runs/vocoder/vocoder.pt \
        --out ../release/multilingual
Copies the runtime code next to the weights so the directory is self-contained:
    <out>/manifest.json, <out>/weights.fp16.bin, <out>/indicml/   (python -m indicml <out> hi_IN "...")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frontend"))
import indic  # noqa: E402


def tensors_of(state: dict, blob: bytearray, int8: bool = False) -> tuple[list[dict], int]:
    table, params = [], 0
    for name in sorted(state):
        v = state[name].detach().cpu()
        if not v.is_floating_point():
            continue
        if int8 and v.ndim >= 2 and v.numel() >= 1024:
            w = v.float().numpy().reshape(v.shape[0], -1)
            scale = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
            q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
            sb = scale.astype(np.float16).tobytes()
            qb = q.tobytes()
            table.append({"name": name, "shape": list(v.shape), "dtype": "int8", "offset_bytes": len(blob),
                          "nbytes": len(qb), "scale_offset_bytes": len(blob) + len(qb), "scale_nbytes": len(sb),
                          "sha256": hashlib.sha256(qb + sb).hexdigest()})
            blob += qb + sb
            params += v.numel()
            continue
        a = (v.reshape(1) if v.ndim == 0 else v).numpy().astype(np.float16)
        if not np.isfinite(a).all():
            raise RuntimeError(f"{name}: not finite in fp16")
        b = a.tobytes()
        table.append({"name": name, "shape": list(a.shape), "dtype": "float16", "offset_bytes": len(blob),
                      "nbytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
        blob += b
        params += a.size
    return table, params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acoustic", type=Path, default=HERE / "runs/acoustic/acoustic.pt")
    ap.add_argument("--vocoder", type=Path, default=HERE / "runs/vocoder/vocoder.pt")
    ap.add_argument("--out", type=Path, default=HERE.parent / "release" / "multilingual")
    ap.add_argument("--int8", action="store_true", help="int8 weights (per-output-channel scales): ~half the size")
    ap.add_argument("--fp16-vocoder", action="store_true", help="with --int8: keep the vocoder fp16 (int8 vocoder measured: waveform corr 0.9993 vs fp16)")
    args = ap.parse_args()

    ac = torch.load(args.acoustic, map_location="cpu", weights_only=False)
    vo = torch.load(args.vocoder, map_location="cpu", weights_only=False)
    if ac.get("vocab") != indic.VOCAB:
        raise SystemExit("acoustic checkpoint vocabulary differs from the current front end")
    blob = bytearray()
    ac_t, ac_n = tensors_of(ac["model_state_dict"], blob, args.int8)
    vo_t, vo_n = tensors_of(vo["model_state_dict"], blob, args.int8 and not args.fp16_vocoder)
    args.out.mkdir(parents=True, exist_ok=True)
    wname = "weights.int8.bin" if args.int8 else "weights.fp16.bin"
    (args.out / wname).write_bytes(bytes(blob))
    manifest = {
        "format": "indicml.fp16.v1", "sample_rate": 22050, "hop_length": 256,
        "weights_file": wname, "weights_size_bytes": len(blob),
        "weights_precision": ("int8 acoustic" + (" + fp16 vocoder" if args.fp16_vocoder else " + int8 vocoder")) if args.int8 else "fp16",
        "weights_sha256": hashlib.sha256(bytes(blob)).hexdigest(),
        "total_parameters": ac_n + vo_n,
        "components": {
            "acoustic": {"config": ac["config"], "parameters": ac_n, "tensors": ac_t,
                         "source": str(args.acoustic), "steps": ac.get("steps")},
            "vocoder": {"config": vo["config"], "parameters": vo_n, "tensors": vo_t,
                        "source": str(args.vocoder), "steps": vo.get("steps")},
        },
        "frontend": {"type": "unified-indic-graphemes", "vocab": indic.VOCAB, "langs": indic.LANGS},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    code = args.out / "indicml"
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(HERE / "runtime" / "indicml", code, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(HERE / "frontend" / "indic.py", code / "frontend.py")      # always ship the current front end
    print(json.dumps({"out": str(args.out), "parameters": ac_n + vo_n, "weights_bytes": len(blob), "int8": args.int8}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
