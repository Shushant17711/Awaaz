"""Zero-parameter anti-imaging filters for the piperlite vocoder's transposed-conv upsamplers.

A transposed conv is linear and time-invariant on the zero-stuffed input, so everything it emits
above the *input* Nyquist is a spectral image of lower content. Those images are the fixed metallic
tones at multiples of 86 Hz (frame rate) and 689 Hz. A fixed Kaiser-sinc low-pass after the
upsampler removes them without touching legitimate content. No learned weights; the state dict is
unchanged, the config gains "aa_up": {"up0": 8, "up1": 8}.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def kaiser_lowpass(stride: int, taps_per_side: int = 6, beta: float = 8.0, cutoff_scale: float = 1.0) -> np.ndarray:
    n = 2 * taps_per_side * stride + 1
    t = np.arange(n) - (n - 1) / 2
    fc = cutoff_scale * 0.5 / stride                    # cycles/sample at the upsampled rate
    h = 2 * fc * np.sinc(2 * fc * t) * np.kaiser(n, beta)
    return (h / h.sum()).astype(np.float32)

def attach(vocoder: torch.nn.Module, spec: dict[str, int], cutoff_scale: float = 1.0) -> list:
    handles = []
    for name, stride in spec.items():
        h = torch.from_numpy(kaiser_lowpass(stride, cutoff_scale=cutoff_scale))
        def hook(mod, inp, out, h=h):
            c = out.shape[1]
            w = h.to(out.device, out.dtype).view(1, 1, -1).expand(c, 1, -1)
            return F.conv1d(out, w, padding=w.shape[-1] // 2, groups=c)
        handles.append(getattr(vocoder, name).register_forward_hook(hook))
    return handles
