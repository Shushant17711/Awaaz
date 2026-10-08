"""The one mel definition shared by the vocoder and the acoustic model.

22.05 kHz, 80 log-mel bins, n_fft/win 1024, hop 256 (= the decoder's 256x
upsampling, so one mel frame -> 256 samples), fmin 0, fmax 8000. Padding is
chosen so frames == samples // 256 exactly, which lets a waveform segment and
its mel segment be cut at the same frame boundary.
"""

from __future__ import annotations

import functools
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F

SR, N_FFT, HOP, WIN, N_MELS, FMIN = 22050, 1024, 256, 1024, 80, 0.0
FMAX = float(os.environ.get("MEL_FMAX", "8000"))   # v2 pipeline uses 11025 (full band: sibilants)
LOG_FLOOR = 1e-5


@functools.lru_cache(maxsize=4)
def _basis(device: str) -> torch.Tensor:
    mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return torch.from_numpy(mel.astype(np.float32)).to(device)


def mel_from_audio(audio: torch.Tensor) -> torch.Tensor:
    """audio [B, T] or [B, 1, T] float in [-1, 1] -> log-mel [B, 80, T // 256]."""
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    frames = audio.shape[-1] // HOP
    audio = audio[..., : frames * HOP]
    pad = (N_FFT - HOP) // 2
    audio = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(audio, N_FFT, hop_length=HOP, win_length=WIN,
                      window=torch.hann_window(WIN, device=audio.device), center=False, return_complex=True)
    mag = spec.abs()
    mel = torch.matmul(_basis(str(audio.device)), mag)
    return torch.log(torch.clamp(mel, min=LOG_FLOOR))[..., :frames]
