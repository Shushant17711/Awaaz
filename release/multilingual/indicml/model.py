"""NumPy runtime for the multilingual Indic TTS: text -> 22.05 kHz waveform.

No PyTorch, no eSpeak. A voice package is a directory with
    manifest.json      format "indicml.fp16.v1": per-component config + tensor table
    weights.fp16.bin   all tensors, fp16, addressed by manifest offsets
Mirrors multilingual/acoustic.py (MultilingualAcoustic.infer) and the
piperlite decoder (vendored, nn_sanotts.decoder_forward). Parity with PyTorch
is checked by multilingual/test_runtime.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import frontend
from .nn_sanotts import conv1d_1x1, conv1d_same, decoder_forward

FORMAT = "indicml.fp16.v1"


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): exact-erf GELU as in torch's default.
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return sign * y


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over channels at each timestep. x: [C, T]."""
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w[:, None] + b[:, None]


def depthwise(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Depthwise Conv1d with bias, same padding. x: [C, T]; w: [C, 1, K]."""
    w = w[:, 0, :]
    C, T = x.shape
    K = w.shape[1]
    pad = K // 2
    out = np.broadcast_to(b[:, None], (C, T)).astype(np.float32).copy()
    for k in range(K):
        off = k - pad
        lo, hi = max(0, -off), min(T, T - off)
        if hi > lo:
            out[:, lo:hi] += w[:, k:k + 1] * x[:, lo + off:hi + off]
    return out


def _load_tensor(blob: bytes, x: dict) -> np.ndarray:
    raw = blob[x["offset_bytes"]:x["offset_bytes"] + x["nbytes"]]
    if x["dtype"] == "int8":
        q = np.frombuffer(raw, dtype=np.int8).astype(np.float32).reshape(x["shape"][0], -1)
        s = np.frombuffer(blob[x["scale_offset_bytes"]:x["scale_offset_bytes"] + x["scale_nbytes"]], dtype="<f2")
        return (q * s.astype(np.float32)[:, None]).reshape(x["shape"])
    return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(x["shape"])


class Voice:
    def __init__(self, package_dir: str | Path) -> None:
        d = Path(package_dir)
        self.manifest: dict[str, Any] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{d}: not an {FORMAT} package")
        blob = (d / self.manifest["weights_file"]).read_bytes()
        self.t: dict[str, dict[str, np.ndarray]] = {}
        for comp, spec in self.manifest["components"].items():
            self.t[comp] = {x["name"]: _load_tensor(blob, x) for x in spec["tensors"]}
        self.ac_cfg = self.manifest["components"]["acoustic"]["config"]
        self.voc_cfg = self.manifest["components"]["vocoder"]["config"]
        if self.manifest["frontend"]["vocab"] != frontend.VOCAB:
            raise ValueError("package vocabulary does not match this runtime's front end")
        self.sample_rate = int(self.manifest["sample_rate"])
        self.mel_fmax = float((self.voc_cfg.get("mel") or {}).get("fmax", 8000.0))
        self.languages = list(self.manifest["frontend"]["langs"])

    # -- acoustic ---------------------------------------------------------------
    def _lin(self, name: str, x: np.ndarray) -> np.ndarray:
        a = self.t["acoustic"]
        return a[f"{name}.weight"] @ x + a[f"{name}.bias"]

    def _block(self, prefix: str, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
        a = self.t["acoustic"]
        h = depthwise(x, a[f"{prefix}.dw1.weight"], a[f"{prefix}.dw1.bias"])
        h = layer_norm(gelu(conv1d_1x1(h, a[f"{prefix}.pw1.weight"], a[f"{prefix}.pw1.bias"])),
                       a[f"{prefix}.n1.weight"], a[f"{prefix}.n1.bias"])
        if f"{prefix}.film.weight" in a:
            scale, shift = np.split(self._lin(f"{prefix}.film", cond), 2)
            h = h * (1.0 + scale[:, None]) + shift[:, None]
        h = depthwise(h, a[f"{prefix}.dw2.weight"], a[f"{prefix}.dw2.bias"])
        h = layer_norm(gelu(conv1d_1x1(h, a[f"{prefix}.pw2.weight"], a[f"{prefix}.pw2.bias"])),
                       a[f"{prefix}.n2.weight"], a[f"{prefix}.n2.bias"])
        return x + h

    def mel(self, text: str, lang: str, length_scale: float = 1.0) -> np.ndarray:
        if lang not in frontend.LANG_ID:
            raise ValueError(f"unknown language {lang!r}; known: {frontend.LANGS}")
        a, cfg = self.t["acoustic"], self.ac_cfg
        ids = np.asarray(frontend.encode(text), dtype=np.int64)
        lang_e = a["lang.weight"][frontend.LANG_ID[lang]]
        x = (a["tok.weight"][ids] + self._lin("lang_in", lang_e)).T                 # [dim, T]
        for i in range(cfg["enc_blocks"]):
            x = self._block(f"encoder.{i}", x, lang_e)
        mu = conv1d_1x1(x, a["mu.weight"], a["mu.bias"])
        d = conv1d_1x1(x, a["dur_in.weight"], a["dur_in.bias"])
        for i in range(2):
            d = self._block(f"dur_blocks.{i}", d, lang_e)
        log_d = conv1d_1x1(d, a["dur_out.weight"], a["dur_out.bias"])[0]
        dur = np.maximum(np.round((np.exp(log_d) - 1.0) * length_scale), 1).astype(np.int64)
        h_f, mu_f = np.repeat(x, dur, axis=1), np.repeat(mu, dur, axis=1)
        y = conv1d_1x1(np.concatenate([h_f, mu_f], axis=0), a["dec_in.weight"], a["dec_in.bias"])
        for i in range(cfg["dec_blocks"]):
            y = self._block(f"decoder.{i}", y, lang_e)
        return conv1d_1x1(y, a["mel_out.weight"], a["mel_out.bias"]) + mu_f      # [80, frames]

    # -- full -------------------------------------------------------------------
    def synthesize(self, text: str, lang: str, length_scale: float = 1.0, *, normalize: bool = True,
                   target_dbfs: float = -16.0, question_rise: bool = False) -> np.ndarray:
        """Text -> float32 waveform in [-1, 1] at self.sample_rate.

        normalize: scale speech to target_dbfs RMS (measured over non-silent 20 ms frames) with
        a -1 dBFS peak ceiling. The raw model output is ~5 dB quieter than its teachers.
        """
        mel = self.mel(text, lang, length_scale)
        if question_rise and text.rstrip().endswith(("?", "؟")):
            mel = question_contour(mel, self.mel_fmax)
        audio = np.clip(decoder_forward(self.t["vocoder"], self.voc_cfg, mel), -1, 1)
        return loudness_normalize(audio, target_dbfs) if normalize else audio.astype(np.float32)


def _mel_scale(f: np.ndarray) -> np.ndarray:            # Slaney mel scale (librosa default)
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3)
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0), lin)


def question_contour(mel: np.ndarray, fmax: float, rise: float = 1.22, tail_s: float = 0.45) -> np.ndarray:
    """EXPERIMENTAL, off by default (measured ineffective: 80-bin mels are too coarse to move pitch).

    Rising terminal pitch for questions, as a zero-parameter mel-domain warp.

    The teachers (and so the student) say questions with the same falling cadence as
    statements. Over the last tail_s seconds this stretches the spectrum upward
    (harmonics move up by a factor growing smoothly from 1 to `rise`), which the
    vocoder renders as a rising pitch contour. Formants shift by the same small factor.
    """
    n_mels, frames = mel.shape
    tail = min(frames, int(tail_s * 22050 / 256))
    if tail < 4:
        return mel
    edges = _mel_scale(np.array([0.0, fmax]))
    centers_mel = np.linspace(edges[0], edges[1], n_mels + 2)[1:-1]          # bin centres in mel units
    def to_hz(m):
        return np.where(m >= 15.0, 1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0), m * 200.0 / 3)
    centers_hz = to_hz(centers_mel)
    out = mel.copy()
    for j in range(tail):
        t = (j + 1) / tail
        k = 1.0 + (rise - 1.0) * (t * t * (3 - 2 * t))                        # smoothstep ramp
        src = np.interp(_mel_scale(centers_hz / k), centers_mel, np.arange(n_mels))   # read from lower freq
        col = mel[:, frames - tail + j]
        out[:, frames - tail + j] = np.interp(src, np.arange(n_mels), col)
    return out


def loudness_normalize(audio: np.ndarray, target_dbfs: float = -16.0, peak_dbfs: float = -1.0) -> np.ndarray:
    frame = 441                                                   # 20 ms at 22.05 kHz
    n = len(audio) // frame
    if n == 0:
        return audio.astype(np.float32)
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(1))
    speech = rms[rms > rms.max() * 10 ** (-40 / 20)]              # ignore pauses (>40 dB below the loudest frame)
    level = np.sqrt((speech ** 2).mean()) if speech.size else 1e-9
    y = audio * (10 ** (target_dbfs / 20) / max(level, 1e-9))
    # Soft-knee limiter: samples above the knee are compressed smoothly toward the ceiling
    # instead of the whole signal being turned down to fit its loudest peak.
    ceil = 10 ** (peak_dbfs / 20)
    knee = 0.6 * ceil
    over = np.abs(y) > knee
    y[over] = np.sign(y[over]) * (knee + (ceil - knee) * np.tanh((np.abs(y[over]) - knee) / (ceil - knee)))
    return y.astype(np.float32)


__all__ = ["Voice", "FORMAT", "conv1d_same"]
