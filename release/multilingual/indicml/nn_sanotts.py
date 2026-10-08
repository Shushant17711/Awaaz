"""NumPy ops and the piperlite decoder forward pass, vendored from sanoTTS.

Source: github.com/Ampixa/sanoTTS, pypkg/sanotts/models.py (MIT License,
Copyright (c) 2026 Ampixa; see LICENSE.sanotts in this directory). Copied
unchanged except for this header; used here as the vocoder (mel -> waveform).
"""

from __future__ import annotations

from typing import Any

import numpy as np

def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def leaky_relu(x: np.ndarray, slope: float) -> np.ndarray:
    return np.where(x > 0.0, x, slope * x)


def linspace01(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float64).astype(np.float32)


def conv1d_same(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dilation: int = 1,
) -> np.ndarray:
    """PyTorch Conv1d "same"-padding semantics: pad = dilation*(K//2).

    x: [in_ch, T]; weight: [out_ch, in_ch, K]; bias: [out_ch].
    Returns [out_ch, T].
    """
    in_ch, T = x.shape
    out_ch, in_ch_w, K = weight.shape
    if in_ch_w != in_ch:
        raise ValueError(f"conv1d_same: channel mismatch {in_ch_w} != {in_ch}")
    pad = dilation * (K // 2)
    out = np.broadcast_to(bias[:, None].astype(np.float32), (out_ch, T)).copy()
    for k in range(K):
        off = k * dilation - pad
        lo = max(0, -off)
        hi = min(T, T - off)
        if hi <= lo:
            continue
        # out[:, lo:hi] += weight[:, :, k] @ x[:, lo+off:hi+off]
        out[:, lo:hi] += weight[:, :, k] @ x[:, lo + off:hi + off]
    return out


def conv1d_1x1(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Conv1d with kernel_size=1: a plain per-timestep linear projection."""
    w = weight[:, :, 0] if weight.ndim == 3 else weight
    return w @ x + bias[:, None]


def depthwise_conv1d_same(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Depthwise Conv1d, no bias. x: [C, T]; weight: [C, K] (or [C,1,K])."""
    if weight.ndim == 3:
        weight = weight[:, 0, :]
    C, T = x.shape
    _, K = weight.shape
    pad = K // 2
    out = np.zeros_like(x)
    for k in range(K):
        off = k - pad
        lo = max(0, -off)
        hi = min(T, T - off)
        if hi <= lo:
            continue
        out[:, lo:hi] += weight[:, k:k + 1] * x[:, lo + off:hi + off]
    return out


def conv_transpose1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> np.ndarray:
    """PyTorch ConvTranspose1d. x: [in_ch, T]; weight: [in_ch, out_ch, K]; bias: [out_ch].

    out[oc, t*stride + k - padding] += weight[ic, oc, k] * x[ic, t], summed over ic, k;
    output length L = (T - 1) * stride - 2 * padding + K.
    """
    in_ch, T = x.shape
    in_ch_w, out_ch, K = weight.shape
    if in_ch_w != in_ch:
        raise ValueError(f"conv_transpose1d: channel mismatch {in_ch_w} != {in_ch}")
    L = (T - 1) * stride - 2 * padding + K
    out = np.broadcast_to(bias[:, None].astype(np.float32), (out_ch, L)).copy()
    for k in range(K):
        shift = k - padding
        if shift >= L:
            continue
        t_lo = 0
        if shift < 0:
            t_lo = (-shift + stride - 1) // stride
        t_hi = (L - 1 - shift) // stride + 1
        t_hi = min(t_hi, T)
        if t_hi <= t_lo:
            continue
        count = t_hi - t_lo
        j_start = t_lo * stride + shift
        j_end = j_start + (count - 1) * stride + 1
        contrib = weight[:, :, k].T @ x[:, t_lo:t_hi]  # [out_ch, count]
        out[:, j_start:j_end:stride] += contrib
    return out




_BANK_KERNELS = (3, 5, 7)
_BANK_DIL1 = (1, 2, 3)
_BANK_DIL2 = (2, 6, 12)


def _residual_bank(x: np.ndarray, tensors: dict[str, np.ndarray], prefix: str, branches: list[int]) -> np.ndarray:
    """PiperResidualBank: mean over active branches of
    y2 = conv2(lrelu(y1, 0.1)) + y1, y1 = conv1(lrelu(x, 0.1)) + x.
    """
    acc = np.zeros_like(x)
    for branch in branches:
        k = _BANK_KERNELS[branch]
        d1 = _BANK_DIL1[branch]
        d2 = _BANK_DIL2[branch]
        t = leaky_relu(x, 0.1)
        u = conv1d_same(t, tensors[f"{prefix}.blocks.{branch}.conv1.weight"], tensors[f"{prefix}.blocks.{branch}.conv1.bias"], dilation=d1)
        y1 = u + x
        t2 = leaky_relu(y1, 0.1)
        u2 = conv1d_same(t2, tensors[f"{prefix}.blocks.{branch}.conv2.weight"], tensors[f"{prefix}.blocks.{branch}.conv2.bias"], dilation=d2)
        y2 = u2 + y1
        acc = acc + y2
    return acc / float(len(branches))


def _apply_post_filter(audio: np.ndarray, tensors: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    channels = int(config.get("post_filter_channels") or 0)
    layers = int(config.get("post_filter_layers") or 0)
    kernel = int(config.get("post_filter_kernel") or 9)
    scale = float(config.get("post_filter_scale") or 0.0)
    if channels <= 0:
        return audio

    r = conv1d_same(audio[None, :], tensors["post_filter.in_conv.weight"], tensors["post_filter.in_conv.bias"])
    for layer in range(layers):
        unit_scale = float(tensors[f"post_filter.units.{layer}.scale"][0])
        t = leaky_relu(r, 0.1)
        u = conv1d_same(t, tensors[f"post_filter.units.{layer}.conv1.weight"], tensors[f"post_filter.units.{layer}.conv1.bias"], dilation=1 + layer)
        t = leaky_relu(u, 0.1)
        u2 = conv1d_same(t, tensors[f"post_filter.units.{layer}.conv2.weight"], tensors[f"post_filter.units.{layer}.conv2.bias"], dilation=1)
        r = r + unit_scale * u2
    out = conv1d_same(r, tensors["post_filter.out_conv.weight"], tensors["post_filter.out_conv.bias"])[0]
    return np.tanh(audio + scale * out).astype(np.float32)


def decoder_forward(tensors: dict[str, np.ndarray], config: dict[str, Any], latent: np.ndarray) -> np.ndarray:
    variant = str(config.get("variant") or "")
    if variant != "piperlite":
        raise NotImplementedError(f"unsupported decoder variant: {variant!r}")
    if str(config.get("activation") or "leaky_relu") != "leaky_relu":
        raise NotImplementedError("only activation='leaky_relu' decoders are implemented")
    if float(config.get("pre_tanh_repair_channels") or 0) > 0:
        raise NotImplementedError("pre_tanh_repair is not implemented (no shipped voice uses it)")
    res_layers = int(config.get("res_layers") or 1)
    if res_layers != 1:
        raise NotImplementedError(f"only res_layers=1 is implemented, got {res_layers}")

    channels = config["channels"]
    c0, c1, c2, c3 = (int(c) for c in channels[:4])

    x = conv1d_same(latent, tensors["pre.weight"], tensors["pre.bias"])  # [c0, frames]

    stage_specs = [
        (c0, c1, 16, 8, 4, "up0", "res0.0", config.get("stage0_branches")),
        (c1, c2, 16, 8, 4, "up1", "res1.0", config.get("stage1_branches")),
        (c2, c3, 8, 4, 2, "up2", "res2.0", config.get("stage2_branches")),
    ]
    for in_c, out_c, up_k, up_s, up_p, up_name, bank_prefix, branches in stage_specs:
        if branches is None:
            branches = [0, 1, 2]
        x = leaky_relu(x, 0.1)
        x = conv_transpose1d(x, tensors[f"{up_name}.weight"], tensors[f"{up_name}.bias"], stride=up_s, padding=up_p)
        x = _residual_bank(x, tensors, bank_prefix, list(branches))

    x = leaky_relu(x, 0.01)
    audio = conv1d_same(x, tensors["post.weight"], tensors["post.bias"])[0]
    audio = np.tanh(audio).astype(np.float32)
    audio = _apply_post_filter(audio, tensors, config)
    return audio
