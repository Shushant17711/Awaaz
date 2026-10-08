"""Multilingual acoustic model: unified Indic tokens + language id -> 80-bin log-mel.

    tokens --embed(+lang)--> encoder (depthwise-separable conv blocks)
           --> per-token prior mean mu (80) ............ aligned to mel by MAS (training only)
           --> duration predictor (log frames) ......... trained on MAS durations
           --> expand by durations --> mel decoder (conv blocks, language FiLM) --> mel

Alignment is Monotonic Alignment Search (Glow-TTS) between the per-token prior
means and the target mel, using Piper's compiled maximum_path. No phonemizer
and no teacher durations: the model learns where each grapheme sits in time.
All ops are conv1d / depthwise conv / linear / layernorm, so the NumPy runtime
can run it later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConvBlock(nn.Module):
    """Residual: depthwise conv -> pointwise conv -> GELU -> LayerNorm, x2, with optional FiLM."""

    def __init__(self, dim: int, kernel: int = 5, dropout: float = 0.1, film_dim: int = 0) -> None:
        super().__init__()
        self.dw1 = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw1 = nn.Conv1d(dim, dim, 1)
        self.dw2 = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(film_dim, 2 * dim) if film_dim else None

    def _norm(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self._norm(self.n1, F.gelu(self.pw1(self.dw1(x * mask))))
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self._norm(self.n2, F.gelu(self.pw2(self.dw2(self.drop(h) * mask))))
        return (x + self.drop(h)) * mask


class MultilingualAcoustic(nn.Module):
    def __init__(self, n_tokens: int, n_langs: int, dim: int = 192, enc_blocks: int = 6, dec_blocks: int = 6,
                 dur_dim: int = 128, n_mels: int = 80, lang_dim: int = 64) -> None:
        super().__init__()
        self.config = dict(n_tokens=n_tokens, n_langs=n_langs, dim=dim, enc_blocks=enc_blocks,
                           dec_blocks=dec_blocks, dur_dim=dur_dim, n_mels=n_mels, lang_dim=lang_dim)
        self.tok = nn.Embedding(n_tokens, dim)
        self.lang = nn.Embedding(n_langs, lang_dim)
        self.lang_in = nn.Linear(lang_dim, dim)
        self.encoder = nn.ModuleList([SepConvBlock(dim, film_dim=lang_dim) for _ in range(enc_blocks)])
        self.mu = nn.Conv1d(dim, n_mels, 1)
        self.dur_in = nn.Conv1d(dim, dur_dim, 1)
        self.dur_blocks = nn.ModuleList([SepConvBlock(dur_dim, kernel=3, film_dim=lang_dim) for _ in range(2)])
        self.dur_out = nn.Conv1d(dur_dim, 1, 1)
        self.dec_in = nn.Conv1d(dim + n_mels, dim, 1)
        self.decoder = nn.ModuleList([SepConvBlock(dim, film_dim=lang_dim) for _ in range(dec_blocks)])
        self.mel_out = nn.Conv1d(dim, n_mels, 1)

    # -- pieces ---------------------------------------------------------------
    def encode(self, tokens: torch.Tensor, tok_mask: torch.Tensor, lang: torch.Tensor):
        lang_e = self.lang(lang)
        x = (self.tok(tokens) + self.lang_in(lang_e).unsqueeze(1)).transpose(1, 2)
        m = tok_mask.unsqueeze(1).float()
        for block in self.encoder:
            x = block(x, m, lang_e)
        return x, self.mu(x) * m, lang_e

    def predict_log_dur(self, h: torch.Tensor, tok_mask: torch.Tensor, lang_e: torch.Tensor) -> torch.Tensor:
        m = tok_mask.unsqueeze(1).float()
        d = self.dur_in(h.detach()) * m
        for block in self.dur_blocks:
            d = block(d, m, lang_e)
        return self.dur_out(d).squeeze(1) * tok_mask.float()        # log(1 + frames)

    def decode(self, h_frames: torch.Tensor, mu_frames: torch.Tensor, frame_mask: torch.Tensor,
               lang_e: torch.Tensor) -> torch.Tensor:
        m = frame_mask.unsqueeze(1).float()
        x = self.dec_in(torch.cat([h_frames, mu_frames], dim=1)) * m
        for block in self.decoder:
            x = block(x, m, lang_e)
        return (self.mel_out(x) + mu_frames) * m                      # residual on the prior

    @staticmethod
    def expand(x: torch.Tensor, durations: torch.Tensor, frames: int) -> torch.Tensor:
        """x [B, C, T_tok], durations [B, T_tok] (int) -> [B, C, frames] by repetition."""
        out = x.new_zeros(x.shape[0], x.shape[1], frames)
        for b in range(x.shape[0]):
            rep = torch.repeat_interleave(x[b], durations[b].clamp(min=0), dim=1)[:, :frames]
            out[b, :, : rep.shape[1]] = rep
        return out

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, lang: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
        tok_mask = torch.ones_like(tokens, dtype=torch.bool)
        h, mu, lang_e = self.encode(tokens, tok_mask, lang)
        log_d = self.predict_log_dur(h, tok_mask, lang_e)
        dur = torch.clamp(torch.round((torch.exp(log_d) - 1) * length_scale), min=1).long()
        frames = int(dur.sum(1).max())
        frame_mask = torch.arange(frames, device=tokens.device).unsqueeze(0) < dur.sum(1, keepdim=True)
        return self.decode(self.expand(h, dur, frames), self.expand(mu, dur, frames), frame_mask, lang_e)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
