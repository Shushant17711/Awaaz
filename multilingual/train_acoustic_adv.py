#!/usr/bin/env python3
"""De-smoothing fine-tune of the acoustic model (fixes the robotic / buzzy sound).

The acoustic model was trained with L1/L2 losses only, which average over the plausible
spectrograms and yield over-smoothed mels: on held-out speech its frame-to-frame detail
is ~20% below real speech (0.45 vs 0.56-0.59 in the low/mid bands). A vocoder fed such
mels sounds robotic. The vocoder itself is fine: on real teacher mels it matches the
teacher within ~1 dB.

This continues training the acoustic model, teacher-forced on the MAS alignment, with
its usual losses plus an LSGAN mel-patch discriminator (2-D convs over 80 bins x 64
frames) and feature matching. The discriminator is training-only and is never shipped.
Writes <out>/acoustic.pt (same format as phase 4). Resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from piper.train.vits.monotonic_align import maximum_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "frontend")); sys.path.insert(0, str(ROOT / "sanoTTS" / "tools"))
import indic  # noqa: E402
import mel as melmod  # noqa: E402
from acoustic import MultilingualAcoustic  # noqa: E402
from resume_state import add_resume_args, make_resumer  # noqa: E402
from train_acoustic import DATA, Utterances, collate  # noqa: E402

CROP = 64


class MelPatchDiscriminator(nn.Module):
    """2-D conv critic on [B, 1, 80, CROP] log-mel patches; returns score map + feature maps."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        spec = [(1, ch, (3, 9), (1, 1)), (ch, ch, (3, 9), (2, 2)), (ch, 2 * ch, (3, 9), (2, 2)),
                (2 * ch, 2 * ch, (3, 5), (2, 1)), (2 * ch, 4 * ch, (3, 3), (1, 1))]
        self.convs = nn.ModuleList([nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(i, o, k, s, padding=(k[0] // 2, k[1] // 2))) for i, o, k, s in spec])
        self.out = nn.Conv2d(4 * ch, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            feats.append(x)
        return self.out(x), feats


def crops(mel: torch.Tensor, n_frames: torch.Tensor, starts: list[int]) -> torch.Tensor:
    return torch.stack([mel[b, :, s:s + CROP] for b, s in enumerate(starts)]).unsqueeze(1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acoustic", type=Path, default=HERE / "runs/acoustic/acoustic.pt")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/acoustic-adv")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--fm-weight", type=float, default=0.5)
    ap.add_argument("--adv-ramp", type=int, default=1000, help="steps to ramp adversarial weights from 0")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    add_resume_args(ap)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    langs = sorted(p.parent.name for p in (DATA / "wav").glob("*/manifest.jsonl"))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.acoustic, map_location="cpu", weights_only=False)
    model = MultilingualAcoustic(**ck["config"]); model.load_state_dict(ck["model_state_dict"]); model.to(device)
    disc = MelPatchDiscriminator().to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99))
    resumer = make_resumer(args, args.out_dir)
    for name, obj in (("model", model), ("disc", disc), ("opt_g", opt_g), ("opt_d", opt_d)):
        resumer.register(name, obj)
    start = resumer.load() + 1
    resumer.install_signal_handlers()
    loader = iter(torch.utils.data.DataLoader(Utterances(langs, args.max_frames, seed=start), batch_size=args.batch,
                                              num_workers=4, collate_fn=collate, persistent_workers=True))
    print(json.dumps({"start": start, "steps": args.steps, "disc_params": sum(p.numel() for p in disc.parameters())}), flush=True)

    for step in range(start, args.steps + 1):
        audio, n_frames, tokens, n_tok, lang = (x.to(device) for x in next(loader))
        with torch.no_grad():
            target = melmod.mel_from_audio(audio)
        F_max = target.shape[-1]
        frame_mask = torch.arange(F_max, device=device).unsqueeze(0) < n_frames.unsqueeze(1)
        tok_mask = torch.arange(tokens.shape[1], device=device).unsqueeze(0) < n_tok.unsqueeze(1)
        target = target * frame_mask.unsqueeze(1)
        h, mu, lang_e = model.encode(tokens, tok_mask, lang)
        with torch.no_grad():
            neg = -0.5 * ((target.unsqueeze(2) - mu.unsqueeze(3)) ** 2).sum(1)
            attn = (tok_mask.unsqueeze(2) & frame_mask.unsqueeze(1)).transpose(1, 2).float()
            path = maximum_path(neg.transpose(1, 2).contiguous(), attn)
            dur = path.sum(1).long()
        mu_f = torch.bmm(mu, path.transpose(1, 2))
        mel_hat = model.decode(torch.bmm(h, path.transpose(1, 2)), mu_f, frame_mask, lang_e)
        valid = frame_mask.unsqueeze(1).float()
        denom = valid.sum() * target.shape[1]
        base = ((mel_hat - target).abs() * valid).sum() / denom + (((mu_f - target) ** 2) * valid).sum() / denom
        log_d = model.predict_log_dur(h, tok_mask, lang_e)
        base = base + (((log_d - torch.log1p(dur.float())) ** 2) * tok_mask).sum() / tok_mask.sum()

        keep = [b for b in range(audio.shape[0]) if int(n_frames[b]) > CROP]
        starts = [random.randint(0, int(n_frames[b]) - CROP) for b in keep]
        real, fake = crops(target[keep], n_frames[keep], starts), crops(mel_hat[keep], n_frames[keep], starts)
        # discriminator step
        for p in disc.parameters():
            p.requires_grad_(True)
        d_real, _ = disc(real)
        d_fake, _ = disc(fake.detach())
        d_loss = ((d_real - 1) ** 2).mean() + (d_fake ** 2).mean()
        opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
        for p in disc.parameters():
            p.requires_grad_(False)
        # generator step
        ramp = min(1.0, step / max(1, args.adv_ramp))
        g_fake, f_fake = disc(fake)
        _, f_real = disc(real)
        adv = ((g_fake - 1) ** 2).mean()
        fm = sum(F.l1_loss(a, b.detach()) for a, b in zip(f_fake, f_real)) / len(f_fake)
        loss = base + ramp * (args.adv_weight * adv + args.fm_weight * fm)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt_g.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_g.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                dyn_r = (real[..., 1:] - real[..., :-1]).abs().mean().item()
                dyn_f = (fake[..., 1:] - fake[..., :-1]).abs().mean().item()
            print(json.dumps({"step": step, "base": round(float(base), 4), "adv": round(float(adv), 4),
                              "fm": round(float(fm), 4), "d": round(float(d_loss), 4),
                              "dyn_real": round(dyn_r, 3), "dyn_fake": round(dyn_f, 3)}), flush=True)
        resumer.after_step(step)

    torch.save({**{k: v for k, v in ck.items() if k != "model_state_dict"},
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "adv_steps": args.steps}, args.out_dir / "acoustic.pt")
    resumer.finish()
    print(json.dumps({"saved": str(args.out_dir / "acoustic.pt")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
