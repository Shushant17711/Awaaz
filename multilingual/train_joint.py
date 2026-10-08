#!/usr/bin/env python3
"""Phase 5: joint fine-tune of acoustic model + vocoder.

The vocoder (phase 3) only ever saw mels computed from real teacher audio; at
inference it gets the acoustic model's predicted mels, which are smoother and
slightly wrong. sanoTTS calls skipping this step "the classic failure": every
loss looks fine and the voice is brittle. Here, per step:

  * the acoustic model runs teacher-forced on the MAS alignment, so its mel is
    frame-aligned with the target audio, and keeps its own losses
    (mel L1 + prior + duration) so it does not drift;
  * the vocoder decodes a random 32-frame window of either the predicted mel
    or the real mel (50/50 per item, sanoTTS's "z-mix"), trained with mel L1 +
    multi-resolution STFT + LSGAN MPD/MRD + feature matching;
  * --ac-grad (default 0.1) of the vocoder's gradient flows back into the
    acoustic model, so it learns to produce mels the vocoder renders well.

Writes <out>/acoustic.pt and <out>/vocoder.pt in the same formats as phases 3-4,
so synth.py and export_runtime.py take them unchanged. Resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from piper.train.vits.monotonic_align import maximum_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "frontend"))
sys.path.insert(0, str(ROOT / "sanoTTS" / "tools"))
import indic  # noqa: E402
import mel as melmod  # noqa: E402
import train_roota_piper_decoder_student as dec  # noqa: E402
from acoustic import MultilingualAcoustic  # noqa: E402
from resume_state import add_resume_args, make_resumer  # noqa: E402
from train_acoustic import DATA, Utterances, collate  # noqa: E402

CTX = 2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acoustic", type=Path, default=HERE / "runs/acoustic/acoustic.pt")
    ap.add_argument("--vocoder", type=Path, default=HERE / "runs/vocoder/vocoder.pt")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/joint")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix-prob", type=float, default=0.5, help="P(vocoder sees the predicted mel) per item")
    ap.add_argument("--ac-grad", type=float, default=0.1, help="fraction of vocoder gradient reaching the acoustic model")
    ap.add_argument("--adv-weight", type=float, default=0.075)
    ap.add_argument("--fm-weight", type=float, default=0.75)
    ap.add_argument("--adv-start", type=int, default=0, help="steps before adversarial losses (if discriminators are fresh, use ~1000)")
    ap.add_argument("--langs", default="")
    ap.add_argument("--freeze-acoustic", action="store_true",
                    help="train only the vocoder (on predicted mels); the acoustic model is not updated at all")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    add_resume_args(ap)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    langs = [l for l in args.langs.split(",") if l] or sorted(p.parent.name for p in (DATA / "wav").glob("*/manifest.jsonl"))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    a_ck = torch.load(args.acoustic, map_location="cpu", weights_only=False)
    acoustic = MultilingualAcoustic(**a_ck["config"]); acoustic.load_state_dict(a_ck["model_state_dict"])
    v_ck = torch.load(args.vocoder, map_location="cpu", weights_only=False)
    vc = v_ck["config"]
    vocoder = dec.DecoderStudent(in_channels=vc["in_channels"], channels=tuple(vc["channels"]),
                                 res_layers=vc["res_layers"], variant=vc["variant"], activation=vc["activation"])
    vocoder.load_state_dict(v_ck["model_state_dict"])
    mpd = dec.MultiPeriodDiscriminator((2, 3, 5, 7, 11), (8, 16, 32, 64))
    mrd = dec.MultiResolutionSpectralDiscriminator()
    if "mpd_state_dict" in v_ck:
        mpd.load_state_dict(v_ck["mpd_state_dict"]); mrd.load_state_dict(v_ck["mrd_state_dict"])
    elif args.adv_start == 0:
        args.adv_start = 1000                       # fresh discriminators need a head start
    acoustic, vocoder, mpd, mrd = (m.to(device) for m in (acoustic, vocoder, mpd, mrd))

    opt_a = torch.optim.AdamW(acoustic.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    opt_v = torch.optim.AdamW(vocoder.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-5)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=args.lr, betas=(0.8, 0.99))
    resumer = make_resumer(args, args.out_dir)
    for name, obj in (("acoustic", acoustic), ("vocoder", vocoder), ("mpd", mpd), ("mrd", mrd),
                      ("opt_a", opt_a), ("opt_v", opt_v), ("opt_d", opt_d)):
        resumer.register(name, obj)
    start = resumer.load() + 1
    resumer.install_signal_handlers()
    min_frames = args.frames + 2 * CTX + 1
    loader = iter(torch.utils.data.DataLoader(Utterances(langs, args.max_frames, seed=start), batch_size=args.batch,
                                              num_workers=4, collate_fn=collate, persistent_workers=True))
    print(json.dumps({"langs": langs, "start": start, "steps": args.steps}), flush=True)

    for step in range(start, args.steps + 1):
        audio, n_frames, tokens, n_tok, lang = (x.to(device) for x in next(loader))
        with torch.no_grad():
            target = melmod.mel_from_audio(audio)
        F_max = target.shape[-1]
        frame_mask = torch.arange(F_max, device=device).unsqueeze(0) < n_frames.unsqueeze(1)
        tok_mask = torch.arange(tokens.shape[1], device=device).unsqueeze(0) < n_tok.unsqueeze(1)
        target = target * frame_mask.unsqueeze(1)

        # -- acoustic, teacher-forced on the MAS alignment ----------------------------
        h, mu, lang_e = acoustic.encode(tokens, tok_mask, lang)
        with torch.no_grad():
            neg = -0.5 * ((target.unsqueeze(2) - mu.unsqueeze(3)) ** 2).sum(1)
            attn = (tok_mask.unsqueeze(2) & frame_mask.unsqueeze(1)).transpose(1, 2).float()
            path = maximum_path(neg.transpose(1, 2).contiguous(), attn)
            dur = path.sum(1).long()
        mu_f = torch.bmm(mu, path.transpose(1, 2))
        mel_hat = acoustic.decode(torch.bmm(h, path.transpose(1, 2)), mu_f, frame_mask, lang_e)
        valid = frame_mask.unsqueeze(1).float()
        denom = valid.sum() * target.shape[1]
        ac_loss = ((mel_hat - target).abs() * valid).sum() / denom + (((mu_f - target) ** 2) * valid).sum() / denom
        log_d = acoustic.predict_log_dur(h, tok_mask, lang_e)
        ac_loss = ac_loss + (((log_d - torch.log1p(dur.float())) ** 2) * tok_mask).sum() / tok_mask.sum()

        # -- vocoder windows: predicted (z-mix) or real mel ---------------------------
        mel_for_voc = mel_hat.detach() if args.freeze_acoustic else args.ac_grad * mel_hat + (1 - args.ac_grad) * mel_hat.detach()
        mel_wins, audio_wins = [], []
        for b in range(audio.shape[0]):
            nf = int(n_frames[b])
            if nf < min_frames:
                continue
            s = random.randint(CTX, nf - args.frames - CTX)
            src = mel_for_voc if random.random() < args.mix_prob else target
            mel_wins.append(src[b, :, s - CTX: s + args.frames + CTX])
            audio_wins.append(audio[b, s * melmod.HOP: (s + args.frames) * melmod.HOP])
        if not mel_wins:
            continue
        mel_in = torch.stack(mel_wins)
        tgt = torch.stack(audio_wins).unsqueeze(1)
        pred = vocoder(mel_in)
        pred = (pred[0] if isinstance(pred, tuple) else pred)[..., CTX * melmod.HOP: (CTX + args.frames) * melmod.HOP]

        adv_on = step >= args.adv_start
        if adv_on:
            for disc in (mpd, mrd):
                for p in disc.parameters():
                    p.requires_grad_(True)
            d_loss = sum(dec.discriminator_lsgan_loss(disc(tgt)[0], disc(pred.detach())[0]) for disc in (mpd, mrd))
            opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
            for disc in (mpd, mrd):
                for p in disc.parameters():
                    p.requires_grad_(False)
        v_mel = F.l1_loss(melmod.mel_from_audio(pred), melmod.mel_from_audio(tgt))
        v_loss = v_mel + dec.multi_resolution_stft_loss(pred, tgt)
        adv_g = fm = torch.zeros((), device=device)
        if adv_on:
            for disc in (mpd, mrd):
                real_s, real_f = disc(tgt)
                fake_s, fake_f = disc(pred)
                adv_g = adv_g + dec.generator_lsgan_loss(fake_s)
                fm = fm + dec.discriminator_feature_matching_loss(real_f, fake_f)
            v_loss = v_loss + args.adv_weight * adv_g + args.fm_weight * fm

        loss = v_loss if args.freeze_acoustic else ac_loss + v_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt_a.zero_grad(set_to_none=True); opt_v.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 5.0)
        if not args.freeze_acoustic:
            torch.nn.utils.clip_grad_norm_(acoustic.parameters(), 1.0)
            opt_a.step()
        opt_v.step()

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "ac_loss": round(float(ac_loss), 4), "voc_mel_l1": round(float(v_mel), 4),
                              "adv_g": round(float(adv_g), 4), "fm": round(float(fm), 4)}), flush=True)
        resumer.after_step(step)

    torch.save({**{k: v for k, v in a_ck.items() if k != "model_state_dict"},
                "model_state_dict": {k: v.detach().cpu() for k, v in acoustic.state_dict().items()},
                "joint_steps": args.steps}, args.out_dir / "acoustic.pt")
    torch.save({**{k: v for k, v in v_ck.items() if k not in ("model_state_dict", "mpd_state_dict", "mrd_state_dict")},
                "model_state_dict": {k: v.detach().cpu() for k, v in vocoder.state_dict().items()},
                "mpd_state_dict": mpd.state_dict(), "mrd_state_dict": mrd.state_dict(),
                "joint_steps": args.steps}, args.out_dir / "vocoder.pt")
    resumer.finish()
    print(json.dumps({"saved": str(args.out_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
