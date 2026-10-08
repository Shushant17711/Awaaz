#!/usr/bin/env python3
"""Phase 3: one shared mel -> waveform vocoder for all languages.

Architecture is sanoTTS's piperlite decoder (DecoderStudent, variant
"piperlite") with 80 mel channels in instead of the 192-dim teacher latent.
Everything after the input conv is initialised from a Piper teacher's
generator (channel-sliced, importance-ranked), which sanoTTS found is what
makes sub-1M decoders work; the input conv starts fresh because its input
changed. Trained on the teachers' rendered audio from every language, sampled
evenly per language.

Losses: log-mel L1 + multi-resolution STFT from step 0; LSGAN multi-period +
multi-resolution-spectral discriminators with feature matching after
--adv-start (weights as in train_voice_from_piper.py's decoder stage).

Resumable (resume.pt every 250 steps and on SIGTERM). Final: <out>/vocoder.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "sanoTTS" / "tools"))
import mel as melmod  # noqa: E402
import train_roota_piper_decoder_student as dec  # noqa: E402
from piper_teacher_decoder import teacher_state_from_onnx  # noqa: E402
from resume_state import add_resume_args, make_resumer  # noqa: E402

DATA = Path(os.environ.get("ML_DATA", str(HERE / "data")))
CTX = 2  # mel context frames on each side of a training segment


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs" / "vocoder")
    ap.add_argument("--init-teacher", type=Path,
                    default=ROOT / "sanoTTS/models/teachers/hi_IN-priyamvada-medium/hi_IN-priyamvada-medium.onnx")
    ap.add_argument("--channels", default="192,96,48,24")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=32, help="mel frames per segment (x256 samples)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adv-start", type=int, default=2000)
    ap.add_argument("--adv-weight", type=float, default=0.075)
    ap.add_argument("--fm-weight", type=float, default=0.75)
    ap.add_argument("--langs", default="", help="comma list; default = every language with a manifest")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="continue from a finished vocoder.pt (generator + discriminators) instead of the teacher init")
    ap.add_argument("--lr-final", type=float, default=None, help="cosine-decay the learning rate to this by --steps")
    add_resume_args(ap)
    return ap.parse_args()


class Segments(torch.utils.data.IterableDataset):
    """Endless stream of (audio [frames*256], mel-context audio) pairs, languages sampled uniformly."""

    def __init__(self, langs: list[str], frames: int, seed: int) -> None:
        self.frames, self.seed = frames, seed
        self.items: dict[str, list[tuple[str, int]]] = {}
        for lang in langs:
            rows = [json.loads(l) for l in (DATA / "wav" / lang / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            self.items[lang] = [(str(DATA / r["wav"]), int(r["seconds"] * melmod.SR))
                                for r in rows if r.get("split") == "train" and r["seconds"] * melmod.SR > (frames + 2 * CTX + 2) * melmod.HOP]
        self.langs = [l for l in langs if self.items[l]]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 9973 + random.randrange(1 << 30))
        need = (self.frames + 2 * CTX) * melmod.HOP
        while True:
            lang = rng.choice(self.langs)
            path, n = rng.choice(self.items[lang])
            start = rng.randrange(0, max(1, n - need))
            audio, _ = sf.read(path, start=start, stop=start + need, dtype="float32")
            if audio.shape[0] < need:
                continue
            yield torch.from_numpy(audio)


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict]:
    channels = tuple(int(c) for c in args.channels.split(","))
    config = {"in_channels": melmod.N_MELS, "channels": list(channels), "res_layers": 1, "variant": "piperlite",
              "activation": "leaky_relu", "mel": {"sr": melmod.SR, "n_fft": melmod.N_FFT, "hop": melmod.HOP,
                                                    "n_mels": melmod.N_MELS, "fmin": melmod.FMIN, "fmax": melmod.FMAX}}
    model = dec.DecoderStudent(in_channels=melmod.N_MELS, channels=channels, res_layers=1,
                               variant="piperlite", activation="leaky_relu")
    # Teacher init: everything but the input conv. The teacher's conv_pre reads 192
    # latent channels; give the init routine a correctly-shaped stand-in, then reset it.
    state = teacher_state_from_onnx(args.init_teacher)
    state["conv_pre.weight"] = state["conv_pre.weight"][:, : melmod.N_MELS, :].clone()
    tmp = args.out_dir / "teacher-init.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": {}}, tmp)
    summary = dec.initialize_piperlite_from_teacher(model, tmp, channels, "importance")
    model.pre.reset_parameters()
    config["teacher_init"] = {"teacher": str(args.init_teacher), "copied": summary.get("copied")}
    return model.to(device), config


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    langs = [l for l in args.langs.split(",") if l] or sorted(p.parent.name for p in (DATA / "wav").glob("*/manifest.jsonl"))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, config = build_model(args, device)
    mpd = dec.MultiPeriodDiscriminator((2, 3, 5, 7, 11), (8, 16, 32, 64)).to(device)
    mrd = dec.MultiResolutionSpectralDiscriminator().to(device)
    if args.init_from is not None:
        prev = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(prev["model_state_dict"])
        mpd.load_state_dict(prev["mpd_state_dict"]); mrd.load_state_dict(prev["mrd_state_dict"])
        config["continued_from"] = {"checkpoint": str(args.init_from), "steps": prev.get("steps")}
        args.adv_start = 0                      # discriminators are already trained
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-5)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=args.lr, betas=(0.8, 0.99))
    print(json.dumps({"langs": langs, "vocoder_params": dec.count_parameters(model), "config": config}), flush=True)

    resumer = make_resumer(args, args.out_dir)
    for name, obj in (("model", model), ("mpd", mpd), ("mrd", mrd), ("opt_g", opt_g), ("opt_d", opt_d)):
        resumer.register(name, obj)
    import math  # noqa: PLC0415
    lr_at = (lambda st: args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * st / args.steps))) \
        if args.lr_final is not None else (lambda st: args.lr)
    start = resumer.load() + 1
    resumer.install_signal_handlers()

    loader = iter(torch.utils.data.DataLoader(Segments(langs, args.frames, seed=start), batch_size=args.batch,
                                              num_workers=4, persistent_workers=True))
    cut = CTX * melmod.HOP
    for step in range(start, args.steps + 1):
        for opt in (opt_g, opt_d):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
        audio_ctx = next(loader).to(device)
        with torch.no_grad():
            mel_in = melmod.mel_from_audio(audio_ctx)[:, :, CTX:CTX + args.frames]
        target = audio_ctx[:, cut:cut + args.frames * melmod.HOP].unsqueeze(1)
        pred = model(mel_in)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred[..., : target.shape[-1]]

        adv_on = step >= args.adv_start
        if adv_on:
            for disc in (mpd, mrd):
                for p in disc.parameters():
                    p.requires_grad_(True)
            d_loss = 0.0
            for disc in (mpd, mrd):
                real, _ = disc(target)
                fake, _ = disc(pred.detach())
                d_loss = d_loss + dec.discriminator_lsgan_loss(real, fake)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            for disc in (mpd, mrd):
                for p in disc.parameters():
                    p.requires_grad_(False)

        mel_loss = F.l1_loss(melmod.mel_from_audio(pred), melmod.mel_from_audio(target))
        stft_loss = dec.multi_resolution_stft_loss(pred, target)
        loss = mel_loss + stft_loss
        adv_g = fm = torch.zeros((), device=device)
        if adv_on:
            for disc in (mpd, mrd):
                real_s, real_f = disc(target)
                fake_s, fake_f = disc(pred)
                adv_g = adv_g + dec.generator_lsgan_loss(fake_s)
                fm = fm + dec.discriminator_feature_matching_loss(real_f, fake_f)
            loss = loss + args.adv_weight * adv_g + args.fm_weight * fm
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt_g.step()

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "mel_l1": round(float(mel_loss), 4),
                              "stft": round(float(stft_loss), 4), "adv_g": round(float(adv_g), 4),
                              "fm": round(float(fm), 4), "lr": round(lr_at(step), 7)}), flush=True)
        resumer.after_step(step)

    out = args.out_dir / "vocoder.pt"
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "steps": args.steps, "langs": langs,
                # Kept so the phase-5 joint fine-tune continues the adversarial game
                # instead of restarting it against fresh discriminators.
                "mpd_state_dict": mpd.state_dict(), "mrd_state_dict": mrd.state_dict()}, out)
    resumer.finish()
    print(json.dumps({"saved": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
