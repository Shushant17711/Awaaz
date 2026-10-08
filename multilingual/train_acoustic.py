#!/usr/bin/env python3
"""Phase 4: train the multilingual acoustic model on the synthetic corpus.

Per step: wav -> mel (GPU, mel.py), tokens from the unified front end, MAS
alignment between per-token prior means and the mel (no grad), then
    L = L1(mel_hat, mel) + MSE(mu_expanded, mel) + MSE(log_dur, log(1 + mas_dur))
Languages are sampled uniformly per batch. Resumable (resume.pt every 250 steps
and on SIGTERM). Final: <out>/acoustic.pt.
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
from piper.train.vits.monotonic_align import maximum_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "frontend"))
sys.path.insert(0, str(ROOT / "sanoTTS" / "tools"))
import indic  # noqa: E402
import mel as melmod  # noqa: E402
from acoustic import MultilingualAcoustic, count_parameters  # noqa: E402
from resume_state import add_resume_args, make_resumer  # noqa: E402

DATA = Path(os.environ.get("ML_DATA", str(HERE / "data")))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs" / "acoustic")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--enc-blocks", type=int, default=6)
    ap.add_argument("--dec-blocks", type=int, default=6)
    ap.add_argument("--langs", default="")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", type=Path, default=None, help="initialise weights from an existing acoustic.pt (same config)")
    add_resume_args(ap)
    return ap.parse_args()


class Utterances(torch.utils.data.IterableDataset):
    def __init__(self, langs: list[str], max_frames: int, seed: int) -> None:
        self.seed, self.langs, self.items = seed, langs, {}
        for lang in langs:
            rows = [json.loads(l) for l in (DATA / "wav" / lang / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            self.items[lang] = [r for r in rows if r.get("split") == "train"
                                and r["seconds"] * melmod.SR / melmod.HOP <= max_frames]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 7919 + random.randrange(1 << 30))
        while True:
            lang = rng.choice(self.langs)
            row = rng.choice(self.items[lang])
            audio, _ = sf.read(DATA / row["wav"], dtype="float32")
            ids = indic.encode(row["text"])
            frames = audio.shape[0] // melmod.HOP
            if frames < len(ids) + 2:          # MAS needs >= 1 frame per token
                continue
            yield torch.from_numpy(audio[: frames * melmod.HOP]), torch.tensor(ids), indic.LANG_ID[lang]


def collate(batch):
    audio, ids, lang = zip(*batch)
    a = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
    t = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=indic.TOKEN_ID[indic.PAD])
    return a, torch.tensor([x.shape[0] // melmod.HOP for x in audio]), t, torch.tensor([len(x) for x in ids]), torch.tensor(lang)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    langs = [l for l in args.langs.split(",") if l] or sorted(p.parent.name for p in (DATA / "wav").glob("*/manifest.jsonl"))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = MultilingualAcoustic(len(indic.VOCAB), len(indic.LANGS), dim=args.dim,
                                 enc_blocks=args.enc_blocks, dec_blocks=args.dec_blocks).to(device)
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.05, 1 - s / args.steps))
    print(json.dumps({"langs": langs, "acoustic_params": count_parameters(model), "config": model.config}), flush=True)

    resumer = make_resumer(args, args.out_dir)
    for name, obj in (("model", model), ("opt", opt), ("sched", sched)):
        resumer.register(name, obj)
    start = resumer.load() + 1
    resumer.install_signal_handlers()
    loader = iter(torch.utils.data.DataLoader(Utterances(langs, args.max_frames, seed=start), batch_size=args.batch,
                                              num_workers=4, collate_fn=collate, persistent_workers=True))

    for step in range(start, args.steps + 1):
        audio, n_frames, tokens, n_tok, lang = (x.to(device) for x in next(loader))
        with torch.no_grad():
            target = melmod.mel_from_audio(audio)                          # [B, 80, F]
        F_max = target.shape[-1]
        frame_mask = torch.arange(F_max, device=device).unsqueeze(0) < n_frames.unsqueeze(1)
        tok_mask = torch.arange(tokens.shape[1], device=device).unsqueeze(0) < n_tok.unsqueeze(1)
        target = target * frame_mask.unsqueeze(1)

        h, mu, lang_e = model.encode(tokens, tok_mask, lang)
        with torch.no_grad():                                              # MAS: log-likelihood under unit Gaussians
            neg = -0.5 * ((target.unsqueeze(2) - mu.unsqueeze(3)) ** 2).sum(1)   # [B, T_tok, F]
            attn_mask = tok_mask.unsqueeze(2) & frame_mask.unsqueeze(1)
            path = maximum_path(neg.transpose(1, 2).contiguous(), attn_mask.transpose(1, 2).float())  # [B, F, T_tok]
            dur = path.sum(1).long()                                       # frames per token
        mu_frames = torch.bmm(mu, path.transpose(1, 2))                   # [B, 80, F]
        h_frames = torch.bmm(h, path.transpose(1, 2))
        mel_hat = model.decode(h_frames, mu_frames, frame_mask, lang_e)

        valid = frame_mask.unsqueeze(1).float()
        denom = valid.sum() * target.shape[1]
        mel_loss = (torch.abs(mel_hat - target) * valid).sum() / denom
        prior_loss = (((mu_frames - target) ** 2) * valid).sum() / denom
        log_d = model.predict_log_dur(h, tok_mask, lang_e)
        dur_loss = (((log_d - torch.log1p(dur.float())) ** 2) * tok_mask).sum() / tok_mask.sum()
        loss = mel_loss + prior_loss + dur_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "mel_l1": round(float(mel_loss), 4),
                              "prior": round(float(prior_loss), 4), "dur": round(float(dur_loss), 4),
                              "lr": round(sched.get_last_lr()[0], 6)}), flush=True)
        resumer.after_step(step)

    out = args.out_dir / "acoustic.pt"
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": model.config, "vocab": indic.VOCAB, "langs": indic.LANGS, "steps": args.steps}, out)
    resumer.finish()
    print(json.dumps({"saved": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
