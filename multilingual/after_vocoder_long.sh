#!/usr/bin/env bash
# After the 200k-step vocoder continuation: objective checks, A/B, and ship if better.
set -uo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$M/.." && pwd)"; PY="$ROOT/sanoTTS/.venv/bin/python"; cd "$M"
while systemctl --user is-active --quiet sanotts-vocoder-long; do sleep 120; done
NEWV="$M/runs/vocoder-long/vocoder.pt"; A="$M/runs/acoustic-adv/acoustic.pt"
[[ -f "$NEWV" ]] || { echo "no long vocoder checkpoint"; exit 1; }
echo "== resynthesis check (teacher mel -> vocoder), old vs new"
"$PY" - "$M" <<'PY'
import sys, json, numpy as np, soundfile as sf, torch
M = sys.argv[1]; sys.path.insert(0, M); sys.path.insert(0, M + "/../sanoTTS/tools")
import mel as melmod, train_roota_piper_decoder_student as dec
from scipy.signal import welch
def load(p):
    v = torch.load(p, map_location="cpu", weights_only=False); c = v["config"]
    m = dec.DecoderStudent(in_channels=80, channels=tuple(c["channels"]), res_layers=1, variant="piperlite", activation="leaky_relu")
    m.load_state_dict(v["model_state_dict"]); return m.eval()
vocs = {"old": load(M + "/runs/vocoder/vocoder.pt"), "new": load(M + "/runs/vocoder-long/vocoder.pt")}
res = {k: [] for k in vocs}; hf = {k: [] for k in vocs}; hf_t = []
for lang in ["hi_IN", "ta_IN", "bn_BD", "te_IN", "kn_IN", "gu_IN", "mr_IN", "ur_PK", "pa_IN", "or_IN", "as_IN", "ml_IN"]:
    rows = [json.loads(l) for l in open(f"{M}/data/wav/{lang}/manifest.jsonl", encoding="utf-8") if '"heldout"' in l][12:16]
    for r in rows:
        a, sr = sf.read(f"{M}/data/{r['wav']}", dtype="float32"); n = len(a) // 256 * 256; a = a[:n]
        ma = melmod.mel_from_audio(torch.from_numpy(a)[None])
        fr, p = welch(a, sr, nperseg=2048); hf_t.append(10*np.log10(p[fr >= 4000].sum() / p.sum()))
        for k, m in vocs.items():
            with torch.no_grad(): y = m(ma); y = (y[0] if isinstance(y, tuple) else y).reshape(-1)[:n]
            res[k].append(float((melmod.mel_from_audio(y[None]) - ma).abs().mean()))
            fr, p = welch(y.numpy(), sr, nperseg=2048); hf[k].append(10*np.log10(p[fr >= 4000].sum() / p.sum()))
for k in vocs:
    print(f"{k} vocoder: resynthesis mel-L1 {np.mean(res[k]):.4f}  energy above 4 kHz vs teacher {np.mean(hf[k]) - np.mean(hf_t):+.2f} dB")
PY
echo "== poem"
rm -rf /tmp/awaaz-pkg-long && "$PY" export_runtime.py --acoustic "$A" --vocoder "$NEWV" --out /tmp/awaaz-pkg-long >/dev/null
(cd /tmp/awaaz-pkg-long && "$PY" -m indicml . hi_IN "$(cat "$ROOT/tests/poem_hi.txt")" "$ROOT/tests/poem_hi_multilingual_longvocoder.wav")
echo "== A/B intelligibility (8 langs x 8 sentences); shipped model scored 0.175"
AB_VARIANTS=advlong nice -n 5 "$PY" ab_joint.py
