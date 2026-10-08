"""Isolate where the buzz comes from: teacher vs vocoder(teacher mel) vs full pipeline."""
import sys, json, os
from pathlib import Path
import numpy as np, soundfile as sf, torch, librosa
H = Path(__file__).resolve().parents[1]/"multilingual"
sys.path[:0] = [str(H), str(H/"frontend"), str(H.parent/"sanoTTS/tools")]
import mel as melmod
from synth import Synthesizer
tag, ac, voc = sys.argv[1], sys.argv[2], sys.argv[3]
syn = Synthesizer(Path(ac), Path(voc))
if os.environ.get("AA"):
    import aa; aa.attach(syn.vocoder, {k: 8 for k in os.environ["AA"].split(",")})
out = Path(tag); out.mkdir(exist_ok=True)
for lang in ["hi_IN","ta_IN","kn_IN","or_IN","as_IN","bn_BD"]:
    man = [json.loads(l) for l in open(H.parent/f"eval32-mlfinal/{lang}/teacher/manifest.jsonl")]
    for k in (0, 1):
        r = man[k]; t, _ = sf.read(H.parent/f"eval32-mlfinal/{lang}/teacher/{r['wav']}", dtype="float32")
        with torch.no_grad():
            m = melmod.mel_from_audio(torch.from_numpy(t)[None])
            v = syn.vocoder(m); v = (v[0] if isinstance(v, tuple) else v).reshape(-1).clamp(-1,1).numpy()
        f = syn(r["text"], lang)
        sf.write(out/f"{lang}_{k}_teacher.wav", t, 22050)
        sf.write(out/f"{lang}_{k}_voconly.wav", v, 22050)
        sf.write(out/f"{lang}_{k}_full.wav", f, 22050)
print("ok", tag)
