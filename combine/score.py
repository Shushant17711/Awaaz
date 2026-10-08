"""Student-only Whisper CER on GPU for eval32-<tag> dirs (same CER/norm as eval_voices.py)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import eval_voices as ev
from faster_whisper import WhisperModel
R = Path(__file__).resolve().parents[1]; m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
for tag in sys.argv[1:]:
    res = {}
    for d in sorted((R / f"eval32-{tag}").iterdir()):
        lang = ev.LANG.get(d.name[:2]); c = []
        if lang is None: continue
        for l in (d / "student" / "manifest.jsonl").read_text().splitlines():
            r = json.loads(l)
            if not r.get("ok", True): continue
            segs, _ = m.transcribe(str(d / "student" / r["wav"]), language=lang, beam_size=5)
            ref = ev.to_bengali(r["text"], ev.TO_BENGALI[d.name[:2]]) if d.name[:2] in ev.TO_BENGALI else r["text"]
            c.append(ev.cer(ref, "".join(s.text for s in segs)))
        res[d.name] = sum(c) / len(c)
    (R / "combine" / f"cer-{tag}.json").write_text(json.dumps(res, indent=1)); print(tag, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
