#!/usr/bin/env bash
# Kangri: recordings in kangri/wavs/<id>.wav -> teacher -> distillation jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; T="$ROOT/teachers"; K="$ROOT/kangri"
n=$(ls "$K"/wavs/*.wav 2>/dev/null | wc -l)
[[ $n -ge 200 ]] || { echo "only $n recordings in $K/wavs (need >=200, ideally 500+)"; exit 1; }
mkdir -p "$T/xnr" && : > "$T/xnr/metadata.csv"
"$ROOT/sanoTTS/.venv/bin/python" - "$K" "$T/xnr" <<'PY'
import sys, soundfile as sf, soxr, numpy as np
from pathlib import Path
k, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "wavs").mkdir(exist_ok=True)
rows = dict(l.rstrip("\n").split("\t", 1) for l in (k / "recording_script.tsv").read_text(encoding="utf-8").splitlines()[1:])
with open(out / "metadata.csv", "w", encoding="utf-8") as h:
    for wav in sorted((k / "wavs").glob("*.wav")):
        if wav.stem not in rows: continue
        a, sr = sf.read(wav, dtype="float32", always_2d=True); a = a.mean(1)
        if sr != 22050: a = soxr.resample(a, sr, 22050)
        sf.write(out / "wavs" / wav.name, a, 22050, subtype="PCM_16"); h.write(f"{wav.name}|{rows[wav.stem]}\n")
(out / "audio_dir").write_text(str(out / "wavs")); (out / "prep.log").write_text("done, kangri\n")
PY
# Kangri corpus rows (not the recorded ones) as distillation text.
"$ROOT/sanoTTS/.venv/bin/python" - "$K" "$ROOT/data/text/xnr_IN.jsonl" <<'PY'
import sys, json, re, random
from pathlib import Path
k, out = Path(sys.argv[1]), Path(sys.argv[2])
rec = {l.split("\t", 1)[1] for l in (k / "recording_script.tsv").read_text(encoding="utf-8").splitlines()[1:]}
c = [" ".join(l.split()) for l in (k / "Kr_4_kangri.txt").read_text(encoding="utf-8").splitlines()]
c = [t for t in dict.fromkeys(c) if 25 <= len(t) <= 170 and not re.search(r"[A-Za-z0-9]", t) and t not in rec]
random.Random(0).shuffle(c)
out.write_text("".join(json.dumps({"id": f"xnr-{i:05d}", "text": t}, ensure_ascii=False) + "\n" for i, t in enumerate(c[:3200])), encoding="utf-8")
PY
# Run the standard teacher step for this one language (Hindi base, Hindi eSpeak).
python3 - "$T" <<'PY'
import sys
from pathlib import Path
t = Path(sys.argv[1]); s = (t / "pipeline.sh").read_text()
lines = [('LANGS=("xnr|hi|rohan|xnr_IN")' if l.startswith("LANGS=(") else
          "UNIT=sanotts-teachers-kangri" if l == "UNIT=sanotts-teachers" else l) for l in s.splitlines()]
s = "\n".join(lines) + "\n"
s = s.replace('"$T/pipeline.sh" _run', '"$T/pipeline-kangri.sh" _run').replace("-indictts-medium", "-recorded-medium")
(t / "pipeline-kangri.sh").write_text(s)
PY
chmod +x "$T/pipeline-kangri.sh" && "$T/pipeline-kangri.sh" start
