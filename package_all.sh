#!/usr/bin/env bash
# Package every finished voice for the sanoTTS runtime, and render one held-out
# sentence through that runtime as a smoke test.
#   -> sanoTTS/artifacts/voices/<v>/package/   and  release/<v>/  (copy)
ROOT="$(cd "$(dirname "$0")" && pwd)"; S="$ROOT/sanoTTS"; PY="$S/.venv/bin/python"
cd "$S"
for V in artifacts/voices/*/; do
  name=$(basename "$V"); [[ -f "$V/joint/decoder-student.pt" ]] || continue
  [[ -f "$V/package/manifest.json" && -f "$ROOT/release/$name/runtime-check.wav" ]] && continue
  teacher=${name%-long}; loc=${name%%-*}
  cfg="models/teachers/$teacher/$teacher.onnx.json"
  "$PY" tools/export_roota_self_contained_package.py --package-name "$name" --language "$loc" --voice "$teacher" \
    --duration-checkpoint "$V/duration/duration-student.pt" --acoustic-checkpoint "$V/joint/latent-student.pt" \
    --decoder-checkpoint "$V/joint/decoder-student.pt" --piper-config "$cfg" --out-dir "$V/package" || { echo "export failed: $name"; continue; }
  mkdir -p "$ROOT/release/$name" && cp "$V"/package/* "$ROOT/release/$name/"
  (cd pypkg && "$PY" - "$ROOT/release/$name" "$ROOT/data/text/$loc.jsonl" <<'PYEOF'
import json, sys, time, numpy as np, soundfile as sf
from pathlib import Path
from sanotts.engine import Synthesizer
pkg, corpus = Path(sys.argv[1]), sys.argv[2]
text = json.loads(open(corpus, encoding="utf-8").readlines()[14])["text"]  # held-out row
s = Synthesizer(voice_dir=pkg); t0 = time.time(); r = s.synthesize(text); dt = time.time() - t0
a = np.asarray(r); sf.write(pkg / "runtime-check.wav", a, r.sample_rate)
ok = len(a) > 0.3 * r.sample_rate and np.isfinite(a).all() and 0.01 < float(np.sqrt((a ** 2).mean())) < 0.9
(pkg / "runtime-check.json").write_text(json.dumps({"text": text, "seconds": len(a) / r.sample_rate,
    "rtf_cpu": dt / (len(a) / r.sample_rate), "rms": float(np.sqrt((a ** 2).mean())), "ok": bool(ok)}, ensure_ascii=False))
print(pkg.name, "runtime ok" if ok else "RUNTIME CHECK FAILED")
PYEOF
  ) || echo "runtime check crashed: $name"
done
