#!/usr/bin/env bash
# v2: one unattended run that addresses the listening-review feedback without changing
# model size (2.4M params). Every stage is resumable and skipped once done
# (markers in v2/state/); trainers resume mid-stage from resume.pt.
#
#   1  teachers  continue or/as/kn/pa teachers +10k steps; keep each only if Whisper CER
#                doesn't get worse (metallic voices, dataset disparity)   [GPU]
#      text+render (in parallel) render 10k new sentences + questions (with rising
#                pitch) for the 8 other languages                          [CPU]
#   2  render    the 4 teacher languages (all rows if their teacher changed)
#   3  vocoder   full-band mel (fmax 11025: sibilants), 150k steps from the 260k vocoder
#   4  acoustic  150k steps on the doubled corpus (+questions), init from v1
#   5  desmooth  20k steps mel-adversarial
#   6  release   export, parity, 12x32 Whisper eval, question-pitch eval; promote to
#                release/multilingual{,-int8} only if not worse than the current release;
#                review recording; REPORT-v2.md
#
#   v2/run_v2.sh            run (or resume)
#   journalctl --user -u sanotts-v2 -f      watch;   cat multilingual/v2/REPORT-v2.md    result
set -uo pipefail
V="$(cd "$(dirname "$0")" && pwd)"; M="$(cd "$V/.." && pwd)"; ROOT="$(cd "$M/.." && pwd)"
PY="$ROOT/sanoTTS/.venv/bin/python"; S="$ROOT/sanoTTS"; T="$ROOT/teachers"; ST="$V/state"
export ML_DATA="$M/data-v2" MEL_FMAX=11025 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R2="$M/runs-v2"; mkdir -p "$ST" "$R2" "$ML_DATA/wav"
log() { echo "== $(date '+%F %T') $*"; }
done_() { [[ -f "$ST/$1.done" ]]; }
mark() { touch "$ST/$1.done"; }
attempt() {   # attempt NAME: give up on a stage after 3 failed tries (so a restart loop can't spin)
  local n=$(( $(cat "$ST/$1.tries" 2>/dev/null || echo 0) + 1 )); echo $n > "$ST/$1.tries"
  [[ $n -le 3 ]] || { log "$1 failed 3 times; giving up on it"; return 1; }
}
gpu() { flock "$ROOT/.gpu.lock" "$@"; }

link_v1() {   # reuse the v1 renders of a language (hard links, no extra disk)
  local loc=$1 src="$M/data/wav/$loc" dst="$ML_DATA/wav/$loc"
  [[ -f "$dst/manifest.jsonl" ]] && return 0
  mkdir -p "$dst" && cp -l "$src"/*.wav "$dst"/ 2>/dev/null; cp "$src/manifest.jsonl" "$dst/manifest.jsonl"
}

stage_teachers() {   # [GPU]
  for x in "or|or|bengali|or_IN" "as|as|bengali|as_IN" "kn|kn|maya|kn_IN" "pa|pa|rohan|pa_IN"; do
    IFS='|' read -r lang espeak base loc <<<"$x"
    done_ "teacher_$lang" && continue
    attempt "teacher_$lang" || { echo "$lang keep-v1" >> "$ST/teacher_decisions"; mark "teacher_$lang"; continue; }
    local name="${loc}-indictts-medium" d="$S/models/teachers/${loc}-indictts-medium" td="$T/$lang"
    [[ -f "$td/final.v1.onnx" ]] || cp "$td/final.onnx" "$td/final.v1.onnx"
    local ep n extra
    ep=$("$PY" -c "import torch,glob; print(torch.load(sorted(glob.glob('$td/lightning_logs/*/checkpoints/last.ckpt'))[-1],map_location='cpu',weights_only=False)['epoch'])")
    n=$(wc -l < "$td/metadata.train.csv"); extra=$(( (10000 * 12 + n - 1) / n ))
    log "teacher $lang: continue from epoch $ep for +$extra epochs (~10k steps)"
    ( cd "$td" && gpu "$T/train_teacher.sh" "$lang" "$espeak" "$base" $(( ep + 1 + extra )) 12 ) || { log "teacher $lang training failed"; return 1; }
    local last; last=$(ls -t "$td"/lightning_logs/*/checkpoints/last.ckpt | head -1)
    "$ROOT/train-venv/bin/python" -m piper.train.export_onnx --checkpoint "$last" --output-file "$td/final.v2.raw.onnx" >/dev/null 2>&1 \
      && "$PY" "$S/tools/canonicalize_piper_onnx.py" "$td/final.v2.raw.onnx" "$td/final.v2.onnx" >/dev/null || { log "teacher $lang export failed"; return 1; }
    local g; g=$("$PY" "$V/teacher_gate.py" "$loc" "$d/$name.onnx" "$td/final.v2.onnx" "$d/$name.onnx.json" | tail -1)
    log "teacher $lang gate: $g"
    if [[ "$g" == *'"accept": true'* ]]; then
      [[ -f "$d/$name.v1.onnx" ]] || cp "$d/$name.onnx" "$d/$name.v1.onnx"
      cp "$td/final.v2.onnx" "$d/$name.onnx"; echo "$lang v2 $g" >> "$ST/teacher_decisions"
    else
      echo "$lang keep-v1 $g" >> "$ST/teacher_decisions"
    fi
    rm -rf "$td/cache"; find "$td/lightning_logs" -name "*.ckpt" ! -name last.ckpt -delete
    mark "teacher_$lang"
  done
  mark teachers
}

stage_render_other() {   # [CPU] the 8 languages whose teachers are unchanged
  done_ render_other && return 0
  attempt render_other || return 1
  for loc in hi_IN mr_IN bn_BD gu_IN ta_IN te_IN ml_IN ur_PK; do link_v1 "$loc"; done
  "$PY" "$M/render_corpus.py" --threads 6 --langs hi_IN,mr_IN,bn_BD,gu_IN,ta_IN,te_IN,ml_IN,ur_PK || return 1
  mark render_other
}

stage_render_teacher_langs() {
  done_ render_teach && return 0
  attempt render_teach || return 1
  for lang in or as kn pa; do
    loc=$(case $lang in or) echo or_IN;; as) echo as_IN;; kn) echo kn_IN;; pa) echo pa_IN;; esac)
    grep -q "^$lang v2" "$ST/teacher_decisions" 2>/dev/null || link_v1 "$loc"   # new teacher -> render everything fresh
  done
  "$PY" "$M/render_corpus.py" --threads 8 --langs or_IN,as_IN,kn_IN,pa_IN || return 1
  mark render_teach
}

run_stage() {   # run_stage NAME cmd...
  local name=$1; shift
  done_ "$name" && return 0
  attempt "$name" || return 1
  log "$name"
  gpu "$@" || { log "$name failed"; return 1; }
  mark "$name"
}

stage_release() {
  done_ release && return 0
  attempt release || return 1
  local A="$R2/acoustic-adv/acoustic.pt" VO="$R2/vocoder/vocoder.pt" C="$ROOT/release/candidate-v2" C8="$ROOT/release/candidate-v2-int8"
  rm -rf "$C" "$C8"
  "$PY" "$M/export_runtime.py" --acoustic "$A" --vocoder "$VO" --out "$C" >/dev/null || return 1
  "$PY" "$M/export_runtime.py" --acoustic "$A" --vocoder "$VO" --out "$C8" --int8 >/dev/null || return 1
  "$PY" "$M/test_runtime.py" --acoustic "$A" --vocoder "$VO" > "$V/parity.txt" 2>&1 || { log "parity failed"; return 1; }
  rm -rf "$ROOT/eval32-v2"; "$PY" "$M/synth.py" --eval32 --tag v2 --acoustic "$A" --vocoder "$VO" || return 1
  rm -f "$ROOT/eval/results-eval32-v2."*
  ( cd "$ROOT" && nice -n 5 "$PY" eval_voices.py large-v3-turbo eval32-v2 ) || return 1
  "$PY" "$V/question_eval.py" "$ROOT/release/multilingual" > "$V/question_v1.txt" 2>&1
  "$PY" "$V/question_eval.py" "$C" > "$V/question_v2.txt" 2>&1
  local verdict; verdict=$("$PY" - "$ROOT" <<'PY'
import json, sys, statistics
root = sys.argv[1]
new = json.load(open(f"{root}/eval/results-eval32-v2.json")); cur = json.load(open(f"{root}/eval/results-eval32-mlfinal.json"))
langs = sorted(set(new) & set(cur))
mn = statistics.mean(new[l]["student_cer"] for l in langs); mc = statistics.mean(cur[l]["student_cer"] for l in langs)
worst = max(new[l]["student_cer"] - cur[l]["student_cer"] for l in langs if l != "ml_IN")
ok = mn <= mc + 0.003 and worst <= 0.06
print(json.dumps({"promote": ok, "mean_v2": round(mn, 4), "mean_current": round(mc, 4), "worst_lang_regression": round(worst, 4),
                  "per_lang": {l: [round(cur[l]["student_cer"], 3), round(new[l]["student_cer"], 3)] for l in langs}}))
PY
)
  echo "$verdict" > "$V/verdict.json"; log "verdict: $verdict"
  local target="$C"
  if [[ "$verdict" == *'"promote": true'* ]]; then
    local stamp; stamp=$(date +%Y%m%d-%H%M)
    mv "$ROOT/release/multilingual" "$ROOT/release/multilingual.prev-$stamp"
    mv "$ROOT/release/multilingual-int8" "$ROOT/release/multilingual-int8.prev-$stamp"
    cp -r "$C" "$ROOT/release/multilingual"; cp -r "$C8" "$ROOT/release/multilingual-int8"
    ML_RESULTS="$ROOT/eval/results-eval32-v2.json" "$M/finalize.sh" >/dev/null 2>&1
    RELEASE_DIR="$ROOT/release/multilingual-int8" ML_RESULTS="$ROOT/eval/results-eval32-v2.json" "$M/finalize.sh" >/dev/null 2>&1
    target="$ROOT/release/multilingual"; log "PROMOTED v2 to release/multilingual (previous kept as .prev-$stamp)"
  else
    RELEASE_DIR="$C" ML_RESULTS="$ROOT/eval/results-eval32-v2.json" "$M/finalize.sh" >/dev/null 2>&1
    log "v2 NOT promoted (worse on the gate); candidate kept in release/candidate-v2"
  fi
  "$PY" "$V/make_review.py" "$target" "$ROOT/review" v2 || true
  write_report "$verdict"
  mark release
}

write_report() {
  {
    echo "# v2 run report ($(date '+%F %T'))"; echo
    echo "## Verdict"; echo '```'; echo "$1"; echo '```'
    echo "Gate: promote only if mean Whisper CER (12 langs x 32 held-out sentences) <= current + 0.003 and no language (excluding Malayalam) worse by > 0.06."; echo
    echo "## Teachers (continued +10k steps; kept only if not worse)"; echo '```'; cat "$ST/teacher_decisions" 2>/dev/null; echo '```'
    echo "## Question intonation (end-of-question pitch / first-half pitch; higher = more rising)"; echo '```'
    grep "mean question" "$V/question_v1.txt" "$V/question_v2.txt" 2>/dev/null; echo '```'
    echo "## Runtime parity"; echo '```'; grep -E "PASS|FAIL|package" "$V/parity.txt" 2>/dev/null; echo '```'
    echo "## Listen"; echo "- review/indic_tts_review_v2.mp3 (+ review_sheet_v2.pdf): each language now ends with a question."
    echo "- release/multilingual (if promoted) or release/candidate-v2 (if not)."
  } > "$V/REPORT-v2.md"
}

main() {
  log "v2 start"
  [[ -f "$ML_DATA/text/hi_IN.jsonl" ]] || "$PY" "$V/build_text.py" || exit 1
  # stage 1: teachers on the GPU while the CPU renders the other 8 languages
  stage_render_other & RP=$!
  done_ teachers || stage_teachers || log "teacher stage error (continuing with whatever teachers are in place)"
  wait $RP || { log "render (8 langs) failed"; exit 1; }
  stage_render_teacher_langs || { log "render (teacher langs) failed"; exit 1; }
  run_stage vocoder "$PY" "$M/train_vocoder.py" --steps 150000 --init-from "$M/runs/vocoder-long/vocoder.pt" --lr-final 1e-5 --out-dir "$R2/vocoder" || exit 1
  run_stage acoustic "$PY" "$M/train_acoustic.py" --steps 150000 --init "$M/runs/acoustic/acoustic.pt" --out-dir "$R2/acoustic" || exit 1
  run_stage desmooth "$PY" "$M/train_acoustic_adv.py" --steps 20000 --acoustic "$R2/acoustic/acoustic.pt" --out-dir "$R2/acoustic-adv" || exit 1
  stage_release || exit 1
  log "v2 DONE: see multilingual/v2/REPORT-v2.md"
}
main
