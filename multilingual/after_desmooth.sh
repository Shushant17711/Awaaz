#!/usr/bin/env bash
# After the de-smoothing fine-tune: render the user's poem with it, and A/B its intelligibility
# against the shipped (pre-joint) model on the same 8 languages x 8 held-out sentences.
cd "$(dirname "$0")"; PY=../sanoTTS/.venv/bin/python
while systemctl --user is-active --quiet sanotts-desmooth; do sleep 60; done
[[ -f runs/acoustic-adv/acoustic.pt ]] || { echo "no adv checkpoint"; exit 1; }
rm -rf /tmp/awaaz-pkg-adv && $PY export_runtime.py --acoustic runs/acoustic-adv/acoustic.pt --vocoder runs/vocoder/vocoder.pt --out /tmp/awaaz-pkg-adv >/dev/null
(cd /tmp/awaaz-pkg-adv && $PY -m indicml . hi_IN "$(cat ../tests/poem_hi.txt)" ../tests/poem_hi_multilingual_desmoothed.wav)
AB_VARIANTS=adv nice -n 5 $PY ab_joint.py
