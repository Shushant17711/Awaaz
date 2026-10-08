#!/bin/bash
# Build v1/v2 combinations, render eval32 for each, score student CER with GPU Whisper.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M=$ROOT/multilingual; PY=$ROOT/sanoTTS/.venv/bin/python; C=$ROOT/combine
SP=$ROOT/sanoTTS/.venv/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cudnn/lib
A1=$M/runs/acoustic-adv/acoustic.pt A2=$M/runs-v2/acoustic-adv/acoustic.pt V1=$M/runs/vocoder-long/vocoder.pt V2=$M/runs-v2/vocoder/vocoder.pt
cd $C
# $PY score.py mlfinal v2                                    # baselines rescored on the same GPU scorer
# for a in 0.25 0.5 0.75; do $PY soup.py $A1 $A2 $a ac_$a.pt; done
# $PY soup.py $V1 $V2 0.5 voc_0.5.pt
render() { rm -rf ../eval32-$1; (cd $M && $PY synth.py --eval32 --tag $1 --acoustic $2 --vocoder $3 >/dev/null); $PY score.py $1; }
# render a1v2 $A1 $V2
# render a2v1 $A2 $V1
render s50v1 $C/ac_0.5.pt $V1
render s50v2 $C/ac_0.5.pt $V2
render s50s50 $C/ac_0.5.pt $C/voc_0.5.pt
render s25v2 $C/ac_0.25.pt $V2
render s75v2 $C/ac_0.75.pt $V2
echo ALLDONE
