#!/usr/bin/env python3
"""Mean end-of-utterance pitch ratio on questions (higher = more rising), per package.
    question_eval.py <package_dir> [<package_dir> ...]"""
import json, sys
from pathlib import Path
import numpy as np
QS = {"hi_IN": "क्या तुम कल स्कूल जाओगे?", "pa_IN": "ਕੀ ਤੁਸੀਂ ਕੱਲ੍ਹ ਆਓਗੇ?", "te_IN": "మీరు రేపు వస్తారా?", "ta_IN": "நீங்கள் நாளை வருவீர்களா?",
      "bn_BD": "তুমি কি কাল আসবে?", "mr_IN": "तुम्ही उद्या याल का?", "gu_IN": "તમે કાલે આવશો?", "kn_IN": "ನೀವು ನಾಳೆ ಬರುತ್ತೀರಾ?",
      "ml_IN": "നിങ്ങൾ നാളെ വരുമോ?", "ur_PK": "کیا آپ کل آئیں گے؟", "or_IN": "ଆପଣ କାଲି ଆସିବେ କି?", "as_IN": "আপুনি কাইলৈ আহিব নেকি?"}
def ratio(a, sr=22050):
    f = []
    for s in range(0, len(a) - 1024, 256):
        w = a[s:s+1024]
        if np.sqrt((w**2).mean()) < 0.02: continue
        ac = np.correlate(w, w, "full")[1023:]; k = 50 + int(np.argmax(ac[50:400]))
        if ac[k] > 0.3*ac[0]: f.append(sr / k)
    f = np.array(f); return float(np.median(f[-6:]) / np.median(f[:len(f)//2])) if len(f) > 8 else float("nan")
res = {}
for pkg in sys.argv[1:]:
    sys.path.insert(0, pkg); import importlib, indicml; importlib.reload(indicml.model); importlib.reload(indicml)
    v = indicml.Voice(pkg); r = {l: ratio(v.synthesize(q, l)) for l, q in QS.items()}
    res[pkg] = r; sys.path.remove(pkg)
    print(f"{pkg}: mean question end-pitch ratio {np.nanmean(list(r.values())):.3f}")
print(json.dumps(res))
