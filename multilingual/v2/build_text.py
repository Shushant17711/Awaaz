#!/usr/bin/env python3
"""v2 corpus text: existing rows (same held-out set) + 10k new sentences + questions, per language.

Writes data-v2/text/<loc>.jsonl. Rows marked "question": true get a rising terminal
pitch when rendered (render_corpus.question_rise): every real question ('?'-final)
in the language's Leipzig corpora, plus 500 statements turned into questions by
replacing the final punctuation with '?'.
"""

from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "multilingual" / "data" / "text"
OUT = ROOT / "multilingual" / "data-v2" / "text"
RAW = ROOT / "data" / "text" / "raw"
L = {"hin": ("hi_IN", r"[ऀ-ॿ]"), "mar": ("mr_IN", r"[ऀ-ॿ]"), "ben": ("bn_BD", r"[ঀ-৿]"),
     "asm": ("as_IN", r"[ঀ-৿]"), "pan": ("pa_IN", r"[਀-੿]"), "guj": ("gu_IN", r"[઀-૿]"),
     "ori": ("or_IN", r"[଀-୿]"), "tam": ("ta_IN", r"[஀-௿]"), "tel": ("te_IN", r"[ఀ-౿]"),
     "kan": ("kn_IN", r"[ಀ-೿]"), "mal": ("ml_IN", r"[ഀ-ൿ]"), "urd": ("ur_PK", r"[؀-ۿ]")}
NEW, CONVERTED = 10000, 500


def clean(t: str, code: str, script: str) -> str | None:
    if code == "asm":
        t = t.replace("য়", "য়")
    if not (12 <= len(t) <= 170) or re.search(r"[A-Za-z]", t) or re.search(r"[0-9]{3,}", t) \
            or re.search(r'[\[\]{}<>|=_#@*/\\"]', t):
        return None
    letters = len(re.findall(r"\w", t))
    if not letters or len(re.findall(script, t)) / letters < 0.9:
        return None
    return t


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for code, (loc, script) in L.items():
        old = [json.loads(l) for l in (V1 / f"{loc}.jsonl").read_text(encoding="utf-8").splitlines()]
        seen = {r["text"] for r in old}
        pool, questions = [], []
        for f in sorted(glob.glob(str(RAW / f"{code}_wikipedia_*" / "*-sentences.txt"))):
            for line in open(f, encoding="utf-8"):
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) < 2:
                    continue
                t = clean(parts[1].strip(), code, script)
                if t is None or t in seen:
                    continue
                seen.add(t)
                (questions if t.endswith(("?", "؟")) else pool).append(t)
        rng = random.Random(2)
        rng.shuffle(pool)
        new, conv = pool[:NEW], pool[NEW:NEW + CONVERTED]
        conv = [re.sub(r"[\s.।॥۔!,;:]*$", "", t) + ("؟" if loc == "ur_PK" else "?") for t in conv]
        rows = list(old)
        rows += [{"id": f"{loc}-v2-{i:05d}", "text": t, "split": "train"} for i, t in enumerate(new)]
        rows += [{"id": f"{loc}-q-{i:05d}", "text": t, "split": "train", "question": True}
                 for i, t in enumerate(questions + conv)]
        with (OUT / f"{loc}.jsonl").open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{loc}: {len(old)} existing + {len(new)} new + {len(questions)} real questions + {len(conv)} converted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
