"""Tests for the unified Indic front end. Run: python -m pytest multilingual/frontend -q
(or plain `python multilingual/frontend/test_indic.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indic  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "text"


def test_ka_is_one_token_across_scripts():
    kas = ["क", "ক", "ਕ", "ક", "କ", "க", "క", "ಕ", "ക"]
    assert {tuple(indic.tokenize(k)) for k in kas} == {("b:15",)}


def test_virama_shared():
    viramas = ["्", "্", "੍", "્", "୍", "்", "్", "್", "്"]
    assert {indic.char_token(v) for v in viramas} == {"b:4d"}


def test_script_specific_letters_kept_apart():
    assert indic.char_token("ৰ") == "beng:70"           # Assamese ra
    assert indic.char_token("ੰ") == "guru:70"           # Gurmukhi tippi
    assert indic.char_token("ൺ") == "mlym:7a"           # Malayalam chillu nn
    assert indic.char_token("ৰ") != indic.char_token("र")


def test_nukta_forms_identical():
    # Assamese/Bengali য়: precomposed U+09DF and decomposed U+09AF U+09BC must match.
    assert indic.tokenize("য়") == indic.tokenize("য়") == ["b:2f", "b:3c"]
    # Devanagari क़: precomposed U+0958 vs decomposed.
    assert indic.tokenize("क़") == indic.tokenize("क़")


def test_digits_shared():
    assert indic.tokenize("5") == indic.tokenize("५") == indic.tokenize("৫") == indic.tokenize("۵") == ["b:6b"]


def test_punctuation_and_spaces():
    assert indic.tokenize("नमस्ते ,  दुनिया।") == indic.tokenize("नमस्ते, दुनिया ।")
    assert indic.tokenize("کیا؟")[-1] == "?"
    assert indic.tokenize("  ") == []


def test_zero_width_joiners_dropped():
    assert indic.tokenize("क्‍ष") == indic.tokenize("क्ष")


def test_encode_has_bos_eos_and_ids_in_range():
    ids = indic.encode("ਪੰਜਾਬੀ ਭਾਸ਼ਾ")
    assert ids[0] == indic.TOKEN_ID[indic.BOS] and ids[-1] == indic.TOKEN_ID[indic.EOS]
    assert all(0 <= i < len(indic.VOCAB) for i in ids)


def test_corpus_coverage():
    """Every language corpus: >= 99.5% of characters map to a real token (not <unk>)."""
    for lang in indic.LANGS:
        path = DATA / f"{lang}.jsonl"
        if not path.is_file():
            continue
        total = unk = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            toks = indic.tokenize(json.loads(line)["text"])
            total += len(toks)
            unk += sum(t == indic.UNK for t in toks)
        assert total > 0
        assert unk / total <= 0.005, f"{lang}: {unk}/{total} unknown tokens"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as exc:
                fails += 1
                print("FAIL", name, exc)
    raise SystemExit(1 if fails else 0)
