"""Unified Indic text front end: Unicode text -> shared token ids. No eSpeak.

The nine Brahmic Unicode blocks (Devanagari, Bengali/Assamese, Gurmukhi,
Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) share one 128-codepoint
layout inherited from ISCII: the same offset is the same letter. So
क (U+0915) ক (U+0995) ਕ (U+0A15) ક (U+0A95) କ (U+0B15) க (U+0B95) క (U+0C15)
ಕ (U+0C95) ക (U+0D15) all become one token, "b:15". Offsets 0x70 and up hold
script-specific letters (Assamese ৰ ৱ, Gurmukhi tippi/addak, Malayalam chillus),
which keep a per-script token. Digits (offsets 0x66-0x6F, plus ASCII 0-9) are
shared too. Urdu (Arabic block) gets its own range.

Pronunciation (schwa deletion, inherent vowels, Tamil's unmarked voicing) is
left to the model, conditioned on a language id passed separately.

Normalization: nukta letters are always split into base + nukta (U+093C and
its counterparts), so the same letter is the same tokens in every script, and
whether the input used precomposed or decomposed nukta forms doesn't matter.
"""

from __future__ import annotations

import re
import unicodedata

BRAHMIC = {  # block base -> script tag
    0x0900: "deva", 0x0980: "beng", 0x0A00: "guru", 0x0A80: "gujr", 0x0B00: "orya",
    0x0B80: "taml", 0x0C00: "telu", 0x0C80: "knda", 0x0D00: "mlym",
}
LANGS = ["hi_IN", "mr_IN", "bn_BD", "as_IN", "pa_IN", "gu_IN", "or_IN", "ta_IN", "te_IN", "kn_IN", "ml_IN", "ur_PK"]

PAD, BOS, EOS, SPACE, UNK = "<pad>", "<bos>", "<eos>", " ", "<unk>"
PUNCT = {  # everything spoken as a pause collapses onto four prosodic tokens
    ",": ",", ";": ",", ":": ",", "-": ",", "–": ",", "—": ",", "،": ",", "؛": ",",
    ".": ".", "।": ".", "॥": ".", "۔": ".",
    "?": "?", "؟": "?",
    "!": "!",
}
# Precomposed nukta letters that NFC leaves composed in some scripts; split them.
_NUKTA_SPLIT = {
    "ऩ": "ऩ", "ऱ": "ऱ", "ऴ": "ऴ",
    "ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়",
    "ਲ਼": "ਲ਼", "ਸ਼": "ਸ਼", "ਖ਼": "ਖ਼", "ਗ਼": "ਗ਼",
    "ਜ਼": "ਜ਼", "ਫ਼": "ਫ਼",
    "ଡ଼": "ଡ଼", "ଢ଼": "ଢ଼",
}
_INVISIBLE = re.compile("[​‌‍⁠﻿­]")


def _build_vocab() -> list[str]:
    vocab = [PAD, BOS, EOS, UNK, SPACE, ",", ".", "?", "!"]
    vocab += [f"b:{o:02x}" for o in range(0x00, 0x70)]           # shared Brahmic letters, marks, digits
    vocab += [f"{tag}:{o:02x}" for tag in BRAHMIC.values() for o in range(0x70, 0x80)]  # script-specific
    vocab += [f"ar:{cp:04x}" for cp in range(0x0600, 0x0700)]     # Urdu / Perso-Arabic
    return vocab


VOCAB = _build_vocab()
TOKEN_ID = {tok: i for i, tok in enumerate(VOCAB)}
LANG_ID = {lang: i for i, lang in enumerate(LANGS)}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = "".join(_NUKTA_SPLIT.get(ch, ch) for ch in unicodedata.normalize("NFD", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_token(ch: str) -> str:
    cp = ord(ch)
    if ch.isspace():
        return SPACE
    if ch in PUNCT:
        return PUNCT[ch]
    if "0" <= ch <= "9":
        return f"b:{0x66 + cp - 0x30:02x}"                        # ASCII digits share the Brahmic digit tokens
    if 0x0900 <= cp < 0x0D80:
        base = cp & ~0x7F
        off = cp - base
        if base in BRAHMIC:
            return f"b:{off:02x}" if off < 0x70 else f"{BRAHMIC[base]}:{off:02x}"
    if 0x0600 <= cp < 0x0700:
        if 0x06F0 <= cp <= 0x06F9:                                # Extended Arabic-Indic digits -> shared digits
            return f"b:{0x66 + cp - 0x06F0:02x}"
        if 0x0660 <= cp <= 0x0669:
            return f"b:{0x66 + cp - 0x0660:02x}"
        return f"ar:{cp:04x}"
    if unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("S"):
        return ","                                                # other punctuation/symbols: short pause
    return UNK


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for ch in normalize(text):
        tok = char_token(ch)
        if tok in (SPACE, ",", ".", "?", "!") and tokens and tokens[-1] == SPACE:
            tokens.pop()                                          # "word ," -> "word,"
        if tok == SPACE and tokens and tokens[-1] in (SPACE, ",", ".", "?", "!"):
            continue
        tokens.append(tok)
    while tokens and tokens[-1] == SPACE:
        tokens.pop()
    return tokens


def encode(text: str) -> list[int]:
    """Text -> token ids with BOS/EOS. Unknown characters map to <unk>."""
    return [TOKEN_ID[BOS]] + [TOKEN_ID.get(t, TOKEN_ID[UNK]) for t in tokenize(text)] + [TOKEN_ID[EOS]]
