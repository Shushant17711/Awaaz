"""indicml: NumPy-only runtime for the multilingual Indic TTS.

    from indicml import Voice
    voice = Voice("path/to/package")
    audio = voice.synthesize("नमस्ते", "hi_IN")      # float32 @ voice.sample_rate

    python -m indicml PACKAGE LANG "text" out.wav
"""

from .model import FORMAT, Voice

__all__ = ["Voice", "FORMAT"]
