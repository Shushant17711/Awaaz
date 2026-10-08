#!/usr/bin/env python3
"""Pick the better of the standard / long student per language by held-out CER.

Reads eval/results-eval32.json (32 held-out sentences, Whisper CER) and links the
winner's package to release/best/<language>/. Languages scored only on one
variant take that one. Writes release/best/README.md with the table.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    results = json.loads((ROOT / "eval" / "results-eval32.json").read_text(encoding="utf-8"))
    by_lang: dict[str, dict[str, dict]] = {}
    for name, r in results.items():
        base = name[:-5] if name.endswith("-long") else name
        by_lang.setdefault(base, {})["long" if name.endswith("-long") else "standard"] = {**r, "name": name}
    best_dir = ROOT / "release" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    lines = ["| language (voice) | chosen | student CER | teacher CER | other variant CER |", "|---|---|---|---|---|"]
    for base, variants in sorted(by_lang.items()):
        winner = min(variants.values(), key=lambda r: r["student_cer"])
        other = [v for v in variants.values() if v is not winner]
        pkg = ROOT / "release" / winner["name"]
        if not (pkg / "manifest.json").is_file():
            continue
        dest = best_dir / base.split("-")[0]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pkg, dest)
        (dest / "SELECTED_FROM").write_text(winner["name"] + "\n")
        lines.append(f"| {base} | {'long' if winner['name'].endswith('-long') else 'standard'} | "
                     f"{winner['student_cer']:.3f} | {winner['teacher_cer']:.3f} | "
                     f"{other[0]['student_cer']:.3f} |" if other else
                     f"| {base} | only variant | {winner['student_cer']:.3f} | {winner['teacher_cer']:.3f} | — |")
    # Packaged but unscorable (no Whisper model, e.g. Odia): take the long variant,
    # which won or tied wherever it could be measured.
    scored = {b.split("-")[0] for b in by_lang}
    for pkg in sorted((ROOT / "release").glob("*/manifest.json")):
        name = pkg.parent.name
        loc = name.split("-")[0]
        if name == "best" or loc in scored or "-" not in name:
            continue
        long_pkg = ROOT / "release" / (name if name.endswith("-long") else name + "-long")
        pick = long_pkg if (long_pkg / "manifest.json").is_file() else pkg.parent
        dest = best_dir / loc
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pick, dest)
        (dest / "SELECTED_FROM").write_text(pick.name + " (unscored: no Whisper model)\n")
        scored.add(loc)
        lines.append(f"| {pick.name} | {'long' if pick.name.endswith('-long') else 'standard'} (unscored) | — | — | — |")
    (best_dir / "README.md").write_text(
        "# Best voice per language\n\nChosen by Whisper large-v3-turbo character error rate on 32 held-out "
        "sentences (lower is better). Listen before trusting.\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
