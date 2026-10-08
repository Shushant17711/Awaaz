#!/usr/bin/env python3
"""Build the 12-language review recording + PDF sheet for a release package.

    make_review.py <package_dir> <out_dir> <tag>
Writes <out_dir>/indic_tts_review_<tag>.{wav,mp3}, review_sheet_<tag>.{md,html,pdf}.
Same sentences as the first review, plus one question per language so reviewers can
judge question intonation.
"""
import html, json, subprocess, sys, wave
from pathlib import Path
import numpy as np
pkg, out, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(pkg))
from indicml import Voice
v = Voice(pkg); SR = v.sample_rate; out.mkdir(parents=True, exist_ok=True)
L = [("hi_IN","Hindi","हिंदी","आज मौसम बहुत सुहाना है, चलो पार्क में टहलने चलते हैं।","भारत में सौ से भी अधिक भाषाएँ बोली जाती हैं।","क्या तुम कल स्कूल जाओगे?"),
 ("bn_BD","Bengali","বাংলা","আজ সকালে আমি বাজার থেকে তাজা সবজি কিনেছি।","কলকাতা শহরটি তার সংস্কৃতি এবং খাবারের জন্য বিখ্যাত।","তুমি কি কাল আসবে?"),
 ("ta_IN","Tamil","தமிழ்","இன்று மாலை நாம் கடற்கரைக்குச் செல்லலாம்.","தமிழ் உலகின் மிகப் பழமையான மொழிகளில் ஒன்று.","நீங்கள் நாளை வருவீர்களா?"),
 ("te_IN","Telugu","తెలుగు","మీరు ఈ రోజు ఎలా ఉన్నారు?","హైదరాబాద్ బిర్యానీ ప్రపంచవ్యాప్తంగా ప్రసిద్ధి చెందింది.","మీరు రేపు వస్తారా?"),
 ("mr_IN","Marathi","मराठी","मला मराठी बोलायला खूप आवडते.","पुणे हे महाराष्ट्रातील एक महत्त्वाचे शहर आहे.","तुम्ही उद्या याल का?"),
 ("gu_IN","Gujarati","ગુજરાતી","આજે આપણે સાથે ચા પીશું.","અમદાવાદ ગુજરાતનું સૌથી મોટું શહેર છે.","તમે કાલે આવશો?"),
 ("pa_IN","Punjabi","ਪੰਜਾਬੀ","ਤੁਹਾਡਾ ਕੀ ਹਾਲ ਹੈ?","ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਸਥਿਤ ਹੈ।","ਕੀ ਤੁਸੀਂ ਕੱਲ੍ਹ ਆਓਗੇ?"),
 ("ur_PK","Urdu","اردو","آپ سے مل کر بہت خوشی ہوئی۔","لاہور اپنے باغوں اور کھانوں کے لیے مشہور ہے۔","کیا آپ کل آئیں گے؟"),
 ("kn_IN","Kannada","ಕನ್ನಡ","ಬೆಂಗಳೂರಿನಲ್ಲಿ ಇಂದು ಮಳೆ ಬರಬಹುದು.","ಕನ್ನಡ ಸಾಹಿತ್ಯಕ್ಕೆ ಎಂಟು ಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿಗಳು ಬಂದಿವೆ.","ನೀವು ನಾಳೆ ಬರುತ್ತೀರಾ?"),
 ("ml_IN","Malayalam","മലയാളം","ഇന്ന് നല്ല ദിവസമാണ്.","കേരളം അതിന്റെ പ്രകൃതി ഭംഗിക്ക് പ്രശസ്തമാണ്.","നിങ്ങൾ നാളെ വരുമോ?"),
 ("or_IN","Odia","ଓଡ଼ିଆ","ଆପଣ କେମିତି ଅଛନ୍ତି?","ପୁରୀର ଜଗନ୍ନାଥ ମନ୍ଦିର ବହୁତ ପ୍ରସିଦ୍ଧ।","ଆପଣ କାଲି ଆସିବେ କି?"),
 ("as_IN","Assamese","অসমীয়া","আপোনাৰ নাম কি?","গুৱাহাটী অসমৰ আটাইতকৈ ডাঙৰ চহৰ।","আপুনি কাইলৈ আহিব নেকি?")]
sil = lambda s: np.zeros(int(s * SR), np.float32)
t = np.arange(int(0.18 * SR)) / SR; chime = (0.12 * np.sin(2*np.pi*880*t) * np.exp(-t*18)).astype(np.float32)
parts, rows, pos = [sil(0.5)], [], 0.5
for i, (code, en, nat, s1, s2, q) in enumerate(L, 1):
    if i > 1: parts += [sil(0.6), chime, sil(0.6)]; pos += 1.2 + len(chime) / SR
    start = pos
    for txt, gap in ((nat, 0.7), (s1, 0.55), (s2, 0.55), (q, 0.0)):
        a = v.synthesize(txt, code); parts += [a, sil(gap)]; pos += len(a) / SR + gap
    rows.append(dict(n=i, language=en, code=code, start=start, native=nat, s=[s1, s2, q]))
audio = np.concatenate(parts + [sil(0.8)])
wav = out / f"indic_tts_review_{tag}.wav"
with wave.open(str(wav), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "96k", str(wav.with_suffix(".mp3"))], check=True)
mmss = lambda s: f"{int(s//60)}:{int(s%60):02d}"
fonts = "'Noto Sans','Noto Sans Devanagari','Noto Sans Bengali','Noto Sans Tamil','Noto Sans Telugu','Noto Sans Gujarati','Noto Sans Gurmukhi','Noto Naskh Arabic','Noto Sans Kannada','Noto Sans Malayalam','Noto Sans Oriya',sans-serif"
trs = "".join(f"<tr><td>{r['n']}</td><td>{mmss(r['start'])}</td><td><b>{r['language']}</b><br>{html.escape(r['native'])}</td>"
              f"<td class=s{' dir=rtl' if r['code']=='ur_PK' else ''}>" + "<br>".join(f"{k}. {html.escape(x)}" for k, x in enumerate(r['s'], 1)) +
              "</td><td></td><td></td><td></td><td class=w></td></tr>" for r in rows)
doc = f"""<!doctype html><meta charset=utf-8><style>@page{{size:A4 landscape;margin:12mm}}body{{font-family:{fonts};font-size:10pt}}
table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #c9ced3;padding:4pt 6pt;vertical-align:top}}th{{background:#e9edf1;text-align:left}}
td.s{{font-size:10.5pt;line-height:1.4}}td.w{{width:140pt}}tr{{page-break-inside:avoid}}.how{{background:#f4f6f8;padding:6pt 10pt;margin-bottom:8pt}}</style>
<h2>Listening review ({tag}): experimental Indian-language text-to-speech</h2>
<div class=how>Audio: <b>{wav.with_suffix('.mp3').name}</b>. Each language: its name, two sentences, then <b>a question</b> (does the voice rise like a real question?).
Rate only languages you know. <b>Clarity</b> 1–5, <b>Natural</b> 1–5, <b>Question sounds like a question</b> yes/no, and note any mispronounced words.</div>
<table><tr><th>#</th><th>Time</th><th>Language</th><th>Sentences (3 = question)</th><th>Clarity</th><th>Natural</th><th>Question?</th><th>Mistakes / comments</th></tr>{trs}</table>"""
(out / f"review_sheet_{tag}.html").write_text(doc, encoding="utf-8")
subprocess.run(["chromium", "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={out / f'review_sheet_{tag}.pdf'}", str(out / f"review_sheet_{tag}.html")],
               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"review {tag}: {len(audio)/SR:.0f}s -> {wav.with_suffix('.mp3')}")
