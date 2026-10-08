# v2 run report (2026-10-06 18:09:44)

## Verdict
```
{"promote": false, "mean_v2": 0.2802, "mean_current": 0.2726, "worst_lang_regression": 0.0717, "per_lang": {"as_IN": [0.501, 0.447], "bn_BD": [0.305, 0.295], "gu_IN": [0.279, 0.268], "hi_IN": [0.105, 0.098], "kn_IN": [0.19, 0.178], "ml_IN": [0.547, 0.618], "mr_IN": [0.217, 0.237], "or_IN": [0.335, 0.337], "pa_IN": [0.416, 0.441], "ta_IN": [0.151, 0.223], "te_IN": [0.132, 0.139], "ur_PK": [0.092, 0.081]}}
```
Gate: promote only if mean Whisper CER (12 langs x 32 held-out sentences) <= current + 0.003 and no language (excluding Malayalam) worse by > 0.06.

## Teachers (continued +10k steps; kept only if not worse)
```
or keep-v1 {"old": 0.3322, "new": 0.3941, "accept": false}
as v2 {"old": 0.5273, "new": 0.5144, "accept": true}
kn v2 {"old": 0.1939, "new": 0.1506, "accept": true}
pa keep-v1 {"old": 0.2821, "new": 0.3636, "accept": false}
```
## Question intonation (end-of-question pitch / first-half pitch; higher = more rising)
```
multilingual/v2/question_v1.txt:release/multilingual: mean question end-pitch ratio 0.592
multilingual/v2/question_v2.txt:release/candidate-v2: mean question end-pitch ratio 0.620
```
## Runtime parity
```
PASS hi_IN: frames 189 vs 189, mel max|Δ| 7.63e-06, wav corr 1.000000
PASS ta_IN: frames 140 vs 140, mel max|Δ| 1.14e-05, wav corr 1.000000
PASS bn_BD: frames 184 vs 184, mel max|Δ| 1.29e-05, wav corr 1.000000
PASS ur_PK: frames 124 vs 124, mel max|Δ| 9.54e-06, wav corr 1.000000
PASS as_IN: frames 146 vs 146, mel max|Δ| 3.81e-06, wav corr 1.000000
package weights: 4.80 MB, parameters: 2,399,330
```
## Listen
- review/indic_tts_review_v2.mp3 (+ review_sheet_v2.pdf): each language now ends with a question.
- release/multilingual (if promoted) or release/candidate-v2 (if not).
