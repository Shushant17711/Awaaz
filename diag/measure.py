import sys, glob, numpy as np, soundfile as sf, librosa
from feats import feats
for tag in sys.argv[1:]:
    for lane in ("voconly","full"):
        rows = [feats(sf.read(p, dtype="float32")[0]) for p in sorted(glob.glob(f"{tag}/*_{lane}.wav"))]
        print(f"{tag:6s} {lane:8s}", *(f"{k}={np.mean([r[k] for r in rows]):.3f}" for k in ("flat_1_4k","flat_4_8k","peaky","hf_db")))
    d=[]
    for p in sorted(glob.glob(f"{tag}/*_teacher.wav")):
        a=sf.read(p,dtype="float32")[0]; b=sf.read(p.replace("teacher","voconly"),dtype="float32")[0]; n=min(len(a),len(b))
        A=np.log10(np.abs(librosa.stft(a[:n],n_fft=1024))**2+1e-8); B=np.log10(np.abs(librosa.stft(b[:n],n_fft=1024))**2+1e-8)
        d.append(np.mean(np.sqrt(np.mean((A-B)**2,0))))
    print(f"{tag:6s} voconly LSD vs teacher {np.mean(d):.3f}")
