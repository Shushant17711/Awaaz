import glob, numpy as np, soundfile as sf, librosa
from feats import feats
f = librosa.fft_frequencies(sr=22050, n_fft=2048)
def resid(p):
    l = feats(sf.read(p, dtype="float32")[0])["ltas"]; sm = np.convolve(l, np.ones(27)/27, mode="same"); return l-sm
for lang in ["hi_IN","ta_IN","kn_IN","or_IN","as_IN","bn_BD"]:
    print(lang, *(f"{lane}={np.mean([np.mean(np.clip(resid(p),0,None)[(f>1000)&(f<10000)]) for p in glob.glob(f'v2/{lang}_*_{lane}.wav')]):.2f}" for lane in ("teacher","voconly","full")))
for tag in sys.argv[1:] or ("v1","v2"):
  for lane in ("voconly","full"):
    ex = np.mean([resid(p) - resid(p.replace(lane,"teacher")) for p in glob.glob(f"{tag}/*_{lane}.wav")], 0)
    top = np.argsort(ex)[::-1][:12]
    print(tag, lane, "excess tonal peaks Hz:", sorted((int(f[i]), round(float(ex[i]),1)) for i in top))
