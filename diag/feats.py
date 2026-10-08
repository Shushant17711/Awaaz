import sys, glob, numpy as np, soundfile as sf, librosa
def feats(x):
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=256))+1e-7
    f = librosa.fft_frequencies(sr=22050, n_fft=2048)
    rms = S.sum(0); act = rms > np.percentile(rms, 40)
    S = S[:, act]
    flat = lambda lo, hi: float(np.mean(librosa.feature.spectral_flatness(S=S[(f>=lo)&(f<hi)])))
    ltas = 20*np.log10(S.mean(1))
    # narrow-peak "tonality": LTAS minus its 300 Hz-smoothed version
    k = 27; sm = np.convolve(ltas, np.ones(k)/k, mode="same")
    peaky = float(np.mean(np.clip(ltas-sm, 0, None)[(f>1000)&(f<10000)]))
    e = lambda lo, hi: S[(f>=lo)&(f<hi)].__pow__(2).sum()
    return dict(flat_1_4k=flat(1000,4000), flat_4_8k=flat(4000,8000), peaky=peaky,
                hf_db=10*np.log10(e(5000,11025)/e(0,11025)), ltas=ltas)
