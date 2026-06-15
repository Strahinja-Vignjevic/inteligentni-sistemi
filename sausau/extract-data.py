import librosa
import numpy as np
import soundfile as sf
import os

y, sr = librosa.load("overTheRainbow.mp4", sr=22050)

rms = librosa.feature.rms(y=y)[0]
rms = rms / np.max(rms)

threshold = 0.05

active = rms > threshold

segments = []
start = None

for i, a in enumerate(active):
    if a and start is None:
        start = i
    elif not a and start is not None:
        segments.append((start, i))
        start = None

if start is not None:
    segments.append((start, len(active)))

hop = 512
out_dir = "newmel"
os.makedirs(out_dir, exist_ok=True)

for i, (s, e) in enumerate(segments):
    audio = y[s*hop:e*hop]

    if len(audio) > sr * 5:  # minimum 5 sec
        sf.write(f"{out_dir}/melody_{i}.wav", audio, sr)