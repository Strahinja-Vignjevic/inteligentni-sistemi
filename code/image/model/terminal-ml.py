import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import joblib

# =====================
# MODEL
# =====================
model = joblib.load("../../../dataset/audio/saved_models/random_forest.pkl")
label_map = np.load("../../../dataset/audio/label_map_ml.npy", allow_pickle=True).item()
idx_to_label = {v: k for k, v in label_map.items()}

SR = 22050
DURATION = 3

# =====================
# EXTRACT FEATURES
# =====================
def extract_features(path):
    y, sr = librosa.load(path, sr=SR)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A2"),
        fmax=librosa.note_to_hz("E6")
    )
    pitch = np.nanmean(f0)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return np.array([pitch, rms, centroid, zcr, *mfcc_mean])

# =====================
# RECORD MIC
# =====================
def record():
    print("\n🎤 Snimam... sviraj sax!")

    audio = sd.rec(int(DURATION * SR),
                   samplerate=SR,
                   channels=1,
                   dtype="float32")

    sd.wait()

    write("temp.wav", SR, audio)
    print("✅ Snimljeno")

# =====================
# PREDICTION
# =====================
def predict():
    features = extract_features("temp.wav").reshape(1, -1)

    pred_idx = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    top3 = np.argsort(proba)[-3:][::-1]

    print("\n🎷 Top predictions:")
    for i in top3:
        print(f"{idx_to_label[i]} -> {proba[i]:.2%}")

# =====================
# LOOP MODE
# =====================
while True:
    input("\nENTER za snimanje (CTRL+C za exit)")
    record()
    predict()