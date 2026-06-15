import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import tensorflow as tf

# =====================
# MODEL
# =====================
model = tf.keras.models.load_model("sax_model.h5")

SR = 22050
DURATION = 3
N_MELS = 128
MAX_LEN = 128

idx_to_label = {
    0: "A",
    1: "A#",
    2: "B",
    3: "C",
    4: "C#",
    5: "C2",
    6: "D",
    7: "D#",
    8: "E",
    9: "F",
    10: "F#",
    11: "G",
    12: "G#"
}


# =====================
# PREPROCESS (ISTO KAO FLASK)
# =====================
def preprocess(path):
    y, sr = librosa.load(path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    return mel


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
    x = preprocess("temp.wav")
    x = x[np.newaxis, ..., np.newaxis]

    pred = model.predict(x)[0]

    top3 = np.argsort(pred)[-3:][::-1]

    print("\n🎷 Top predictions:")
    for i in top3:
        print(f"{idx_to_label[i]} -> {pred[i]:.2%}")
# =====================
# LOOP MODE
# =====================
while True:
    input("\nENTER za snimanje (CTRL+C za exit)")
    record()
    predict()