import librosa
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("sax_model.h5")

SR = 22050
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

x = preprocess("test2.wav")
x = x[np.newaxis, ..., np.newaxis]

pred = model.predict(x, verbose=0)[0]

best_idx = np.argmax(pred)

print("Predicted note:", idx_to_label[best_idx])
print("Confidence:", f"{pred[best_idx]:.2%}")

print("\nTop 3:")
for i in np.argsort(pred)[-3:][::-1]:
    print(f"{idx_to_label[i]} -> {pred[i]:.2%}")