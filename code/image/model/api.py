from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os

app = Flask(__name__)

SR = 22050
N_MELS = 128
MAX_LEN = 128

model = tf.keras.models.load_model("sax_model.h5")
label_map = np.load("label_map.npy", allow_pickle=True).item()
inv_map = {v: k for k, v in label_map.items()}

def preprocess(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)

        x = preprocess(tmp.name)
        x = x[np.newaxis, ..., np.newaxis]

        pred = model.predict(x)
        idx = np.argmax(pred)

        os.remove(tmp.name)

        return jsonify({
            "note": inv_map[idx],
            "class": int(idx)
        })

if __name__ == "__main__":
    app.run(debug=True)