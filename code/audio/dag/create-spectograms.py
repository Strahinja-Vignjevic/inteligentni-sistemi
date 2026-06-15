import os
import librosa
import numpy as np

INPUT_DIR = "test-normalize-data"
OUTPUT_DIR = "test-spectograms"

SR = 22050
N_MELS = 128
MAX_LEN = 128

for note in os.listdir(INPUT_DIR):

    note_path = os.path.join(INPUT_DIR, note)

    if not os.path.isdir(note_path):
        continue

    # napravi isti podfolder u output-u
    out_note_path = os.path.join(OUTPUT_DIR, note)
    os.makedirs(out_note_path, exist_ok=True)

    for file in os.listdir(note_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(note_path, file)

        y, sr = librosa.load(file_path, sr=SR)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        if mel.shape[1] < MAX_LEN:
            mel = np.pad(
                mel,
                ((0, 0), (0, MAX_LEN - mel.shape[1]))
            )
        else:
            mel = mel[:, :MAX_LEN]

        out_file = os.path.join(
            out_note_path,
            file.replace(".wav", ".npy")
        )

        np.save(out_file, mel)

        print("Saved:", out_file)