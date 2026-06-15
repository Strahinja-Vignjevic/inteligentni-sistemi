import os
import librosa
import soundfile as sf
from functions import augment_audio

INPUT_DIR = "test-data"
OUTPUT_DIR = "augmented_dataset"

SR = 22050
TARGET_DURATION = 2.0

def process_file(path):
    y, sr = librosa.load(path, sr=SR)
    return augment_audio(y, sr)


for note_folder in os.listdir(INPUT_DIR):

    input_path = os.path.join(INPUT_DIR, note_folder)

    if not os.path.isdir(input_path):
        continue

    output_path = os.path.join(OUTPUT_DIR, note_folder)
    os.makedirs(output_path, exist_ok=True)

    print(f"Processing {note_folder}")

    for file in os.listdir(input_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(input_path, file)

        augmented_versions = process_file(file_path)

        base = os.path.splitext(file)[0]

        for i, audio in enumerate(augmented_versions):

            out_file = os.path.join(
                output_path,
                f"{base}_aug_{i}.wav"
            )

            sf.write(out_file, audio, SR)