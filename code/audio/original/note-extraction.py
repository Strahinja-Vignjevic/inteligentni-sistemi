import librosa
import numpy as np
import soundfile as sf
import os

input_dir = "recordings"
output_dir = "dataset"

frame_length = 2048
hop_length = 512

# mamba

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.endswith(".m4a"):
        continue

    note_name = file.replace(".m4a", "")
    file_path = os.path.join(input_dir, file)

    print(f"Processing: {note_name}")

    y, sr = librosa.load(file_path, sr=None)

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    threshold = rms.max() * 0.2
    active = rms > threshold

    segments = []
    start = None

    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(active)))

    out_folder = os.path.join(output_dir, note_name)
    os.makedirs(out_folder, exist_ok=True)

    count = 0

    for start_frame, end_frame in segments:
        start_sample = start_frame * hop_length
        end_sample = min(len(y), end_frame * hop_length)

        segment = y[start_sample:end_sample]

        if len(segment) > sr * 0.2:
            sf.write(
                os.path.join(out_folder, f"{note_name}_{count:03d}.wav"),
                segment,
                sr
            )
            count += 1

    print(f"{note_name}: {count} samples extracted")