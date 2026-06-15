import librosa
import numpy as np
import tensorflow as tf
import sounddevice as sd
import queue
import subprocess

from music21 import stream, note as m21note, meter, tempo

# -----------------------------
# MODEL
# -----------------------------
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

note_map = {
    "C": "C5",
    "C#": "C#5",
    "D": "D5",
    "D#": "D#5",
    "E": "E5",
    "F": "F5",
    "F#": "F#5",
    "G": "G5",
    "G#": "G#5",
    "A": "A5",
    "A#": "A#5",
    "B": "B5",
    "C2": "C6"
}

# -----------------------------
# RECORD AUDIO
# -----------------------------
audio_queue = queue.Queue()
recording = []

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

input("PRESS ENTER TO START RECORDING...")

stream_audio = sd.InputStream(
    samplerate=SR,
    channels=1,
    callback=callback
)

stream_audio.start()

input("RECORDING... PRESS ENTER TO STOP...")

stream_audio.stop()
stream_audio.close()

while not audio_queue.empty():
    recording.append(audio_queue.get())

audio = np.concatenate(recording, axis=0).flatten()

print("AUDIO CAPTURED")

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_audio(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(
            mel,
            ((0, 0), (0, MAX_LEN - mel.shape[1]))
        )
    else:
        mel = mel[:, :MAX_LEN]

    return mel

# -----------------------------
# PITCH TRACKING
# -----------------------------
f0, _, _ = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz("A2"),
    fmax=librosa.note_to_hz("E6")
)

notes_raw = []

for freq in f0:
    if np.isnan(freq) or freq <= 0:
        continue

    notes_raw.append(librosa.hz_to_note(freq))

# -----------------------------
# SEGMENTATION
# -----------------------------
segments = []
start = 0

for i in range(1, len(notes_raw)):
    if notes_raw[i] != notes_raw[i - 1]:
        segments.append((start, i))
        start = i

segments.append((start, len(notes_raw)))

# -----------------------------
# CLASSIFICATION
# -----------------------------
notes = []

hop_length = 512

for s, e in segments:

    start_sample = s * hop_length
    end_sample = e * hop_length

    segment = audio[start_sample:end_sample]

    if len(segment) < SR * 0.1:
        continue

    mel = preprocess_audio(segment, SR)

    x = mel[np.newaxis, ..., np.newaxis]

    pred = model.predict(
        x,
        verbose=0
    )[0]

    note = idx_to_label[np.argmax(pred)]
    confidence = float(np.max(pred))
    duration = len(segment) / SR

    notes.append(
        (note, confidence, duration)
    )

# -----------------------------
# PRINT RESULT
# -----------------------------
print("\nDETECTED MELODY\n")

for n, c, d in notes:
    print(
        f"{n} ({c:.2%}) {d:.2f}s"
    )

# -----------------------------
# QUANTIZE
# -----------------------------
def quantize(duration):
    if duration < 0.3:
        return 0.25
    elif duration < 0.7:
        return 0.5
    elif duration < 1.3:
        return 1.0
    elif duration < 2.5:
        return 2.0
    else:
        return 4.0

# -----------------------------
# MUSICXML
# -----------------------------
BPM = 60

score = stream.Stream()

score.append(
    tempo.MetronomeMark(number=BPM)
)

score.append(
    meter.TimeSignature("4/4")
)

for n, c, d in notes:

    mapped_note = note_map[n]

    note_obj = m21note.Note(
        mapped_note
    )

    beat_value = d * (BPM / 60)

    note_obj.quarterLength = quantize(
        beat_value
    )

    score.append(note_obj)

score.makeMeasures(inPlace=True)

score.write(
    "musicxml",
    fp="sax_output.xml"
)

# -----------------------------
# OPEN MUSESCORE
# -----------------------------
subprocess.Popen([
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    "sax_output.xml"
])

print("DONE")