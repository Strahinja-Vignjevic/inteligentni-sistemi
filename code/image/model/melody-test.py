import librosa
import numpy as np
import tensorflow as tf
from music21 import stream, note as m21note, meter, tempo
import subprocess

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
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    return mel


# -----------------------------
# LOAD AUDIO
# -----------------------------
y, sr = librosa.load("../../../sausau/rainbow/melody_8.wav", sr=SR)


# -----------------------------
# PITCH TRACKING (GLAVNI DEO)
# -----------------------------
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("A2"),
    fmax=librosa.note_to_hz("E6")
)

notes_raw = []

for freq in f0:
    if np.isnan(freq) or freq <= 0:
        continue  # ❌ IZBACUJE potpuno

    notes_raw.append(librosa.hz_to_note(freq))

# -----------------------------
# SEGMENTATION (PROMENA FREKVENCIJE)
# -----------------------------
segments = []
start = 0

for i in range(1, len(notes_raw)):
    if notes_raw[i] != notes_raw[i - 1]:
        segments.append((start, i))
        start = i

segments.append((start, len(notes_raw)))


# -----------------------------
# CONVERT SEGMENTS → AUDIO → MODEL
# -----------------------------
notes = []

hop_length = 512

for s, e in segments:

    start_sample = s * hop_length
    end_sample = e * hop_length

    start_time = start_sample / sr
    end_time = end_sample / sr
    duration = end_time - start_time

    segment = y[start_sample:end_sample]

    if len(segment) < sr * 0.1:
        continue

    mel = preprocess_audio(segment, sr)

    x = mel[np.newaxis, ..., np.newaxis]

    pred = model.predict(x, verbose=0)[0]

    note = idx_to_label[np.argmax(pred)]
    confidence = float(np.max(pred))

    notes.append((note, confidence, duration))

# -----------------------------
# OUTPUT
# -----------------------------
print("\nDETECTED MELODY:\n")

for n, c, d in notes:
    print(f"{n}  ({c:.2%})  duration: {d:.2f}s")

print("\nDETECTED MELODY (MUSIC VALUES):\n")

music_notes = []
BPM = 60

for n, c, d in notes:

    beat_value = d * (BPM / 60)

    if beat_value < 0.3:
        note_type = "16th"
    elif beat_value < 0.7:
        note_type = "8th"
    elif beat_value < 1.5:
        note_type = "quarter"
    elif beat_value < 2.5:
        note_type = "half"
    else:
        note_type = "whole"

    music_notes.append((n, note_type, beat_value))

    print(f"{n} -> {note_type} ({beat_value:.2f} beats)")

def quantize(duration):
    if duration < 0.3:
        return 0.25   # 16th note
    elif duration < 0.7:
        return 0.5    # 8th note
    elif duration < 1.3:
        return 1.0    # quarter note
    elif duration < 2.5:
        return 2.0    # half note
    else:
        return 4.0    # whole note
    
score = stream.Stream()

score.append(tempo.MetronomeMark(number=BPM))
score.append(meter.TimeSignature('4/4'))

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

for n, c, d in notes:

    mapped_note = note_map.get(n, "C5")
    n_obj = m21note.Note(mapped_note)

    # convert seconds → beats
    beat_value = d * (BPM / 60)

    # QUANTIZE (KLJUČNO)
    q_len = quantize(beat_value)

    n_obj.quarterLength = q_len

    score.append(n_obj)

score.makeMeasures(inPlace=True)

score.write('musicxml', fp='sax_output.xml')
score.write('midi', fp='sax_output.mid')


subprocess.Popen([
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    "sax_output.xml"
])
