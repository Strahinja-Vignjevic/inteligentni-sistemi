import os
import numpy as np

DATASET_PATH = "test-spectograms"

X = []
Y = []

label_map = {}
label_index = 0

for note in os.listdir(DATASET_PATH):

    note_path = os.path.join(DATASET_PATH, note)

    if not os.path.isdir(note_path):
        continue

    # dodeli broj svakoj noti
    if note not in label_map:
        label_map[note] = label_index
        label_index += 1

    label = label_map[note]

    print("Loading:", note)

    for file in os.listdir(note_path):

        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(note_path, file)

        spectrogram = np.load(file_path)

        X.append(spectrogram)
        Y.append(label)

# pretvori u numpy
X = np.array(X)
Y = np.array(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# sačuvaj
np.save("X.npy", X)
np.save("Y.npy", Y)
np.save("label_map.npy", label_map)