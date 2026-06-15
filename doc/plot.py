import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# ucitaj jedan spektrogram
mel = np.load("C.npy")

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mel,
    sr=22050,
    x_axis='time',
    y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spektrogram — C5')
plt.tight_layout()
plt.savefig('mel_spektrogram_C5.png', dpi=150)
plt.show()