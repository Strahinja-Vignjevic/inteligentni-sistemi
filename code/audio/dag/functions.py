import numpy as np
import librosa

TARGET_DURATION = 2.0

def pad_or_crop(y, sr):
    target = int(sr * TARGET_DURATION)

    if len(y) > target:
        start = np.random.randint(0, len(y) - target + 1)
        return y[start:start + target]

    if len(y) < target:
        return np.pad(y, (0, target - len(y)))

    return y


def normalize(y):
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def add_noise(y):
    noise_factor = np.random.uniform(0.001, 0.005)
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def random_gain(y):
    gain = np.random.uniform(0.8, 1.2)
    return y * gain


def time_stretch(y):
    rate = np.random.uniform(0.95, 1.05)
    return librosa.effects.time_stretch(y, rate=rate)


def small_pitch_shift(y, sr):
    # ±5 centi (~0.05 semitona)
    n_steps = np.random.uniform(-0.05, 0.05)
    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=n_steps
    )


def augment_audio(y, sr):
    y = normalize(y)
    y = pad_or_crop(y, sr)

    augmented = []

    augmented.append(y)

    augmented.append(add_noise(y))

    augmented.append(random_gain(y))

    augmented.append(
        pad_or_crop(
            time_stretch(y),
            sr
        )
    )

    augmented.append(
        small_pitch_shift(y, sr)
    )

    augmented.append(
        random_gain(
            add_noise(y)
        )
    )

    return augmented