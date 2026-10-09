
import numpy as np


def add_noise(signal, noise_level=10):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise


def moving_average(signal, window_size=10):
    window = np.ones(window_size) / window_size

    padded = np.pad(
        signal,
        (window_size // 2, window_size - 1 - window_size // 2),
        mode="edge"
    )

    return np.convolve(padded, window, mode="valid")
