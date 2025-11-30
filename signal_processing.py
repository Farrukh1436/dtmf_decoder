import numpy as np
from scipy.signal import butter, sosfiltfilt


TONE_MATRIX = {
    (697, 1209): "1",
    (697, 1336): "2",
    (697, 1477): "3",
    (770, 1209): "4",
    (770, 1336): "5",
    (770, 1477): "6",
    (852, 1209): "7",
    (852, 1336): "8",
    (852, 1477): "9",
    (941, 1336): "0",
}

F_LOW = np.array([697, 770, 852, 941], dtype=float)
F_HIGH = np.array([1209, 1336, 1477], dtype=float)


class SignalProcessor:
    def __init__(self, fs, order=6):
        self.fs = fs
        self.order = order

    def apply_bandpass(self, data, f1=600, f2=1600):
        sos = butter(self.order, [f1, f2], btype="band", fs=self.fs, output="sos")
        return sosfiltfilt(sos, data)

    def compute_tone_energy(self, samples, target_freq):
        n = len(samples)
        if n == 0:
            return 0.0

        normalized_k = n * target_freq / self.fs
        ang = 2.0 * np.pi * normalized_k / n
        multiplier = 2.0 * np.cos(ang)

        state1 = 0.0
        state2 = 0.0

        for val in samples:
            temp = val + multiplier * state1 - state2
            state2, state1 = state1, temp

        magnitude = state2 ** 2 + state1 ** 2 - multiplier * state1 * state2
        return np.sqrt(abs(magnitude))
