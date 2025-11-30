import numpy as np

from signal_processing import SignalProcessor, F_LOW, F_HIGH, TONE_MATRIX


class ToneAnalyzer:
    def __init__(self, processor: SignalProcessor, low_set=F_LOW, high_set=F_HIGH, mapping=TONE_MATRIX):
        self.proc = processor
        self.low_set = low_set
        self.high_set = high_set
        self.mapping = mapping

    def identify_digit(self, chunk, min_power=1e-4, peak_ratio=2.0, min_magnitude=0.01):
        if len(chunk) == 0:
            return None

        windowed = chunk * np.hamming(len(chunk))
        power_level = np.mean(windowed * windowed)
        if power_level < min_power:
            return None

        low_mags = np.array([self.proc.compute_tone_energy(windowed, f) for f in self.low_set])
        high_mags = np.array([self.proc.compute_tone_energy(windowed, f) for f in self.high_set])

        if low_mags.max() < 1e-8 and high_mags.max() < 1e-8:
            return None

        idx_l = int(np.argmax(low_mags))
        idx_h = int(np.argmax(high_mags))

        freq_l = float(self.low_set[idx_l])
        freq_h = float(self.high_set[idx_h])

        # Check absolute magnitude threshold - both peaks must be strong
        if low_mags[idx_l] < min_magnitude or high_mags[idx_h] < min_magnitude:
            return None

        avg_l = np.mean(low_mags) + 1e-12
        avg_h = np.mean(high_mags) + 1e-12

        # Both peaks must stand out from the average
        if low_mags[idx_l] < peak_ratio * avg_l or high_mags[idx_h] < peak_ratio * avg_h:
            return None

        # Additional validation: ensure the two peaks are relatively balanced
        # DTMF tones should have similar magnitudes for low and high frequencies
        ratio = max(low_mags[idx_l], high_mags[idx_h]) / (min(low_mags[idx_l], high_mags[idx_h]) + 1e-12)
        if ratio > 10.0:  # If one frequency is 10x stronger, likely not a valid DTMF
            return None

        digit = self.mapping.get((int(round(freq_l)), int(round(freq_h))))
        if digit is None:
            return None
        
        return (digit, freq_l, freq_h)
