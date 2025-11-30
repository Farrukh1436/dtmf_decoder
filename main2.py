import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks, windows

# ===============================
# 1. DTMF table and mapping
# ===============================

dtmf_low = [697, 770, 852, 941]
dtmf_high = [1209, 1336, 1477]

dtmf_map = {
    (697, 1209): '1',
    (697, 1336): '2',
    (697, 1477): '3',
    (770, 1209): '4',
    (770, 1336): '5',
    (770, 1477): '6',
    (852, 1209): '7',
    (852, 1336): '8',
    (852, 1477): '9',
    (941, 1336): '0',
    (697, 1633): 'A',
    (770, 1633): 'B',
    (852, 1633): 'C',
    (941, 1633): 'D',
    # (941, 1209): '*',
    # (941, 1477): '#',
}

# ===============================
# 2. Helper: detect active regions (non-silence)
# ===============================

def detect_active_regions(x, fs, frame_ms=20, thresh_ratio=0.1, min_duration_ms=50):
    """
    Split signal into active segments using simple energy-based VAD.
    Returns list of (start_sample, end_sample) tuples.
    """
    frame_len = int(fs * frame_ms / 1000)
    if frame_len <= 0:
        frame_len = 1

    x = x.astype(float)
    max_abs = np.max(np.abs(x)) + 1e-12
    x = x / max_abs

    num_frames = len(x) // frame_len
    energies = np.zeros(num_frames)

    for i in range(num_frames):
        frame = x[i * frame_len:(i + 1) * frame_len]
        energies[i] = np.mean(frame ** 2)

    thresh = thresh_ratio * np.max(energies)
    active = energies > thresh

    regions = []
    in_region = False
    start_f = 0

    for i, a in enumerate(active):
        if a and not in_region:
            in_region = True
            start_f = i
        elif not a and in_region:
            in_region = False
            end_f = i
            s = start_f * frame_len
            e = end_f * frame_len
            if (e - s) >= fs * min_duration_ms / 1000:
                regions.append((s, e))

    if in_region:
        e = num_frames * frame_len
        s = start_f * frame_len
        if (e - s) >= fs * min_duration_ms / 1000:
            regions.append((s, e))

    return regions

# ===============================
# 3. Find main two tones in one segment (using FFT)
# ===============================

def find_two_tones(segment, fs):
    """
    Return (low_freq, high_freq) for one DTMF digit segment using FFT.
    """
    if len(segment) == 0:
        return None, None

    w = windows.hann(len(segment))
    x = segment * w

    N = len(x)
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mag = np.abs(fft_vals)

    # Look only in DTMF range ~600–1700 Hz
    mask = (freqs >= 600) & (freqs <= 1700)
    freqs_sub = freqs[mask]
    mag_sub = mag[mask]

    if len(freqs_sub) == 0:
        return None, None

    # Peaks
    peaks, props = find_peaks(mag_sub, height=np.max(mag_sub) * 0.3, distance=10)
    if len(peaks) == 0:
        return None, None

    # Sort peaks by magnitude (largest first)
    peaks_sorted = sorted(peaks, key=lambda p: mag_sub[p], reverse=True)
    # Take top few
    top_freqs = [freqs_sub[p] for p in peaks_sorted[:5]]

    low_cand = [f for f in top_freqs if f < 1100]
    high_cand = [f for f in top_freqs if f >= 1100]

    if not low_cand or not high_cand:
        return None, None

    low_freq = low_cand[0]
    high_freq = high_cand[0]
    return low_freq, high_freq

# ===============================
# 4. Match to nearest DTMF standard freq
# ===============================

def nearest_dtmf_freq(f, dtmf_list, tol=4.0):
    diffs = [abs(f - d) for d in dtmf_list]
    idx = int(np.argmin(diffs))
    if diffs[idx] <= tol:
        return dtmf_list[idx]
    return None

def decode_digit(segment, fs):
    low_f, high_f = find_two_tones(segment, fs)
    if low_f is None or high_f is None:
        return None

    low_std = nearest_dtmf_freq(low_f, dtmf_low)
    high_std = nearest_dtmf_freq(high_f, dtmf_high)

    if low_std is None or high_std is None:
        return None

    return dtmf_map.get((low_std, high_std), None)

# ===============================
# 5. Main decoder: wav -> digits
# ===============================

def decode_dtmf_wav(filename, plot_debug=True):
    fs, x = wavfile.read(filename)

    # Use one channel if stereo
    if x.ndim > 1:
        x = x[:, 0]

    # Plot waveform
    if plot_debug:
        t = np.arange(len(x)) / fs
        plt.figure(figsize=(12, 4))
        plt.plot(t, x)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform: {filename}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Detect active regions (tone segments)
    regions = detect_active_regions(x, fs)
    print("Detected active regions (sample indices):")
    for i, (s, e) in enumerate(regions):
        print(f"  Region {i}: {s} -> {e} (duration {(e-s)/fs:.3f} s)")

    digits = []
    for idx, (s, e) in enumerate(regions):
        segment = x[s:e]
        d = decode_digit(segment, fs)
        if plot_debug:
            # Plot FFT of this segment
            w = windows.hann(len(segment))
            seg_win = segment * w
            N = len(seg_win)
            fft_vals = np.fft.rfft(seg_win)
            freqs = np.fft.rfftfreq(N, d=1/fs)
            mag = np.abs(fft_vals)

            plt.figure(figsize=(10, 3))
            plt.plot(freqs, mag)
            plt.xlim(0, 2000)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.title(f"FFT of region {idx}, decoded digit: {d}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        digits.append(d if d is not None else '?')

    phone_number = ''.join(digits)
    return phone_number

# ===============================
# 6. Run
# ===============================

if __name__ == "__main__":
    wav_name = "Project1_v4.wav"   # change if your file has different name
    wav_name = "1.wav"
    number = decode_dtmf_wav(wav_name, plot_debug=True)
    print("\nDecoded phone number:", number)