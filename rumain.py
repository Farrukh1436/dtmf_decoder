#!/usr/bin/env python3
"""
DTMF Signal Decoder (single-file, class-free)

This file is a one-to-one functional rewrite of the original implementation,
but reorganized into clear functions and module-level constants so it's easy
for juniors to read and for seniors to maintain.

Functionality is intentionally NOT changed:
 - same frequency map
 - same energy-based segmentation logic
 - same FFT-based decoding logic and thresholds
 - same plotting behavior (matplotlib)
 - CLI entrypoint preserved

Run:
    python dtmf_decoder.py <path_to_wav_file>

Author: Senior-style rewrite (preserves original behavior)
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import binary_dilation  # kept because original imported it; not used but harmless
import matplotlib.pyplot as plt
import os
import sys

# -------------------------
# Configuration (module-level)
# -------------------------
# Standard DTMF low and high tone frequencies (Hz)
LOW_FREQS: List[int] = [697, 770, 852, 941]
HIGH_FREQS: List[int] = [1209, 1336, 1477, 1633]

# Tolerances and timing (same semantics as original DTMFConfig)
FREQ_TOLERANCE: float = 10.0               # tolerance (Hz) for matching FFT peaks to DTMF grid
MIN_SIGNAL_DURATION_MS: int = 40           # minimum tone duration to accept (milliseconds)
MIN_SILENCE_DURATION_MS: int = 20          # minimum silence to separate keys (milliseconds) (kept for clarity)
SILENCE_THRESHOLD: float = 0.05            # not directly used in segmentation (kept for parity)
MAX_SAME_DIGIT_GAP_MS: int = 50            # max gap to deduplicate duplicate detections (milliseconds)

# FFT / SNR thresholds (kept as in original)
MIN_SNR_RATIO: float = 10.0                # require peaks to be >= 10x noise floor
MIN_PEAK_RELATIVE_TO_MAX: float = 0.1      # require peaks to be at least 10% of global max spectrum

# DTMF mapping (pair of (low_freq, high_freq) -> character)
def build_dtmf_map() -> Dict[Tuple[int, int], str]:
    """
    Build mapping from frequency-pair (low, high) to DTMF character.
    This mirrors the classic telephone keypad layout including A-D keys.
    """
    keys = [
        ['1', '2', '3', 'A'],
        ['4', '5', '6', 'B'],
        ['7', '8', '9', 'C'],
        ['*', '0', '#', 'D']
    ]
    mapping: Dict[Tuple[int, int], str] = {}
    for r, low in enumerate(LOW_FREQS):
        for c, high in enumerate(HIGH_FREQS):
            mapping[(low, high)] = keys[r][c]
    return mapping

DTMF_MAP = build_dtmf_map()

# Processed audio buffer (tones-only) saved for potential plotting or inspection
processed_audio: Optional[np.ndarray] = None

# -------------------------
# Utility functions
# -------------------------
def load_audio(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Load a WAV file and return (sample_rate, normalized_audio).
    Normalization: output floats in [-1.0, 1.0].
    Stereo files are converted to mono by averaging channels.

    Raises:
        FileNotFoundError if the file doesn't exist.
        Any errors from scipy.io.wavfile.read will propagate.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    sample_rate, data = wavfile.read(file_path)

    # Convert integer PCM to float in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        # 8-bit WAVs are unsigned with 128 as zero-level
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        # If already float, assume it's in -1..1 or similar; just cast to float64
        data = data.astype(np.float64)

    # If stereo (or multi-channel), convert to mono by averaging channels
    if data.ndim > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)

    return sample_rate, data


def _find_nearest_freq(target: float, valid_freqs: List[int]) -> Optional[int]:
    """
    Given a target frequency (from FFT peak), return the nearest
    valid DTMF frequency (from valid_freqs) within FREQ_TOLERANCE.
    If no frequency is within tolerance, return None.

    This is a small helper to discretize continuous FFT peaks to the
    fixed DTMF frequency grid.
    """
    best_match: Optional[int] = None
    min_diff = float('inf')

    for freq in valid_freqs:
        diff = abs(target - freq)
        if diff <= FREQ_TOLERANCE and diff < min_diff:
            min_diff = diff
            best_match = freq

    return best_match


def plot_fft(segment: np.ndarray, fs: int, digit: Optional[str], index: int) -> None:
    """
    Plot the magnitude spectrum (FFT) of the given audio segment.
    This is used for debugging and visual verification of detected peaks.
    """
    if segment.size == 0:
        return

    N = len(segment)
    window = np.hamming(N)
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(N, 1 / fs)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, spectrum)
    plt.title(f"FFT of region {index}, decoded digit: {digit}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.xlim(0, 2000)  # DTMF frequencies lie well below 2000 Hz
    plt.show()


# -------------------------
# Core signal processing
# -------------------------
def decode_segment(audio_segment: np.ndarray, sample_rate: int) -> Optional[str]:
    """
    Decode a single audio segment containing a DTMF tone.
    Steps:
      1. Apply Hamming window to reduce spectral leakage.
      2. Compute FFT (real-valued).
      3. Find strongest peaks in low and high DTMF bands.
      4. Apply noise rejection using median noise floor and SNR constraints.
      5. Match peaks to nearest DTMF grid frequencies and return mapped character.

    Returns:
      A single character (e.g. '5', '#', 'A') if decoded, otherwise None.
    """
    if len(audio_segment) == 0:
        return None

    # Window the signal to reduce spectral leakage
    windowed = audio_segment * np.hamming(len(audio_segment))

    # Select N for FFT. The original used 4096 as baseline but increased N when segment was larger.
    N = 4096
    if len(windowed) > N:
        N = len(windowed)

    spectrum = np.abs(np.fft.rfft(windowed, n=N))
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # Masks for the low and high DTMF bands (conservative ranges)
    low_band_mask = (freqs >= 600) & (freqs <= 1000)
    high_band_mask = (freqs >= 1100) & (freqs <= 1700)

    # Safety: ensure the masks actually cover some bins
    if not np.any(low_band_mask) or not np.any(high_band_mask):
        return None

    # Extract peak in low band
    low_spectrum = spectrum * low_band_mask
    low_idx = np.argmax(low_spectrum)
    low_peak = freqs[low_idx]
    low_mag = spectrum[low_idx]

    # Extract peak in high band
    high_spectrum = spectrum * high_band_mask
    high_idx = np.argmax(high_spectrum)
    high_peak = freqs[high_idx]
    high_mag = spectrum[high_idx]

    # Noise estimation within the DTMF frequency range
    dtmf_mask = (freqs >= 600) & (freqs <= 1700)
    noise_floor = np.median(spectrum[dtmf_mask])

    # Require both peaks to be significantly above noise floor (SNR check)
    if low_mag < noise_floor * MIN_SNR_RATIO or high_mag < noise_floor * MIN_SNR_RATIO:
        return None

    # Also require each peak to be at least some fraction of the global maximum to avoid picking tiny peaks
    global_max = np.max(spectrum)
    if low_mag < global_max * MIN_PEAK_RELATIVE_TO_MAX or high_mag < global_max * MIN_PEAK_RELATIVE_TO_MAX:
        return None

    # Map the measured peaks to the nearest valid DTMF frequency (within tolerance)
    matched_low = _find_nearest_freq(low_peak, LOW_FREQS)
    matched_high = _find_nearest_freq(high_peak, HIGH_FREQS)

    if matched_low is not None and matched_high is not None:
        return DTMF_MAP.get((matched_low, matched_high))
    return None


def segment_audio(audio_data: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
    """
    Segment the full audio stream into regions likely containing DTMF tones.

    Method:
      - Compute RMS energy over short frames (20 ms frames with 10 ms hop).
      - Build an adaptive threshold from median and max frame energy.
      - Mark frames above threshold as 'active' and join into contiguous segments.
      - Enforce a minimum duration for a valid segment (MIN_SIGNAL_DURATION_MS).
      - Store a 'processed' audio version with only the detected segments (kept for plotting).

    Returns:
      A list of (start_sample, end_sample) tuples, in sample indices.
    """
    global processed_audio

    frame_length = int(sample_rate * 0.020)  # 20 ms
    hop_length = int(sample_rate * 0.010)    # 10 ms

    # Edge case: if audio is shorter than a single frame, handle gracefully
    if len(audio_data) < frame_length:
        print("[-] Audio shorter than one frame; no segments found.")
        processed_audio = np.zeros_like(audio_data)
        return []

    # Compute number of frames (floor division to ensure frames fully inside signal)
    num_frames = (len(audio_data) - frame_length) // hop_length + 1
    energy = np.zeros(num_frames, dtype=np.float64)

    # RMS energy per frame
    for i in range(num_frames):
        s = i * hop_length
        e = s + frame_length
        frame = audio_data[s:e]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    max_energy = float(np.max(energy))
    median_energy = float(np.median(energy))

    if max_energy == 0.0:
        # Entire file is silent
        print("[-] Warning: Audio file appears to be completely silent.")
        processed_audio = np.zeros_like(audio_data)
        return []

    # Adaptive threshold tuned to reject noise but accept tones
    threshold = median_energy + 0.20 * (max_energy - median_energy)

    print(f"[-] Max amplitude: {max_energy:.4f}")
    print(f"[-] Dynamic silence threshold: {threshold:.4f}")

    is_active = energy > threshold

    # Convert contiguous active frames to sample-index segments
    segments: List[Tuple[int, int]] = []
    in_segment = False
    seg_start_sample = 0

    min_samples = int((MIN_SIGNAL_DURATION_MS / 1000.0) * sample_rate)

    for i in range(len(is_active)):
        if is_active[i] and not in_segment:
            # segment start (convert frame index to sample index)
            in_segment = True
            seg_start_sample = i * hop_length
        elif not is_active[i] and in_segment:
            # segment end
            in_segment = False
            seg_end_sample = i * hop_length
            duration = seg_end_sample - seg_start_sample
            if duration >= min_samples:
                segments.append((seg_start_sample, seg_end_sample))

    # If audio ends while inside a segment, close it to the end
    if in_segment:
        seg_end_sample = len(audio_data)
        duration = seg_end_sample - seg_start_sample
        if duration >= min_samples:
            segments.append((seg_start_sample, seg_end_sample))

    # Build processed audio array (tones only)
    processed = np.zeros_like(audio_data)
    for s, e in segments:
        processed[s:e] = audio_data[s:e]

    processed_audio = processed
    return segments


# -------------------------
# High-level run() function (replaces previous class run)
# -------------------------
def run(file_path: str) -> str:
    """
    Top-level routine:
      - load audio
      - plot full waveform
      - segment audio into candidate tone regions
      - decode each region via FFT and apply deduplication rules
      - return the concatenated decoded string

    This preserves the original program's CLI prints and plotting behavior.
    """
    print(f"[-] Loading: {file_path}")
    try:
        fs, audio = load_audio(file_path)
    except Exception as exc:
        return f"Error loading file: {exc}"

    # === PLOT FULL AUDIO SIGNAL ===
    plt.figure(figsize=(12, 4))
    plt.plot(audio, linewidth=0.8)
    plt.title("Full Audio Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    print("[-] Segmenting audio based on silence...")
    segment_bounds = segment_audio(audio, fs)
    print(f"[-] Found {len(segment_bounds)} potential key presses.")

    # Deduplication threshold: if same char appears within this gap, it's a duplicate
    max_dup_gap_samples = int(fs * (MAX_SAME_DIGIT_GAP_MS / 1000.0))

    decoded_chars: List[str] = []
    last_char: Optional[str] = None
    last_end = 0

    print("[-] Decoding segments via FFT...")
    for i, (start, end) in enumerate(segment_bounds):
        segment = audio[start:end]

        # Plot local segment waveform for inspection
        plt.figure(figsize=(10, 3))
        plt.plot(segment, linewidth=0.8)
        plt.title(f"Segment {i+1}: Samples {start} to {end}")
        plt.xlabel("Sample Index (local to segment)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

        char = decode_segment(segment, fs)

        if char:
            gap = start - last_end
            gap_ms = (gap / fs) * 1000.0

            # Plot FFT for visual confirmation with the decoded digit
            plot_fft(segment, fs, char, i)

            # Deduplicate: skip if same char and gap is small
            if char == last_char and gap < max_dup_gap_samples:
                print(f"    Segment {i+1}: '{char}' (duplicate, gap={gap_ms:.1f}ms, skipped)")
            else:
                decoded_chars.append(char)
                print(f"    Segment {i+1}: Decoded '{char}' (gap={gap_ms:.1f}ms)")
                last_char = char

            last_end = end
        else:
            # If undecodable, preserve last_char so short noisy gaps don't break duplicates handling
            print(f"    Segment {i+1}: Could not decode (Noise or Invalid Tone)")

    result = "".join(decoded_chars)
    return result


# -------------------------
# CLI entrypoint (keeps original script behavior)
# -------------------------
if __name__ == "__main__":
    # Default file name (as in original)
    filename = "1.wav"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    print("=======================================")
    print("   DTMF SIGNAL DECODER - TEAM 4")
    print("=======================================")

    if os.path.exists(filename):
        phone_number = run(filename)
        print("=======================================")
        print(f"FINAL DECODED NUMBER: {phone_number}")
        print("=======================================")
    else:
        print(f"File '{filename}' not found.")
        print("Usage: python dtmf_decoder.py <path_to_wav_file>")
