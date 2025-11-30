"""Signal Processing Functions for DTMF Analysis"""

import numpy as np
from typing import List, Tuple, Optional
from config import DTMFConfig

def apply_hamming_window(signal: np.ndarray) -> np.ndarray:
    """
    Apply Hamming window to reduce spectral leakage.
    
    Args:
        signal: Input audio signal
        
    Returns:
        Windowed signal
    """
    return signal * np.hamming(len(signal))

def compute_fft_spectrum(signal: np.ndarray, sample_rate: int, 
                        min_fft_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude spectrum of the input signal.
    
    Args:
        signal: Input audio signal (should be windowed)
        sample_rate: Audio sample rate
        min_fft_size: Minimum FFT size (default from config)
        
    Returns:
        Tuple of (frequency_bins, magnitude_spectrum)
    """
    if min_fft_size is None:
        min_fft_size = DTMFConfig.MIN_FFT_SIZE
    
    # Choose FFT size (minimum size or signal length, whichever is larger)
    N = max(min_fft_size, len(signal))
    
    # Compute FFT
    spectrum = np.abs(np.fft.rfft(signal, n=N))
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    
    return freqs, spectrum

def find_peak_in_band(freqs: np.ndarray, spectrum: np.ndarray, 
                      band_min: float, band_max: float) -> Tuple[Optional[float], float]:
    """
    Find the frequency with maximum magnitude within a specific band.
    
    Args:
        freqs: Frequency bins from FFT
        spectrum: Magnitude spectrum
        band_min: Lower bound of frequency band
        band_max: Upper bound of frequency band
        
    Returns:
        Tuple of (peak_frequency, magnitude) or (None, 0) if no peak found
    """
    # Create band mask
    band_mask = (freqs >= band_min) & (freqs <= band_max)
    
    if not np.any(band_mask):
        return None, 0.0
    
    # Find peak in band
    band_spectrum = spectrum * band_mask
    peak_idx = np.argmax(band_spectrum)
    peak_freq = freqs[peak_idx]
    peak_magnitude = spectrum[peak_idx]
    
    return peak_freq, peak_magnitude

def find_dtmf_peaks(freqs: np.ndarray, spectrum: np.ndarray) -> Tuple[Optional[float], Optional[float], float, float]:
    """
    Find low and high frequency peaks in DTMF spectrum.
    
    Args:
        freqs: Frequency bins from FFT
        spectrum: Magnitude spectrum
        
    Returns:
        Tuple of (low_freq, high_freq, low_magnitude, high_magnitude)
    """
    # Find peaks in low and high DTMF bands
    low_freq, low_mag = find_peak_in_band(freqs, spectrum, 
                                          DTMFConfig.LOW_BAND_MIN, DTMFConfig.LOW_BAND_MAX)
    high_freq, high_mag = find_peak_in_band(freqs, spectrum, 
                                            DTMFConfig.HIGH_BAND_MIN, DTMFConfig.HIGH_BAND_MAX)
    
    return low_freq, high_freq, low_mag, high_mag

def validate_signal_quality(spectrum: np.ndarray, freqs: np.ndarray, 
                           low_mag: float, high_mag: float) -> bool:
    """
    Validate signal quality using SNR and relative magnitude checks.
    
    Args:
        spectrum: Full magnitude spectrum
        freqs: Frequency bins
        low_mag: Low frequency peak magnitude
        high_mag: High frequency peak magnitude
        
    Returns:
        True if signal quality is acceptable, False otherwise
    """
    # Calculate noise floor in DTMF frequency range
    dtmf_mask = (freqs >= DTMFConfig.DTMF_RANGE_MIN) & (freqs <= DTMFConfig.DTMF_RANGE_MAX)
    noise_floor = np.median(spectrum[dtmf_mask])
    
    # SNR check: peaks must be significantly above noise floor
    min_magnitude = noise_floor * DTMFConfig.MIN_SNR_RATIO
    if low_mag < min_magnitude or high_mag < min_magnitude:
        return False
    
    # Relative magnitude check: peaks must be significant relative to spectrum max
    max_spectrum = np.max(spectrum)
    min_relative = max_spectrum * DTMFConfig.MIN_PEAK_RATIO
    if low_mag < min_relative or high_mag < min_relative:
        return False
    
    return True

def segment_audio_by_energy(audio_data: np.ndarray, sample_rate: int, 
                           energy: np.ndarray, hop_length: int) -> List[Tuple[int, int]]:
    """
    Segment audio into active regions based on energy levels.
    
    Args:
        audio_data: Input audio signal
        sample_rate: Audio sample rate
        energy: Frame energy array
        hop_length: Hop length in samples
        
    Returns:
        List of (start_sample, end_sample) tuples for active segments
    """
    # Calculate adaptive threshold
    max_energy = np.max(energy)
    median_energy = np.median(energy)
    
    if max_energy == 0:
        print("[WARNING] Audio appears to be completely silent")
        return []
    
    threshold = median_energy + DTMFConfig.ENERGY_THRESHOLD_FACTOR * (max_energy - median_energy)
    
    print(f"[INFO] Energy analysis: max={max_energy:.4f}, threshold={threshold:.4f}")
    
    # Detect active frames
    is_active = energy > threshold
    
    # Convert frame indices to sample indices and group into segments
    segments = []
    in_segment = False
    segment_start = 0
    min_samples = int((DTMFConfig.MIN_SIGNAL_DURATION_MS / 1000.0) * sample_rate)
    
    for i in range(len(is_active)):
        if is_active[i] and not in_segment:
            # Start of new segment
            in_segment = True
            segment_start = i * hop_length
        elif not is_active[i] and in_segment:
            # End of segment
            in_segment = False
            segment_end = i * hop_length
            duration_samples = segment_end - segment_start
            
            # Only keep segments longer than minimum duration
            if duration_samples >= min_samples:
                segments.append((segment_start, segment_end))
    
    # Handle case where audio ends while in a segment
    if in_segment:
        segment_end = len(audio_data)
        duration_samples = segment_end - segment_start
        if duration_samples >= min_samples:
            segments.append((segment_start, segment_end))
    
    print(f"[INFO] Found {len(segments)} audio segments")
    return segments
