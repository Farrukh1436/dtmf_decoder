import numpy as np
from scipy.io import wavfile
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_dilation

class DTMFConfig:
    """
    Configuration parameters for the DTMF Decoder.
    """
    # Standard DTMF Frequencies
    LOW_FREQS = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477, 1633]
    
    # Frequency deviation tolerance in Hz (from PDF requirements: +/- 1~4Hz)
    FREQ_TOLERANCE = 10.0  # Slightly relaxed to accommodate FFT resolution bins
    
    # Minimum duration (ms) for a signal to be considered a valid key press
    MIN_SIGNAL_DURATION_MS = 40 
    
    # Minimum silence (ms) to separate keys
    MIN_SILENCE_DURATION_MS = 20
    
    # Amplitude threshold to detect signal vs silence (0.0 to 1.0)
    # Adjust based on noise level of the recording
    SILENCE_THRESHOLD = 0.05
    
    # Maximum gap (ms) between segments to consider them as the same digit (deduplication)
    # Reduced to avoid removing legitimate repeated digits with brief pauses
    MAX_SAME_DIGIT_GAP_MS = 50

class DTMFDecoder:
    """
    A class to load audio, segment signals, and decode DTMF tones using FFT.
    """
    def plot_fft(self, segment, fs, digit, index):
        import matplotlib.pyplot as plt
        N = len(segment)

        # FFT with window
        window = np.hamming(N)
        spectrum = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(N, 1/fs)

        plt.figure(figsize=(12, 6))
        plt.plot(freqs, spectrum)
        plt.title(f"FFT of region {index}, decoded digit: {digit}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 2000)  # Only DTMF relevant area
        plt.show()

    
    def __init__(self, config: DTMFConfig = DTMFConfig()):
        self.config = config
        self.dtmf_map = self._build_dtmf_map()

    def _build_dtmf_map(self) -> Dict[Tuple[int, int], str]:
        """Builds the dictionary mapping frequency pairs to characters."""
        keys = [
            ['1', '2', '3', 'A'],
            ['4', '5', '6', 'B'],
            ['7', '8', '9', 'C'],
            ['*', '0', '#', 'D']
        ]
        mapping = {}
        for r, row_freq in enumerate(self.config.LOW_FREQS):
            for c, col_freq in enumerate(self.config.HIGH_FREQS):
                mapping[(row_freq, col_freq)] = keys[r][c]
        return mapping

    def load_audio(self, file_path: str) -> Tuple[int, np.ndarray]:
        """
        Loads a WAV file.
        
        Args:
            file_path: Path to the .wav file.
            
        Returns:
            Tuple containing (sample_rate, audio_data_normalized).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        sample_rate, data = wavfile.read(file_path)

        # Convert to float normalized between -1 and 1
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data - 128) / 128.0

        # Convert stereo to mono if necessary
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        return sample_rate, data

    def _find_nearest_freq(self, target: float, valid_freqs: List[int]) -> Optional[int]:
        """
        Finds the closest frequency in the valid list within tolerance.
        """
        best_match = None
        min_diff = float('inf')

        for freq in valid_freqs:
            diff = abs(target - freq)
            if diff <= self.config.FREQ_TOLERANCE and diff < min_diff:
                min_diff = diff
                best_match = freq
        
        return best_match

    def decode_segment(self, audio_segment: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        Performs FFT on a specific audio segment to identify the DTMF character.
        Includes noise rejection based on peak magnitude and SNR.
        """
        if len(audio_segment) == 0:
            return None

        # Apply Hamming window to reduce spectral leakage
        windowed_segment = audio_segment * np.hamming(len(audio_segment))
        
        # Calculate FFT
        # N should be large enough for resolution. 
        # Resolution = Fs / N. 44100 / 4096 ~= 10Hz resolution.
        N = 4096 
        if len(windowed_segment) > N:
            N = len(windowed_segment)
            
        spectrum = np.abs(np.fft.rfft(windowed_segment, n=N))
        freqs = np.fft.rfftfreq(N, d=1/sample_rate)

        # Separate into Low and High bands
        # Low band: < 1000 Hz, High band: > 1000 Hz
        low_band_mask = (freqs >= 600) & (freqs <= 1000)
        high_band_mask = (freqs >= 1100) & (freqs <= 1700)

        # Find peak in low band
        if not np.any(low_band_mask): return None
        low_spectrum = spectrum * low_band_mask
        low_idx = np.argmax(low_spectrum)
        low_peak = freqs[low_idx]
        low_magnitude = spectrum[low_idx]

        # Find peak in high band
        if not np.any(high_band_mask): return None
        high_spectrum = spectrum * high_band_mask
        high_idx = np.argmax(high_spectrum)
        high_peak = freqs[high_idx]
        high_magnitude = spectrum[high_idx]

        # Noise rejection: check if peaks are strong enough
        # Calculate noise floor (median of spectrum in DTMF range)
        dtmf_mask = (freqs >= 600) & (freqs <= 1700)
        noise_floor = np.median(spectrum[dtmf_mask])
        
        # Require peaks to be at least 10x above noise floor
        min_snr = 10.0
        if low_magnitude < noise_floor * min_snr or high_magnitude < noise_floor * min_snr:
            return None
        
        # Also require absolute minimum magnitude
        max_spectrum = np.max(spectrum)
        if low_magnitude < max_spectrum * 0.1 or high_magnitude < max_spectrum * 0.1:
            return None

        # Match to nearest valid DTMF frequencies
        matched_low = self._find_nearest_freq(low_peak, self.config.LOW_FREQS)
        matched_high = self._find_nearest_freq(high_peak, self.config.HIGH_FREQS)

        if matched_low and matched_high:
            return self.dtmf_map.get((matched_low, matched_high))
        
        return None

    def segment_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        """
        Splits audio into segments using frame-based energy detection with adaptive threshold.
        
        Returns:
            List of (start_index, end_index) tuples for deduplication support.
        """
        # Frame-based energy detection
        frame_length = int(sample_rate * 0.020)  # 20ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop
        
        # Calculate RMS energy per frame
        num_frames = (len(audio_data) - frame_length) // hop_length + 1
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start_idx = i * hop_length
            end_idx = start_idx + frame_length
            frame = audio_data[start_idx:end_idx]
            energy[i] = np.sqrt(np.mean(frame**2))
        
        # Adaptive threshold based on both median and max
        max_energy = np.max(energy)
        median_energy = np.median(energy)
        
        if max_energy == 0:
            print("[-] Warning: Audio file appears to be completely silent.")
            return []
        
        # Use median + factor of (max - median) to set threshold
        # This adapts to the noise floor - tuned to balance noise rejection and signal detection
        threshold = median_energy + 0.20 * (max_energy - median_energy)
        
        print(f"[-] Max amplitude: {max_energy:.4f}")
        print(f"[-] Dynamic silence threshold: {threshold:.4f}")
        
        # Detect active frames
        is_active = energy > threshold
        
        # Convert frame indices to sample indices
        segments = []
        in_segment = False
        segment_start = 0
        
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
                min_samples = int((self.config.MIN_SIGNAL_DURATION_MS / 1000.0) * sample_rate)
                
                if duration_samples >= min_samples:
                    segments.append((segment_start, segment_end))
        
        # Handle case where last frame is active
        if in_segment:
            segment_end = len(audio_data)
            duration_samples = segment_end - segment_start
            min_samples = int((self.config.MIN_SIGNAL_DURATION_MS / 1000.0) * sample_rate)
            if duration_samples >= min_samples:
                segments.append((segment_start, segment_end))
        # --- Build processed audio (tones only, silence removed) ---
        processed = np.zeros_like(audio_data)

        for start, end in segments:
            processed[start:end] = audio_data[start:end]

        self.processed_audio = processed  # store for later plotting

        return segments

    def run(self, file_path: str) -> str:
        """
        Main execution method with deduplication support.
        """
        print(f"[-] Loading: {file_path}")
        try:
            fs, audio = self.load_audio(file_path)
        except Exception as e:
            return f"Error loading file: {e}"
        # === PLOT FULL AUDIO SIGNAL ===
        plt.figure(figsize=(12, 4))
        plt.plot(audio, linewidth=0.8)
        plt.title("Full Audio Signal")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

        print("[-] Segmenting audio based on silence...")
        segment_bounds = self.segment_audio(audio, fs)
        print(f"[-] Found {len(segment_bounds)} potential key presses.")

        # Deduplication threshold: if same char appears within this gap, it's a duplicate
        max_dup_gap_samples = int(fs * (self.config.MAX_SAME_DIGIT_GAP_MS / 1000.0))

        decoded_chars = []
        last_char = None
        last_end = 0
        
        print("[-] Decoding segments via FFT...")
        for i, (start, end) in enumerate(segment_bounds):
            segment = audio[start:end]
            # === PLOT THIS SEGMENT ===
            plt.figure(figsize=(10, 3))
            plt.plot(segment, linewidth=0.8)
            plt.title(f"Segment {i+1}: Samples {start} to {end}")
            plt.xlabel("Sample Index (local to segment)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

            char = self.decode_segment(segment, fs)
            
            if char:
                gap = start - last_end
                gap_ms = (gap / fs) * 1000
                self.plot_fft(segment, fs, char, i)
                
                # Deduplicate: skip if same char and gap is small
                if char == last_char and gap < max_dup_gap_samples:
                    print(f"    Segment {i+1}: '{char}' (duplicate, gap={gap_ms:.1f}ms, skipped)")
                else:
                    decoded_chars.append(char)
                    print(f"    Segment {i+1}: Decoded '{char}' (gap={gap_ms:.1f}ms)")
                    last_char = char
                
                last_end = end
            else:
                # Don't reset last_char on undecodable segments to handle noise between duplicates
                print(f"    Segment {i+1}: Could not decode (Noise or Invalid Tone)")

        result = "".join(decoded_chars)
        return result

# --- Entry Point ---
if __name__ == "__main__":
    import sys

    # Default file from the PDF, or command line argument
    filename = "1.wav"
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    decoder = DTMFDecoder()
    
    print("=======================================")
    print("   DTMF SIGNAL DECODER - TEAM 4")
    print("=======================================")
    
    # Check if file exists in current directory for demonstration
    if os.path.exists(filename):
        phone_number = decoder.run(filename)
        print("=======================================")
        print(f"FINAL DECODED NUMBER: {phone_number}")
        print("=======================================")
    else:
        print(f"File '{filename}' not found.")
        print("Usage: python dtmf_decoder.py <path_to_wav_file>")