"""Main DTMF Decoder Class"""

import time
from typing import List, Tuple, Optional
import numpy as np

from config import DTMFConfig
from audio_utils import load_audio_file, calculate_frame_energy
from signal_processing import (
    apply_hamming_window, compute_fft_spectrum, find_dtmf_peaks, 
    validate_signal_quality, segment_audio_by_energy
)
from dtmf_mapping import DTMFMapper
from visualization import (
    plot_full_waveform, plot_audio_segment, plot_fft_spectrum,
    plot_energy_analysis, plot_summary_results
)

class DTMFDecoder:
    """
    Main DTMF decoder class that orchestrates the complete decoding process.
    """
    
    def __init__(self, config: DTMFConfig = None, enable_plotting: bool = True):
        """
        Initialize the DTMF decoder.
        
        Args:
            config: Configuration object (uses default if None)
            enable_plotting: Whether to show plots during processing
        """
        self.config = config or DTMFConfig()
        self.mapper = DTMFMapper()
        self.enable_plotting = enable_plotting
        self.processed_audio = None
    
    def decode_audio_file(self, file_path: str) -> str:
        """
        Decode DTMF tones from an audio file.
        
        Args:
            file_path: Path to the WAV audio file
            
        Returns:
            Decoded phone number/sequence as string
        """
        start_time = time.time()
        
        print(f"[INFO] Starting DTMF decoding: {file_path}")
        
        # Load and preprocess audio
        try:
            sample_rate, audio_data = load_audio_file(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to load audio: {e}")
            return ""
        
        # Show full waveform
        if self.enable_plotting:
            plot_full_waveform(audio_data, sample_rate, f"Audio Waveform: {file_path}")
        
        # Segment audio based on energy
        segments = self._segment_audio(audio_data, sample_rate)
        
        if not segments:
            print("[WARNING] No audio segments detected")
            return ""
        
        # Decode each segment
        decoded_chars = self._decode_segments(audio_data, sample_rate, segments)
        
        # Apply deduplication
        final_sequence = self._apply_deduplication(decoded_chars, segments, sample_rate)
        
        # Show results
        processing_time = time.time() - start_time
        if self.enable_plotting:
            plot_summary_results(final_sequence, processing_time)
        else:
            print(f"\n[RESULT] Decoded: {final_sequence} (in {processing_time:.2f}s)")
        
        return final_sequence
    
    def _segment_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        """
        Segment audio into regions containing DTMF tones.
        """
        print("[INFO] Analyzing audio energy for segmentation...")
        
        # Calculate frame energy
        energy, hop_length = calculate_frame_energy(
            audio_data, sample_rate, 
            self.config.FRAME_SIZE_MS, self.config.HOP_SIZE_MS
        )
        
        if len(energy) == 0:
            return []
        
        # Segment based on energy
        segments = segment_audio_by_energy(audio_data, sample_rate, energy, hop_length)
        
        # Show energy analysis
        if self.enable_plotting and len(energy) > 0:
            max_energy = np.max(energy)
            median_energy = np.median(energy)
            threshold = median_energy + self.config.ENERGY_THRESHOLD_FACTOR * (max_energy - median_energy)
            plot_energy_analysis(energy, threshold, sample_rate, hop_length, segments)
        
        return segments
    
    def _decode_segments(self, audio_data: np.ndarray, sample_rate: int, 
                        segments: List[Tuple[int, int]]) -> List[Tuple[str, int, int]]:
        """
        Decode individual audio segments using FFT analysis.
        
        Returns:
            List of (character, start_sample, end_sample) tuples
        """
        print(f"[INFO] Decoding {len(segments)} segments...")
        
        decoded_chars = []
        
        for i, (start, end) in enumerate(segments):
            segment = audio_data[start:end]
            
            # Show segment waveform
            if self.enable_plotting:
                plot_audio_segment(segment, i, start, end, sample_rate)
            
            # Decode this segment
            character = self._decode_single_segment(segment, sample_rate)
            
            if character:
                decoded_chars.append((character, start, end))
                print(f"    Segment {i+1}: '{character}' ({len(segment)} samples)")
            else:
                print(f"    Segment {i+1}: [UNDECODABLE] (noise or invalid tone)")
        
        return decoded_chars
    
    def _decode_single_segment(self, segment: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        Decode a single audio segment containing one DTMF tone.
        """
        if len(segment) == 0:
            return None
        
        # Apply window and compute FFT
        windowed_segment = apply_hamming_window(segment)
        freqs, spectrum = compute_fft_spectrum(windowed_segment, sample_rate, 
                                              self.config.MIN_FFT_SIZE)
        
        # Find DTMF frequency peaks
        low_freq, high_freq, low_mag, high_mag = find_dtmf_peaks(freqs, spectrum)
        
        # Validate signal quality
        if low_freq is None or high_freq is None:
            if self.enable_plotting:
                plot_fft_spectrum(freqs, spectrum, None, 0)
            return None
        
        if not validate_signal_quality(spectrum, freqs, low_mag, high_mag):
            if self.enable_plotting:
                plot_fft_spectrum(freqs, spectrum, None, 0, low_freq, high_freq)
            return None
        
        # Map frequencies to character
        character = self.mapper.decode_frequency_pair(low_freq, high_freq)
        
        # Show FFT analysis
        if self.enable_plotting:
            plot_fft_spectrum(freqs, spectrum, character, 0, low_freq, high_freq)
        
        return character
    
    def _apply_deduplication(self, decoded_chars: List[Tuple[str, int, int]], 
                           segments: List[Tuple[int, int]], sample_rate: int) -> str:
        """
        Apply deduplication to remove repeated characters that are too close together.
        """
        if not decoded_chars:
            return ""
        
        print("[INFO] Applying deduplication...")
        
        # Calculate gap threshold
        max_gap_samples = int(sample_rate * self.config.MAX_SAME_DIGIT_GAP_MS / 1000.0)
        
        final_chars = []
        last_char = None
        last_end = 0
        
        for char, start, end in decoded_chars:
            gap_samples = start - last_end
            gap_ms = gap_samples / sample_rate * 1000
            
            # Check for duplication
            if char == last_char and gap_samples < max_gap_samples:
                print(f"    Skipping duplicate '{char}' (gap: {gap_ms:.1f}ms)")
            else:
                final_chars.append(char)
                print(f"    Keeping '{char}' (gap: {gap_ms:.1f}ms)")
                last_char = char
            
            last_end = end
        
        return ''.join(final_chars)
    
    def decode_with_confidence(self, file_path: str) -> Tuple[str, List[dict]]:
        """
        Decode with detailed confidence information for each detected tone.
        
        Returns:
            Tuple of (decoded_string, list_of_detection_details)
        """
        # This could be extended to return detailed analysis
        sequence = self.decode_audio_file(file_path)
        return sequence, []  # Placeholder for detailed analysis
