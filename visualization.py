"""Visualization Functions for DTMF Analysis"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_full_waveform(audio_data: np.ndarray, sample_rate: int, title: str = "Audio Waveform"):
    """
    Plot the complete audio waveform.
    
    Args:
        audio_data: Audio signal data
        sample_rate: Audio sample rate
        title: Plot title
    """
    plt.figure(figsize=(12, 4))
    
    # Create time axis
    time_axis = np.arange(len(audio_data)) / sample_rate
    
    plt.plot(time_axis, audio_data, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_audio_segment(segment: np.ndarray, segment_index: int, 
                      start_sample: int, end_sample: int, sample_rate: int):
    """
    Plot an individual audio segment.
    
    Args:
        segment: Audio segment data
        segment_index: Index of the segment
        start_sample: Start sample index in original audio
        end_sample: End sample index in original audio
        sample_rate: Audio sample rate
    """
    plt.figure(figsize=(10, 3))
    
    # Create time axis for the segment
    time_axis = np.arange(len(segment)) / sample_rate
    duration_ms = len(segment) / sample_rate * 1000
    
    plt.plot(time_axis, segment, linewidth=0.8)
    plt.title(f"Segment {segment_index+1}: Samples {start_sample}-{end_sample} ({duration_ms:.1f}ms)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fft_spectrum(freqs: np.ndarray, spectrum: np.ndarray, 
                     decoded_char: Optional[str], segment_index: int,
                     low_freq: Optional[float] = None, high_freq: Optional[float] = None):
    """
    Plot FFT magnitude spectrum with detected peaks highlighted.
    
    Args:
        freqs: Frequency bins
        spectrum: Magnitude spectrum
        decoded_char: Decoded character (or None)
        segment_index: Index of the segment
        low_freq: Detected low frequency peak
        high_freq: Detected high frequency peak
    """
    plt.figure(figsize=(12, 6))
    
    # Plot spectrum
    plt.plot(freqs, spectrum, 'b-', linewidth=1)
    
    # Highlight detected peaks
    if low_freq is not None:
        plt.axvline(x=low_freq, color='red', linestyle='--', alpha=0.7, label=f'Low: {low_freq:.1f} Hz')
    if high_freq is not None:
        plt.axvline(x=high_freq, color='green', linestyle='--', alpha=0.7, label=f'High: {high_freq:.1f} Hz')
    
    plt.title(f"FFT Spectrum - Segment {segment_index+1} - Decoded: '{decoded_char}'")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 2000)  # Focus on DTMF frequency range
    plt.grid(True)
    
    if low_freq is not None or high_freq is not None:
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_energy_analysis(energy: np.ndarray, threshold: float, sample_rate: int, 
                        hop_length: int, segments: list):
    """
    Plot energy analysis showing detected segments.
    
    Args:
        energy: Frame energy array
        threshold: Energy threshold used for segmentation
        sample_rate: Audio sample rate
        hop_length: Hop length in samples
        segments: List of detected segments
    """
    plt.figure(figsize=(12, 4))
    
    # Create time axis for energy frames
    time_axis = np.arange(len(energy)) * hop_length / sample_rate
    
    # Plot energy
    plt.plot(time_axis, energy, 'b-', linewidth=1, label='Energy')
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold: {threshold:.4f}')
    
    # Highlight detected segments
    for i, (start, end) in enumerate(segments):
        start_time = start / sample_rate
        end_time = end / sample_rate
        plt.axvspan(start_time, end_time, alpha=0.3, color='green', label='Active' if i == 0 else "")
    
    plt.title("Energy-based Segmentation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_summary_results(decoded_sequence: str, processing_time: float = None):
    """
    Display final results in a formatted way.
    
    Args:
        decoded_sequence: Final decoded phone number/sequence
        processing_time: Optional processing time in seconds
    """
    print("\n" + "="*50)
    print("           DTMF DECODING RESULTS")
    print("="*50)
    print(f"Decoded Sequence: {decoded_sequence}")
    if processing_time is not None:
        print(f"Processing Time: {processing_time:.2f} seconds")
    print("="*50)
    print()

def create_dtmf_reference_plot():
    """
    Create a reference plot showing DTMF frequency layout.
    """
    from config import DTMFConfig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grid showing frequency combinations
    keypad = [
        ['1', '2', '3', 'A'],
        ['4', '5', '6', 'B'],
        ['7', '8', '9', 'C'],
        ['*', '0', '#', 'D']
    ]
    
    # Plot frequency points
    for i, low_freq in enumerate(DTMFConfig.LOW_FREQS):
        for j, high_freq in enumerate(DTMFConfig.HIGH_FREQS):
            ax.plot(high_freq, low_freq, 'ro', markersize=10)
            ax.annotate(keypad[i][j], (high_freq, low_freq), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=12, fontweight='bold')
    
    ax.set_xlabel('High Frequency (Hz)')
    ax.set_ylabel('Low Frequency (Hz)')
    ax.set_title('DTMF Frequency Map')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(1150, 1700)
    ax.set_ylim(650, 1000)
    
    plt.tight_layout()
    plt.show()
