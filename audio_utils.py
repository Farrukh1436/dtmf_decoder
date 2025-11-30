"""utils to load and proces audio files"""

import numpy as np
from scipy.io import wavfile
import os
from typing import Tuple


def load_audio_file(file_path: str) -> Tuple[int, np.ndarray]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        raise ValueError(f"error when reading: {e}")

    # normalize audio data to float range +- 1
    data = normalize_audio(data)
    
    # convert to mono if auido is stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
        print(f"[INFO] Converted stereo to mono audio")

    #debug (hate print statements in utils, but oh well)
    print(f"[INFO] Loaded audio: { len(data)} samples at {sample_rate} Hz ({len(data)/sample_rate:.2f}s)")
    return sample_rate, data

def normalize_audio(data: np.ndarray) -> np.ndarray:
    """
    Normalize audio data to float range [-1, 1] based on data type.
    
    Args:
        data: Raw audio data from wavfile.read()
        
    Returns:
        Normalized audio data as float array
    """
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype in [np.float32, np.float64]:
        return data.astype(np.float64)
    else:
        # Fallback: assume 16-bit range
        return data.astype(np.float64) / 32768.0

def calculate_frame_energy(audio_data: np.ndarray, sample_rate: int, 
                          frame_size_ms: float = 20.0, hop_size_ms: float = 10.0) -> Tuple[np.ndarray, int]:
    """
    Calculate RMS energy for each frame of the audio signal.
    
    Args:
        audio_data: Input audio signal
        sample_rate: Audio sample rate
        frame_size_ms: Frame size in milliseconds
        hop_size_ms: Hop size in milliseconds
        
    Returns:
        Tuple of (energy_array, hop_length_samples)
    """
    frame_length = int(sample_rate * frame_size_ms / 1000.0)
    hop_length = int(sample_rate * hop_size_ms / 1000.0)
    
    if len(audio_data) < frame_length:
        print("[WARNING] Audio shorter than one frame")
        return np.array([]), hop_length
    
    # Calculate number of frames
    num_frames = (len(audio_data) - frame_length) // hop_length + 1
    energy = np.zeros(num_frames)
    
    # Calculate RMS energy for each frame
    for i in range(num_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        frame = audio_data[start_idx:end_idx]
        energy[i] = np.sqrt(np.mean(frame**2))
    
    return energy, hop_length
