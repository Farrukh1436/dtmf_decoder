# in this file, we can define config parameeters for decoding

class DTMFConfig:
 
    # standard dtmf freqs.
    LOW_FREQS = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477, 1633]
    
    # frequency analysis parameters
    FREQ_TOLERANCE = 10.0  # frequency matching tolerance (±10 Hz)
    MIN_FFT_SIZE = 4096    # minimum FFT size for frequency resolution
    
    # timing parameters (milliseconds)
    MIN_SIGNAL_DURATION_MS = 40   # minimum tone duration to consider valid
    MIN_SILENCE_DURATION_MS = 20  # minimum silence between tones
    MAX_SAME_DIGIT_GAP_MS = 50    # max gap for duplicate detection
    
    # energy detection parameters
    FRAME_SIZE_MS = 20.0          # frame size for energy calculation
    HOP_SIZE_MS = 10.0            # hop size between frames
    ENERGY_THRESHOLD_FACTOR = 0.20 # factor for adaptive threshold
    SILENCE_THRESHOLD = 0.05       # amplitude threshold for silence
    
    # noise rejection parameters
    MIN_SNR_RATIO = 10.0          # minimum signal-to-noise ratio
    MIN_PEAK_RATIO = 0.1          # minimum peak relative to spectrum max
    
    # frequency band definitions
    LOW_BAND_MIN = 600
    LOW_BAND_MAX = 1000
    HIGH_BAND_MIN = 1100
    HIGH_BAND_MAX = 1700
    DTMF_RANGE_MIN = 600
    DTMF_RANGE_MAX = 1700
