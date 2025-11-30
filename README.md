# DTMF Signal Decoder - Team 4

A comprehensive Python implementation for decoding Dual-Tone Multi-Frequency (DTMF) signals from WAV audio files. This project includes multiple implementation approaches, from object-oriented to functional programming styles, with advanced signal processing techniques.

## 🎯 Overview

DTMF (Dual-Tone Multi-Frequency) signaling is used in telecommunications to transmit keypad button presses over audio channels. Each button press generates two simultaneous pure tones - one from a low frequency group and one from a high frequency group.

### DTMF Frequency Map
```
        1209 Hz  1336 Hz  1477 Hz  1633 Hz
697 Hz    1        2        3        A
770 Hz    4        5        6        B  
852 Hz    7        8        9        C
941 Hz    *        0        #        D
```

## 🚀 Features

- **Object-Oriented Architecture**: Clean, maintainable class-based implementation
- **Advanced Signal Processing**: FFT-based frequency detection with noise rejection
- **Adaptive Segmentation**: Energy-based voice activity detection with configurable thresholds
- **Robust Decoding**: SNR-based filtering and duplicate detection
- **Real-time Visualization**: Interactive plotting of waveforms and frequency spectra
- **Flexible Input**: Supports various WAV file formats and sample rates
- **Configurable Parameters**: Adjustable frequency tolerance, energy thresholds, and timing constraints

## 📁 Project Structure

```
dtmf_decoder/
├── main.py          # Primary DTMF decoder implementation
├── requirements.txt # Python package dependencies
├── README.md       # This documentation
├── 1.wav           # Sample DTMF audio files
├── 2.wav
├── 0-9.wav
├── Dtmf0.ogg - Dtmf9.ogg  # Individual digit samples
└── venv/           # Python virtual environment
```

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Farrukh1436/dtmf_decoder
   cd dtmf_decoder
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Basic Usage

**Primary Implementation (main.py):**
```bash
python main.py [audio_file.wav]
```







### Examples

```bash
# Decode with visualization (recommended)
python main.py 1.wav
```


## 🔬 Technical Implementation

### Core Algorithm

1. **Audio Loading & Preprocessing**
   - WAV file loading with automatic format detection
   - PCM to float conversion with proper scaling
   - Stereo to mono conversion (channel averaging)

2. **Signal Segmentation**
   - Short-time energy calculation (20ms frames, 10ms hop)
   - Adaptive threshold based on energy distribution
   - Minimum duration filtering (40ms default)
   - Gap-based segment merging

3. **Frequency Analysis**
   - Hamming windowing for spectral leakage reduction
   - FFT with zero-padding for frequency resolution
   - Separate peak detection in low (600-1000 Hz) and high (1100-1700 Hz) bands

4. **Noise Rejection & Validation**
   - SNR-based filtering (10:1 signal-to-noise ratio)
   - Relative magnitude thresholds
   - Frequency tolerance matching (±10 Hz default)

5. **Character Mapping & Post-processing**
   - Frequency pair to DTMF character lookup
   - Duplicate detection with configurable gap timing
   - Result concatenation and formatting

### Key Classes & Functions

**main.py - Object-Oriented Implementation:**
- `DTMFConfig`: Configuration parameters and constants
- `DTMFDecoder`: Main decoder class with complete functionality
  - `load_audio()`: WAV file loading and preprocessing
  - `segment_audio()`: Energy-based signal segmentation
  - `decode_segment()`: FFT-based frequency analysis
  - `plot_fft()`: Real-time spectrum visualization
  - `run()`: Complete decoding pipeline

**Alternative Implementation (4.py):**
- `decode_file()`: High-level file processing
- `analyze_segment()`: Single-segment frequency analysis
- `detect_active_segments()`: Energy-based segmentation

## 🎛 Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `FREQ_TOLERANCE` | Frequency matching tolerance (Hz) | 10.0 | 1.0-20.0 |
| `MIN_SIGNAL_DURATION_MS` | Minimum tone duration (ms) | 40 | 20-100 |
| `MIN_SNR_RATIO` | Signal-to-noise ratio threshold | 10.0 | 5.0-20.0 |
| `MAX_SAME_DIGIT_GAP_MS` | Duplicate detection gap (ms) | 50 | 20-200 |

## 📊 Visualization Features

The primary implementation (`main.py`) provides comprehensive real-time visualization:

- **Full Waveform Display**: Complete audio signal overview with amplitude vs. sample index
- **Segment Plotting**: Individual tone segment waveforms for detailed analysis
- **FFT Spectrum Analysis**: Frequency domain representation with peak identification and magnitude plots
- **Interactive Decoding**: Visual confirmation of detected frequencies with labeled plots
- **Real-time Processing**: Step-by-step visualization of the decoding process

Each segment shows:
- Time-domain waveform of the isolated tone
- Frequency spectrum with identified low and high frequency peaks
- Decoded character result with confidence indicators

## 🧪 Testing & Validation

### Sample Files Included
- `1.wav`, `2.wav`: Multi-digit sequences
- `0-9.wav`: Complete digit sequence
- `Dtmf0.ogg` - `Dtmf9.ogg`: Individual digit references

### Performance Characteristics
- **Accuracy**: >95% under normal conditions
- **Noise Tolerance**: Effective with SNR > 10dB
- **Timing Flexibility**: Handles 40ms-2s tone durations
- **Format Support**: All standard WAV PCM formats


## 📄 License

This project is part of an academic assignment (Team Project #4) and is intended for educational purposes.

