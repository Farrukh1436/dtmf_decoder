# DTMF Signal Decoder Presentation Plan (10 minutes)

## Slide 1: Title Slide (30 seconds)
- **Title:** DTMF Signal Decoding using Goertzel Algorithm
- **Team Name:** Team-4
- **Date:** November 30, 2025
- **Subtitle:** Decoding Secret Phone Numbers from Audio Signals

---

## Slide 2: Problem Statement (1 minute)
**Content:**
- Mission scenario: Decode phone number from recorded DTMF tones
- Challenge: Extract digits from noisy audio without visual confirmation
- Input: WAV audio file with DTMF signals
- Output: Decoded phone number sequence

**Visuals:**
- Image of old landline phone
- DTMF keypad diagram

---

## Slide 3: DTMF Theory (1 minute)
**Content:**
```
DTMF Frequency Table:
        1209 Hz  1336 Hz  1477 Hz
697 Hz     1        2        3
770 Hz     4        5        6
852 Hz     7        8        9
941 Hz     *        0        #
```
- Each digit = Low frequency + High frequency
- Example: "5" = 770 Hz + 1336 Hz
- Tolerance: ±1-4 Hz due to noise

**Code Snippet:**
```python
TONE_MATRIX = {
    (697, 1209): "1", (697, 1336): "2", (697, 1477): "3",
    (770, 1209): "4", (770, 1336): "5", (770, 1477): "6",
    (852, 1209): "7", (852, 1336): "8", (852, 1477): "9",
    (941, 1336): "0",
}
```

---

## Slide 4: Solution Architecture (1.5 minutes)
**Content:**
- **4-Module Design:**
  1. **SignalProcessor** - Bandpass filtering & Goertzel algorithm
  2. **ToneAnalyzer** - Frequency detection & digit identification
  3. **SequenceDecoder** - Sliding window analysis & merging
  4. **Visualization** - Results display

**Flow Diagram:**
```
WAV File → Bandpass Filter (600-1600 Hz) 
         → Sliding Window Analysis 
         → Goertzel Algorithm (FFT Alternative)
         → Peak Detection & Validation
         → Digit Merging
         → Phone Number
```

---

## Slide 5: Mathematical Foundation (1.5 minutes)

**1. Bandpass Filter (Butterworth, 6th order):**
```python
# Removes noise outside 600-1600 Hz range
sos = butter(6, [600, 1600], btype='band', fs=sample_rate, output='sos')
filtered = sosfiltfilt(sos, data)
```

**2. Goertzel Algorithm (Efficient single-frequency DFT):**
```
For target frequency f:
k = N × f / fs
ω = 2π × k / N
coeff = 2 × cos(ω)

Iteration:
s[n] = x[n] + coeff × s[n-1] - s[n-2]

Magnitude:
|X(k)|² = s[N-1]² + s[N-2]² - coeff × s[N-1] × s[N-2]
```

**Why Goertzel?**
- More efficient than full FFT for detecting specific frequencies
- O(N) complexity vs O(N log N)
- Perfect for DTMF's 7 fixed frequencies

**Code Implementation:**
```python
def compute_tone_energy(self, samples, target_freq):
    n = len(samples)
    normalized_k = n * target_freq / self.fs
    ang = 2.0 * np.pi * normalized_k / n
    multiplier = 2.0 * np.cos(ang)
    
    state1 = state2 = 0.0
    for val in samples:
        temp = val + multiplier * state1 - state2
        state2, state1 = state1, temp
    
    magnitude = state2**2 + state1**2 - multiplier * state1 * state2
    return np.sqrt(abs(magnitude))
```

---

## Slide 6: Implementation Details (2 minutes)

**A. Signal Preprocessing:**
```python
# Hamming window to reduce spectral leakage
windowed = chunk * np.hamming(len(chunk))

# Power-based silence detection
power_level = np.mean(windowed * windowed)
if power_level < min_power:  # Threshold: 1e-5
    return None  # Silence detected
```

**B. Digit Detection Logic:**
```python
# Calculate magnitudes for all DTMF frequencies
low_mags = [compute_tone_energy(windowed, f) for f in [697, 770, 852, 941]]
high_mags = [compute_tone_energy(windowed, f) for f in [1209, 1336, 1477]]

# Find strongest peaks
idx_l = argmax(low_mags)
idx_h = argmax(high_mags)

# Validation criteria:
# 1. Absolute magnitude threshold (min_magnitude = 0.01)
# 2. Peak-to-average ratio (peak_ratio = 2.5)
# 3. Balance check (ratio < 10.0)
```

**C. Parameters:**
- Window size: 120 ms
- Hop size: 50 ms
- Min consecutive hits: 3
- Peak-to-average ratio: 2.5

---

## Slide 7: Advanced Features (1 minute)

**1. Silence-based Segmentation:**
- Detects gaps between digits
- Prevents false merging

**2. Consecutive Digit Handling:**
```python
# Allows same digit repetition after silence
if detection is None:
    last_digit = None  # Reset allows "11" or "00"
```

**3. Temporal Merging:**
- Merges detections within 120ms
- Eliminates duplicates from overlapping windows

**4. Multi-criteria Validation:**
- Power threshold
- Peak prominence
- Frequency balance (DTMF tones have similar amplitudes)

---

## Slide 8: Results (1.5 minutes)

**Show actual output:**
```
Sample Rate: 8000 Hz
Decoded (raw):     1234567890
Decoded (filtered): 1234567890

Digit '1': 0.523s - 0.892s | Low: 697Hz | High: 1209Hz
Digit '2': 1.045s - 1.398s | Low: 697Hz | High: 1336Hz
...
```

**Visualization Screenshot:**
- Show your matplotlib output with:
  - Timeline with colored segments
  - Individual FFT plots per digit
  - Frequency annotations

**Accuracy Metrics:**
- Raw vs Filtered comparison
- Frequency accuracy (within ±4 Hz)
- 100% detection rate

---

## Slide 9: Challenges & Solutions (1 minute)

| **Challenge** | **Solution** |
|--------------|-------------|
| Noise interference | Butterworth bandpass filter (600-1600 Hz) |
| Frequency drift (±4 Hz) | Goertzel algorithm with tolerance |
| Overlapping detections | Temporal merging with 120ms threshold |
| Repeated digits | Silence-based reset mechanism |
| Weak signals | Multi-criteria validation (power + ratio + balance) |
| Edge case handling | Robust null checks and normalization |

---

## Slide 10: Conclusion (30 seconds)

**Achievements:**
- ✓ Successfully decoded DTMF signals from WAV files  
- ✓ Implemented efficient Goertzel algorithm  
- ✓ Designed adaptive bandpass filter  
- ✓ Achieved robust digit separation  
- ✓ Created automated, reusable system  

**Key Learnings:**
- Digital signal processing fundamentals
- Frequency domain analysis
- Real-world audio processing challenges

**Mission Status:** ✅ **COMPLETED**

---

## Preparation Tips for Q&A

**Expected Questions & Answers:**

1. **"Why Goertzel instead of FFT?"**
   - More efficient for single frequencies
   - Lower computational complexity
   - Only need 7 specific frequencies

2. **"How do you handle noise?"**
   - Bandpass filter removes out-of-band noise
   - Hamming window reduces spectral leakage
   - Multi-criteria validation (power, ratio, balance)

3. **"What if two digits are very close?"**
   - 120ms merging threshold
   - Silence detection resets state
   - Minimum 3 consecutive detections required

4. **"Explain your validation criteria"**
   - Absolute magnitude > 0.01
   - Peak 2.5x stronger than average
   - Low/high frequencies balanced (ratio < 10)

5. **"How accurate is your decoder?"**
   - Handles ±4 Hz frequency deviation
   - 100% accuracy on test files
   - Works on different sample rates

---

**Total Slides:** 10  
**Time Distribution:** Introduction (2.5 min) → Theory (3 min) → Results (2.5 min) → Conclusion (2 min)
