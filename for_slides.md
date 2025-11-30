Here’s a concrete 7‑slide deck you can copy directly into your slides.

Slide 1 – Mission & DTMF Basics

Title:
DTMF Signal Decoder – Secret Agent Mission

Bullet points:

We recorded a suspicious phone call from an old landline at a bar.
We cannot see which keys were pressed, only have the audio.
Our mission: automatically recover the dialed phone number from the recording.
The phone uses DTMF: each key is encoded by two sine waves (one low, one high).
Low frequencies (rows): 697, 770, 852, 941 Hz.
High frequencies (columns): 1209, 1336, 1477 Hz.
Example: key “5” = 770 Hz (row) + 1336 Hz (column).
Equation:
x
(
t
)
=
sin
⁡
(
2
π
f
low
t
)
+
sin
⁡
(
2
π
f
high
t
)
Here’s a concrete 7‑slide deck you can copy directly into your slides.

---

Slide 1 – Mission & DTMF Basics

Title:
DTMF Signal Decoder – Secret Agent Mission

Bullet points:

- We recorded a suspicious phone call from an old landline at a bar.
- We cannot see which keys were pressed, only have the audio.
- Our mission: automatically recover the dialed phone number from the recording.
- The phone uses DTMF: each key is encoded by two sine waves (one low, one high).
- Low frequencies (rows): 697, 770, 852, 941 Hz.
- High frequencies (columns): 1209, 1336, 1477 Hz.
- Example: key “5” = 770 Hz (row) + 1336 Hz (column).
- Equation:  \(x(t) = \sin(2\pi f_{\text{low}} t) + \sin(2\pi f_{\text{high}} t)\)

Visuals:

- Small diagram of DTMF keypad with row and column frequencies labeled.
- Simple cartoon of “spy with microphone” near a phone.

---

Slide 2 – Overall System Pipeline

Title:
From Audio Recording to Phone Number

Bullet points:

- Input: a `.wav` recording of the phone call.
- Step 1: Load audio and show full waveform (`main.py`, `audio_utils.py`).
- Step 2: Use short-time energy to find where tones are (and where silence is).
- Step 3: For each detected tone segment, apply Hamming window and FFT.
- Step 4: Detect two strong peaks in low and high DTMF bands.
- Step 5: Map those two frequencies to a digit using the DTMF table.
- Step 6: Remove accidental duplicates if the same digit repeats too quickly.
- Output: final decoded phone number.

Visuals:

- Big block diagram:  
  Audio file → Segmentation → Window + FFT → Peak detection → Digit mapping → Deduplication → Phone number.

---

Slide 3 – Segmentation by Energy (Time Domain)

Title:
Finding Individual DTMF Digits with Short-Time Energy

Bullet points:

- DTMF digits are separated by silence; we use this to cut the signal into pieces.
- We divide the audio into overlapping frames:
  - Frame size (ms) and hop size (ms) come from `DTMFConfig`.
  - Convert to samples:  
    \(N_{\text{frame}} = \text{FRAME\_SIZE\_MS} \cdot f_s / 1000\)  
    \(N_{\text{hop}} = \text{HOP\_SIZE\_MS} \cdot f_s / 1000\)
- For each frame \(k\) we compute short-time energy:  
  \(E_k = \sum_{n=0}^{N_{\text{frame}}-1} x_k[n]^2\)
- We compute an adaptive threshold:  
  \(T = E_\text{median} + \alpha (E_\max - E_\text{median})\), where \(\alpha\) = `ENERGY_THRESHOLD_FACTOR`.
- Frames with \(E_k > T\) are marked as “active”; consecutive active frames form a segment.
- Only segments with duration ≥ `MIN_SIGNAL_DURATION_MS` are kept as candidate digits.

Visuals:

- Top plot: waveform of the whole signal.
- Bottom plot: energy vs. time with threshold line and highlighted active regions.

---

Slide 4 – Windowing & FFT (Frequency Domain)

Title:
From Time Segment to Frequency Peaks

Bullet points:

- Each segment should contain (approximately) one DTMF digit.
- Before FFT, we apply a Hamming window to reduce spectral leakage:  
  \(w[n] = 0.54 - 0.46\cos\left(\dfrac{2\pi n}{N-1}\right)\)  
  \(x_w[n] = x[n] \cdot w[n]\)
- We then compute the real FFT on the windowed segment (`compute_fft_spectrum`):  
  \(X[k] = \sum_{n=0}^{N-1} x_w[n] e^{-j 2\pi kn/N}\)
- Magnitude spectrum: \(|X[k]|\).
- Frequency axis:  
  \(f_k = \dfrac{k}{N} f_s\)
- In code:  
  `spectrum = np.abs(np.fft.rfft(signal, n=N))`  
  `freqs = np.fft.rfftfreq(N, d=1/sample_rate)`

Visuals:

- Plot of a single segment (time domain) next to its magnitude spectrum showing two clear peaks.

---

Slide 5 – Peak Detection & Signal Quality

Title:
Detecting DTMF Peaks and Rejecting Noise

Bullet points:

- We search for peaks in two bands:
  - Low band (rows): `[LOW_BAND_MIN, LOW_BAND_MAX]` (≈ 697–941 Hz).
  - High band (columns): `[HIGH_BAND_MIN, HIGH_BAND_MAX]` (≈ 1209–1477 Hz).
- For each band:
  - Create mask on `freqs`, zero spectrum outside the band.
  - Find index of maximum value in that band → gives `peak_freq` and `peak_mag`.
- Estimate noise floor in the DTMF range:  
  \(N_{\text{floor}} = \operatorname{median}(|X[k]|,\ f_k \in \text{DTMF range})\)
- SNR check (`MIN_SNR_RATIO`):
  - Required magnitude: \(\text{min\_magnitude} = N_{\text{floor}} \cdot R_{\text{SNR}}\)
  - Both peaks must satisfy:  
    \(|X(f_{\text{low}})| \ge N_{\text{floor}} R_{\text{SNR}}\)  
    \(|X(f_{\text{high}})| \ge N_{\text{floor}} R_{\text{SNR}}\)
- Relative magnitude check (`MIN_PEAK_RATIO`):
  - Let \(|X|_\max = \max_k |X[k]|\).
  - Require: \(|X(f_{\text{low/high}})| \ge R_{\text{rel}} \cdot |X|_\max\).
- If checks fail, segment is considered noise or invalid tone → no digit.

Visuals:

- Spectrum plot with shaded low and high bands and marked peak points.
- Simple illustration of noise floor vs. peaks.

---

Slide 6 – Mapping to Digits & Deduplication

Title:
From Frequencies to Digits (and Cleaning Repeats)

Bullet points:

- `DTMFMapper` stores ideal row and column frequencies and digit table.
- Given detected `(f_low, f_high)` from FFT:
  - Compute distance to each official row frequency \(f_{r,i}\):  
    \(\Delta f_{r,i} = |f_{\text{low}} - f_{r,i}|\)
  - Choose row with smallest \(\Delta f_{r,i}\) within `FREQ_TOLERANCE`.
  - Do same for columns with \(f_{c,j}\) and \(f_{\text{high}}\).
  - Use (row, column) pair to look up digit (0–9, etc.).
- Noise tolerance: real peaks may shift by ±1–4 Hz; tolerance handles this.
- Deduplication (`_apply_deduplication`):
  - Sometimes the same key press is split into two segments.
  - Define max allowed gap between same digits:  
    \(G_\max = \dfrac{\text{MAX\_SAME\_DIGIT\_GAP\_MS}}{1000} f_s\)
  - For each decoded `(char, start, end)` in time order:
    - Compute `gap_samples = start - last_end`, `gap_ms = gap_samples / f_s * 1000`.
    - If `char == last_char` and `gap_samples < G_max` → treat as duplicate and skip.
    - Otherwise keep the digit and update `last_char`, `last_end`.

Visuals:

- Small DTMF grid showing how a row and a column intersect to give a digit.
- Timeline showing raw detections vs. final cleaned sequence.

---

Slide 7 – Results, Parameters & Conclusion

Title:
Results and What We Learned

Bullet points:

- Our program decodes the DTMF sequence from the assignment audio file and recovers the full phone number.
- Configuration (`DTMFConfig`) controls:
  - Frame / hop size (time resolution of energy).
  - `MIN_SIGNAL_DURATION_MS`, `MAX_SAME_DIGIT_GAP_MS` (segment validity and deduplication).
  - `FREQ_TOLERANCE`, `MIN_SNR_RATIO`, `MIN_PEAK_RATIO`, `MIN_FFT_SIZE` (FFT resolution & robustness).
- DSP concepts used in our implementation:
  - Short-time energy and segmentation by silence.
  - Windowing (Hamming) and FFT for frequency analysis.
  - Peak detection, SNR, and threshold-based validation.
- Decoder is robust to small frequency errors (±1–4 Hz) and background noise.
- Possible future work: real-time microphone decoding, confidence scores per digit, more advanced segmentation.

Visuals:

- Screenshot of final decoded phone number from the console.
- Optionally a summary plot (waveform + segments + final result).