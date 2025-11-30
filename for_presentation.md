# Team 4 – DTMF Decoder Project

This file is a guide for preparing your slides and oral presentation. Maximum 10 minutes – keep it focused and visual.

---

## 1. Mission Brief (1 slide)

- **Title:** "Signal decoding via Fourier Transform – DTMF mission"
- Very short story version of the assignment ("bad guy", bar, landline, microphone).
- State the **goal** clearly: "Given a recorded DTMF signal, automatically recover the phone number using FFT–based analysis."  
- Team name and members.

---

## 2. What is DTMF? (1–2 slides)

Slide 1 – Concept:
- Explain DTMF as **Dual Tone Multi-Frequency**: each key = low–frequency tone + high–frequency tone.
- Show the standard **DTMF keypad table** with row/column frequencies (you can copy a clean table from Wikipedia into the slide).
- Mention that actual measured peaks may be ±1–4 Hz because of noise / recording.

Slide 2 – Example tone:
- Show one digit signal example (e.g. from `Dtmf0.ogg`).
- Time–domain plot: amplitude vs time for that digit.
- Frequency–domain plot (FFT magnitude): two peaks corresponding to that digit’s row/column frequencies.

---

## 3. Signal Processing Pipeline (2–3 slides)

Goal: Walk through **each step of your software**, mapping directly to the modules you created.

Slide 1 – High–level pipeline:
- Draw a block diagram with arrows:
  - `Input WAV` → `Band-pass filter` → `Windowing` → `Tone detection per window` → `Digit sequence post-processing` → `Decoded phone number`.
- Mention that the system works for **any given DTMF signal file**, not just the provided one.

Slide 2 – `signal_processing.py`:
- Explain **band-pass filter**:
  - 6th-order Butterworth band-pass from ~600 Hz to 1600 Hz.
  - Removes low-frequency noise and high-frequency noise, leaving only DTMF band.
- Explain **Goertzel / tone energy** computation (the `compute_tone_energy` method):
  - You evaluate energy at specific target frequencies instead of doing a full FFT for every window.
  - Mention the idea: multiply–accumulate recursion that focuses on one frequency bin.
- Optional: Short formula / pseudocode for Goertzel (no need for heavy math).

Slide 3 – `tone_analysis.py`:
- Show how each window is analyzed:
  - Apply Hamming window.
  - Compute energy at all row frequencies (697, 770, 852, 941 Hz).
  - Compute energy at all column frequencies (1209, 1336, 1477 Hz).
- Explain decision logic:
  - Choose the **maximum energy** in the low group and high group.
  - Compare each max to the **average energy** in that group to ensure there is a clear dominant tone (`peak_ratio` threshold).
  - Map the pair `(row_freq, col_freq)` to a digit via the tone matrix.

---

## 4. Windowing, Silence and Sequence Detection (2 slides)

Slide 1 – `sequence_decoder.py`:
- Explain the **sliding window** approach:
  - Window length ~120 ms, hop ~50 ms (adjustable).
  - For each window: normalize signal, then run `ToneAnalyzer`.
- Explain how you **reject silence** and noise:
  - Use **power threshold** (`pwr_thr`) to ignore very low-amplitude windows.
  - Windows without a clear peak are treated as silence / non-DTMF.

Slide 2 – Building the digit sequence:
- Explain **minimum hits per digit**:
  - Require a digit to be detected in at least `min_hits` consecutive windows before accepting it.
- Explain **merging and de-duplicating**:
  - Merge windows from the same digit if they are close in time.
  - Remove immediate duplicates (pressing the same digit once should appear once in final sequence).
- Show a small illustration of windows over time and how they become a clean sequence of digits.

---

## 5. Fourier Transform and Frequency Axis (1–2 slides)

Slide 1 – FFT basics:
- Briefly state what you used FFT for in the project:
  - Visualizing the spectrum of each digit.
  - Verifying that each DTMF digit really has two strong peaks at the expected frequencies.
- Show the equation of the discrete Fourier transform (optional, but recommended to satisfy the assignment):
  - Mention how sampling frequency `fs` and window length `N` give frequency resolution:  
    $$f_k = \frac{k \cdot f_s}{N}$$

Slide 2 – X-axis interpretation:
- Explain how you compute FFT frequency axis in code (e.g. `np.fft.rfftfreq(N, 1/fs)`).
- Link this to reading the graphs correctly: peaks at \~697 Hz, \~1336 Hz etc.

---

## 6. Visualization of Results (1–2 slides)

Slide – `visualization.py`:
- Show the main **timeline plot**:
  - Filtered waveform vs time.
  - Colored regions highlighting each detected digit interval.
  - Digit labels over each highlighted region.
- Show the **per-digit FFT panels**:
  - For each detected digit: spectrum between 600–1600 Hz.
  - Annotate or highlight the two main peaks with their approximate frequencies.
- Explain how these visuals helped you debug and validate the algorithm.

---

## 7. Software Demo (1 slide + live demo)

- Show how to run the decoder from the command line:
  - `python dtmf_decoder.py samples/Project1_v4.wav`
- Display the printed output:
  - Sample rate.
  - Decoded raw sequence.
  - Decoded filtered sequence (final phone number).
- If allowed, briefly run the program live and show the plots.

---

## 8. Results and Discussion (1 slide)

- Present the **final decoded phone number** for `Project1_v4.wav`.
- Comment on:
  - How stable the detection is with respect to noise.
  - Whether raw vs filtered decoding differ.
  - Any tricky parts (digits very close together, silence detection, threshold tuning).

---

## 9. Limitations and Possible Improvements (1 slide)

- Possible points:
  - Sensitivity to parameter choices: window size, hop size, power/peak thresholds.
  - Handling overlapping digits or very fast dialing.
  - Real-time implementation vs offline processing.
  - Extending to detect `*` and `#` or other signaling tones.

---

## 10. Individual Q&A Preparation

Each team member should be ready to answer individually:

- **Theory:**
  - What is DTMF? Why two frequencies per digit?
  - What is the purpose of Fourier Transform / FFT here?
  - How do you compute the frequency corresponding to FFT bin `k`?

- **Implementation:**
  - What does `SignalProcessor` do (filter and Goertzel-style energy)?
  - How does `ToneAnalyzer` decide which digit is present in a window?
  - How does `SequenceDecoder` use silence and consecutive detections to form the final sequence?
  - How is the final number printed and visualized?

- **Practical:**
  - What parameters would you tune if the signal is very noisy?
  - How would you adapt this system to decode a different sampling rate or another recording?

Use this file as a checklist while building your slides.
