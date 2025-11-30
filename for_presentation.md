# Team 4 – DTMF Decoder Presentation Guide
## 6-7 Slide Structure for 10-Minute Presentation

---

## **SLIDE 1: Title Slide**

### Visuals:
- **Title:** "DTMF Signal Decoder via Fourier Transform"
- **Subtitle:** "Team 4 - Mission: Decode the Bad Guy's Phone Call"
- Team member names
- Date: November 30, 2025

### What to Say (15-20 seconds):
"Good morning/afternoon. We're Team 4, and today we'll present our DTMF decoder that successfully recovered a phone number from a recorded telephone signal. Our mission was to decode what number a suspect dialed using only the recorded button tones."

---

## **SLIDE 2: What is DTMF?**

### Visuals:
- **DTMF Keypad Table:**
  ```
       1209Hz  1336Hz  1477Hz
  697Hz   1      2       3
  770Hz   4      5       6
  852Hz   7      8       9
  941Hz          0
  ```
- Small waveform showing two sine waves combining
- Formula: **Each button = Low Freq + High Freq**

### What to Say (30-40 seconds):
"DTMF stands for Dual-Tone Multi-Frequency. When you press a phone button, it generates two simultaneous tones - one from a low-frequency group and one from a high-frequency group. For example, pressing '5' generates 770Hz and 1336Hz together. Our challenge was to detect these frequency pairs from a noisy recording where peaks might shift ±1-4 Hz from the standard values."

---

## **SLIDE 3: Signal Processing Pipeline**

### Visuals:
Flow diagram with arrows:
```
WAV Input → Bandpass Filter → Sliding Windows → Tone Detection → Sequence Decoder → Phone Number
          (600-1600Hz)      (120ms windows)   (Goertzel)     (Silence+Merge)
```

### What to Say (40-50 seconds):
"Our pipeline has five main stages. First, we apply a 6th-order Butterworth bandpass filter from 600-1600Hz to remove noise outside the DTMF frequency range. Then we use sliding windows of 120ms with 50ms hops to analyze the signal piece by piece. For each window, we use the Goertzel algorithm - a computationally efficient method that detects energy at specific target frequencies without computing a full FFT. Then we identify which digit was pressed, and finally we use silence detection to separate individual digits and build the complete phone number."

---

## **SLIDE 4: Tone Detection Algorithm**

### Visuals:
- **Left side - Flowchart:**
  - Compute energy at 4 low frequencies (697, 770, 852, 941)
  - Compute energy at 3 high frequencies (1209, 1336, 1477)
  - Find max in each group
  - Check thresholds: Power > 1e-5, Peak ratio > 2.5
  - Map to digit
- **Right side:** Example spectrum showing two clear peaks

### What to Say (40-50 seconds):
"For each window, we compute the energy at all seven DTMF frequencies using the Goertzel algorithm. We then find the strongest frequency in the low group and the high group. To avoid false positives from noise, we apply three validation checks: First, the window must have sufficient power. Second, the detected peaks must be at least 2.5 times stronger than the average energy in their group. Third, the low and high frequency peaks must be balanced - preventing detection when only one tone is present. If all checks pass, we map the frequency pair to its corresponding digit."

---

## **SLIDE 5: Sequence Decoder & Silence Detection**

### Visuals:
- **Top:** Timeline showing waveform with colored regions marking detected digits
- **Bottom - Diagram showing logic:**
  - Silence (low power) → Reset
  - 3+ consecutive detections → Accept digit
  - Gap < 120ms → Same digit (merge)
  - Gap > 120ms → New digit

### What to Say (40-50 seconds):
"The key challenge is separating individual button presses. DTMF signals are naturally separated by silence between digits. We exploit this by requiring a digit to appear in at least 3 consecutive windows before accepting it - this prevents noise spikes from being misread. When we encounter silence or low power, we reset. After detecting all windows, we merge detections that are within 120ms of each other into a single digit press. This consolidation step handles the fact that a single button press might span multiple analysis windows."

---

## **SLIDE 6: Results & FFT Validation**

### Visuals:
- **Top Left:** Decoded phone number displayed large: **"Result: [your decoded number]"**
- **Top Right:** Full signal waveform with digit markers
- **Bottom:** 3-4 example FFT spectrums for different digits, each showing two clear peaks with frequency labels

### What to Say (40-50 seconds):
"Our decoder successfully extracted the phone number: [state the number]. To validate our results, we used FFT analysis on each detected digit segment. These spectrum plots show clear dual peaks at the expected DTMF frequencies. For example, this digit shows peaks at 770Hz and 1336Hz, confirming it's a '5'. The frequency resolution of our FFT is calculated as fs divided by window length N, which gave us sufficient precision to identify each tone accurately despite minor frequency shifts from noise."

---

## **SLIDE 7: Key Equations & Conclusion**

### Visuals:
- **Top Section - Key Equations:**
  ```
  Goertzel Coefficient: 2·cos(2πk/N)
  FFT Frequency Axis: f_k = k·fs/N
  Bandpass Filter: 6th-order Butterworth [600, 1600] Hz
  ```
- **Bottom Section - Achievements:**
  - ✓ Automatic decoding of any DTMF signal
  - ✓ Noise-robust detection
  - ✓ Handles variable silence gaps
  - ✓ Visualizations for debugging

### What to Say (30-40 seconds):
"To summarize the key technical components: We used the Goertzel algorithm for efficient single-frequency energy computation, applied Butterworth filtering to isolate the DTMF band, and calculated FFT frequency bins using k times sampling rate divided by window length. Our final system can automatically decode any DTMF signal, handles realistic noise conditions, and provides comprehensive visualizations. The modular architecture in Python allows easy parameter tuning for different recording conditions. Thank you, we're ready for questions."

---

## **INDIVIDUAL Q&A PREPARATION**

### Theory Questions:
1. **"What is DTMF? Why two frequencies per digit?"**
   - Answer: "DTMF uses two simultaneous tones to uniquely identify each key. This dual-tone approach reduces false detection from background noise or speech, since it's unlikely random noise will produce two exact frequencies simultaneously."

2. **"What is the purpose of Fourier Transform / FFT here?"**
   - Answer: "FFT converts our time-domain signal into frequency-domain, allowing us to see which frequencies are present and their magnitudes. This lets us identify the two specific DTMF tones for each digit."

3. **"How do you compute the frequency corresponding to FFT bin k?"**
   - Answer: "The frequency is f_k = k × fs / N, where fs is the sampling rate and N is the window length. This gives us the frequency resolution of our analysis."

### Implementation Questions:
1. **"What does SignalProcessor do?"**
   - Answer: "It applies a 6th-order Butterworth bandpass filter to isolate DTMF frequencies, and implements the Goertzel algorithm to compute energy at specific target frequencies efficiently."

2. **"How does ToneAnalyzer decide which digit is present?"**
   - Answer: "It computes energy at all 7 DTMF frequencies, finds the strongest in low and high groups, validates using power and peak ratio thresholds, and maps the frequency pair to a digit using our tone matrix."

3. **"How does SequenceDecoder use silence and consecutive detections?"**
   - Answer: "It uses sliding windows, requires 3+ consecutive detections before accepting a digit, treats low-power windows as silence to separate digits, and merges nearby detections to prevent duplicates."

4. **"How is the final number printed and visualized?"**
   - Answer: "The sequence is printed as a string of digits. Visualization shows the waveform with color-coded regions for each digit, plus FFT spectrums showing the two peaks for each detected tone."

### Practical Questions:
1. **"What parameters would you tune if the signal is very noisy?"**
   - Answer: "I would increase the power threshold and peak ratio to be more selective, possibly increase minimum consecutive hits, and adjust the bandpass filter bandwidth to be narrower if I know the exact frequency range."

2. **"How would you adapt this for different sampling rates?"**
   - Answer: "The system automatically adapts - our Goertzel and FFT computations use the sampling rate parameter. We might need to adjust window length in milliseconds to maintain the same time resolution."

3. **"Why Goertzel instead of full FFT?"**
   - Answer: "Goertzel is more efficient when you only need a few specific frequencies. We only care about 7 DTMF frequencies, not the entire spectrum, so Goertzel saves computation time."

4. **"How do you handle repeated digits like '55'?"**
   - Answer: "The silence gap between button presses resets our detector. When the user releases and presses again, the low-power silence windows clear the previous detection, allowing the same digit to be recognized again."

---

## **TIMING BREAKDOWN**
- Slide 1: 20 seconds
- Slide 2: 40 seconds
- Slide 3: 50 seconds
- Slide 4: 50 seconds
- Slide 5: 50 seconds
- Slide 6: 50 seconds
- Slide 7: 40 seconds
- **Total: ~5 minutes** (leaves 5 minutes for demo + Q&A)

---

## **PRESENTATION TIPS**
- Speak clearly and maintain eye contact
- Point to specific parts of diagrams while explaining
- Practice transitions between slides
- Have the code ready to run for live demo
- Each team member should understand all slides for Q&A
- Be confident - your implementation is solid!
