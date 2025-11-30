import numpy as np

from tone_analysis import ToneAnalyzer


class SequenceDecoder:
    def __init__(self, analyzer: ToneAnalyzer, rate: int):
        self.analyzer = analyzer
        self.rate = rate

    def extract_sequence(self, signal, win_ms=100, hop_ms=None, pwr_thr=1e-4, pk_thr=2.0, min_hits=2):
        hop_ms = hop_ms or win_ms // 2
        win_len = int(self.rate * win_ms / 1000)
        hop_len = int(self.rate * hop_ms / 1000)

        if win_len < 16 or hop_len < 1:
            raise ValueError("Invalid window size")

        norm_sig = signal / (np.max(np.abs(signal)) + 1e-12)

        results = []
        intervals = []
        frequencies = []
        last_digit = None
        temp_digit = None
        temp_hits = 0
        temp_pos = None
        temp_freqs = None

        for pos in range(0, len(norm_sig) - win_len + 1, hop_len):
            window = norm_sig[pos : pos + win_len]
            detection = self.analyzer.identify_digit(window, min_power=pwr_thr, peak_ratio=pk_thr)

            if detection is None:
                temp_digit = None
                temp_hits = 0
                last_digit = None  # Reset to allow repeated digits after silence
                temp_freqs = None
                continue

            current, low_f, high_f = detection

            if temp_digit is None:
                temp_digit = current
                temp_hits = 1
                temp_pos = pos
                temp_freqs = (low_f, high_f)
            elif current == temp_digit:
                temp_hits += 1
            else:
                temp_digit = current
                temp_hits = 1
                temp_pos = pos
                temp_freqs = (low_f, high_f)

            if temp_hits >= min_hits and temp_digit != last_digit:
                start_t = temp_pos / self.rate
                end_t = (pos + win_len) / self.rate
                results.append(temp_digit)
                intervals.append((start_t, end_t))
                frequencies.append(temp_freqs)
                last_digit = temp_digit
                temp_digit = None
                temp_hits = 0
                temp_pos = None
                temp_freqs = None

        if not intervals:
            return "", [], []

        consolidated = []
        consol_times = []
        consol_freqs = []

        active_char = results[0]
        active_start = intervals[0][0]
        active_end = intervals[0][1]
        active_freqs = frequencies[0]

        for char, (t_start, t_end), freqs in zip(results[1:], intervals[1:], frequencies[1:]):
            if t_start - active_end <= 0.12:
                active_end = max(active_end, t_end)
            else:
                consolidated.append(active_char)
                consol_times.append((active_start, active_end))
                consol_freqs.append(active_freqs)
                active_char = char
                active_start = t_start
                active_end = t_end
                active_freqs = freqs

        consolidated.append(active_char)
        consol_times.append((active_start, active_end))
        consol_freqs.append(active_freqs)

        # Do NOT collapse consecutive identical digits: each merged block
        # corresponds to a distinct key press, so keep all entries.
        return "".join(consolidated), consol_times, consol_freqs
