import numpy as np
import librosa

from signal_processing import SignalProcessor
from tone_analysis import ToneAnalyzer
from sequence_decoder import SequenceDecoder
from visualization import visualize_results


def process_audio_file(path, win_ms=120, hop_ms=50, pwr_thr=8e-6, pk_thr=2.0, min_hits=2, display=True):
    waveform, sample_rate = librosa.load(path, sr=None, mono=True)
    print("sample rate:", sample_rate)
    waveform = waveform.astype(np.float32)

    processor = SignalProcessor(sample_rate)
    filtered_wave = processor.apply_bandpass(waveform, 600, 1600)

    analyzer = ToneAnalyzer(processor)
    decoder = SequenceDecoder(analyzer, sample_rate)

    raw_result, raw_times, raw_freqs = decoder.extract_sequence(
        waveform, win_ms, hop_ms, pwr_thr, pk_thr, min_hits
    )
    filt_result, filt_times, filt_freqs = decoder.extract_sequence(
        filtered_wave, win_ms, hop_ms, pwr_thr, pk_thr, min_hits
    )

    print("Decoded (raw)     :", raw_result)
    print("Decoded (filtered):", filt_result)

    if display:
        visualize_results(
            waveform,
            filtered_wave,
            sample_rate,
            raw_times,
            filt_times,
            raw_result,
            filt_result,
            filt_freqs,
        )

    return raw_result, filt_result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dtmf_decoder.py <path_to_wav>")
        sys.exit(1)
    process_audio_file(
        sys.argv[1], win_ms=120, hop_ms=50, pwr_thr=1e-5, pk_thr=2.5, min_hits=3, display=True
    )


