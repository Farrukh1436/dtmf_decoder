import numpy as np
import matplotlib.pyplot as plt


def visualize_results(raw, filt, sr, raw_t, filt_t, raw_str, filt_str):
    timeline = np.arange(len(raw)) / sr
    digits_with_times = list(zip(filt_str, filt_t))
    num_digits = len(digits_with_times)

    print("\n" + "=" * 70)
    print("DTMF DECODER RESULTS".center(70))
    print("=" * 70)
    print(f"Detected Sequence: {filt_str}")
    print(f"Total Digits: {num_digits}")
    print(f"Audio Duration: {timeline[-1]:.3f} seconds")
    print("=" * 70 + "\n")

    fig = plt.figure(figsize=(18, 6))

    ax_full = plt.subplot2grid((3, num_digits), (0, 0), colspan=num_digits, rowspan=2)
    ax_full.plot(timeline, filt, color="#2E86AB", alpha=0.8, linewidth=0.8)

    colors = plt.cm.Set3(np.linspace(0, 1, num_digits))
    for idx, (s, e) in enumerate(filt_t):
        ax_full.axvspan(s, e, color=colors[idx], alpha=0.3)
        mid_point = (s + e) / 2
        ax_full.text(
            mid_point,
            ax_full.get_ylim()[1] * 0.85,
            filt_str[idx],
            ha="center",
            fontsize=16,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=colors[idx],
                edgecolor="black",
                linewidth=2,
            ),
        )

    ax_full.set_ylabel("Amplitude", fontsize=11, fontweight="bold")
    ax_full.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax_full.set_title("DTMF Signal Timeline", fontsize=14, fontweight="bold", pad=15)
    ax_full.grid(True, alpha=0.4, linestyle="--")
    ax_full.spines["top"].set_visible(False)
    ax_full.spines["right"].set_visible(False)

    for idx, (digit, (start_t, end_t)) in enumerate(digits_with_times):
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        slice_data = filt[start_idx:end_idx]

        n_samples = len(slice_data)
        fft_vals = np.fft.rfft(slice_data * np.hamming(n_samples))
        fft_freqs = np.fft.rfftfreq(n_samples, 1 / sr)
        fft_mag = np.abs(fft_vals)

        peak_indices = np.argsort(fft_mag)[-5:]
        peak_freqs = fft_freqs[peak_indices]
        peak_mags = fft_mag[peak_indices]
        top_peaks = sorted(zip(peak_freqs, peak_mags), key=lambda x: x[1], reverse=True)[:2]

        sorted_freqs = sorted([top_peaks[0][0], top_peaks[1][0]])
        low_freq = sorted_freqs[0]
        high_freq = sorted_freqs[1]

        ax_fft = plt.subplot2grid((3, num_digits), (2, idx))
        ax_fft.fill_between(fft_freqs, fft_mag, color=colors[idx], alpha=0.6)
        ax_fft.plot(fft_freqs, fft_mag, color="#A23B72", linewidth=1.5)

        ax_fft.set_xlim(600, 1600)
        ax_fft.set_ylim(bottom=0)
        ax_fft.set_xlabel("", fontsize=9)
        ax_fft.set_ylabel("")
        ax_fft.set_title(
            f"'{digit}'",
            fontsize=16,
            fontweight="bold",
            pad=8,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=colors[idx],
                edgecolor="black",
                linewidth=2,
            ),
        )
        ax_fft.grid(True, alpha=0.3, linestyle=":")
        ax_fft.tick_params(labelsize=7)
        ax_fft.set_xticks([])
        ax_fft.set_yticks([])

        freq_text = f"Low: {low_freq:.0f} Hz\nHigh: {high_freq:.0f} Hz"
        ax_fft.text(
            0.5,
            -0.25,
            freq_text,
            transform=ax_fft.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="gray",
                linewidth=1,
            ),
        )

        print(
            f"Digit '{digit}': {start_t:.3f}s - {end_t:.3f}s | "
            f"Low: {low_freq:.0f}Hz | High: {high_freq:.0f}Hz"
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.12)
    plt.show()
    print("\n" + "=" * 70 + "\n")
