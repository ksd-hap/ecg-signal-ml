import numpy as np
from scipy.signal import find_peaks


def derivative_filter(signal, fs):
    """Approximate slope (dV/dt) of the signal using a central difference."""
    return np.gradient(signal, 1 / fs)


def moving_window_integration(signal, window_size):
    """Moving average (boxcar) over `window_size` samples, same length as input."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")


def detect_r_peaks(filtered_signal, fs, qrs_duration_seconds=0.1,
                   height_frac=0.2, prominence_frac=0.2, max_hr_bpm=200):
    """Simplified Pan-Tompkins: derivative -> square -> integrate -> find_peaks -> search back on the filtered signal."""
    squared = derivative_filter(filtered_signal, fs) ** 2
    window = int(round(qrs_duration_seconds * fs))
    integrated = moving_window_integration(squared, window)

    min_distance = int(round((60 / max_hr_bpm) * fs))
    candidates, _ = find_peaks(
        integrated,
        distance=min_distance,
        height=height_frac * np.max(integrated),
        prominence=prominence_frac * np.max(integrated),
    )

    search_radius = window // 2
    refined = []
    for c in candidates:
        start = max(0, c - search_radius)
        end = min(len(filtered_signal), c + search_radius)
        refined.append(start + np.argmax(filtered_signal[start:end]))
    return np.unique(refined)


def match_peaks(detected_peaks, annotated_peaks, tolerance_samples):
    """One-to-one TP/FP/FN matching via a two-pointer sweep over two sorted sequences."""
    detected_peaks = np.sort(np.asarray(detected_peaks))
    annotated_peaks = np.sort(np.asarray(annotated_peaks))

    i, j = 0, 0  # i -> annotated_peaks pointer, j -> detected_peaks pointer
    tp_detected, tp_annotated, fp, fn = [], [], [], []

    while i < len(annotated_peaks) and j < len(detected_peaks):
        diff = int(detected_peaks[j]) - int(annotated_peaks[i])
        if abs(diff) <= tolerance_samples:
            tp_detected.append(detected_peaks[j])
            tp_annotated.append(annotated_peaks[i])
            i += 1
            j += 1
        elif diff < -tolerance_samples:
            fp.append(detected_peaks[j])
            j += 1
        else:
            fn.append(annotated_peaks[i])
            i += 1

    fp.extend(detected_peaks[j:])
    fn.extend(annotated_peaks[i:])

    return {
        "tp_detected": np.array(tp_detected, dtype=int),
        "tp_annotated": np.array(tp_annotated, dtype=int),
        "fp": np.array(fp, dtype=int),
        "fn": np.array(fn, dtype=int),
    }
