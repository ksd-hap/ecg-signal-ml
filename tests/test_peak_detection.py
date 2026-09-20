"""Tests for src/peak_detection.py."""
import numpy as np
import pytest

from src.peak_detection import derivative_filter, detect_r_peaks, detection_metrics, match_peaks, moving_window_integration
from src.preprocessing import filter_ecg

FS = 360


def synthetic_ecg(beat_times_s, seconds=30, noise=0.02, seed=0):
    """Narrow QRS-like spikes at known times, plus slow baseline wander and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    signal = 0.3 * np.sin(2 * np.pi * 0.2 * t) + noise * rng.normal(size=len(t))
    for beat in beat_times_s:
        signal += np.exp(-0.5 * ((t - beat) / 0.012) ** 2)
    return signal


def test_derivative_of_a_ramp_is_its_slope():
    ramp = 2.0 * np.arange(100) / FS  # 2 mV per second
    assert derivative_filter(ramp, FS) == pytest.approx(np.full(100, 2.0))


def test_moving_window_integration_smooths_a_spike_to_a_plateau():
    spike = np.zeros(100)
    spike[50] = 10.0
    out = moving_window_integration(spike, 10)
    assert out.sum() == pytest.approx(10.0)   # a moving average conserves the area
    assert out.max() == pytest.approx(1.0)    # 10 / window


def test_detect_r_peaks_finds_every_beat_at_the_right_place():
    beats = np.arange(1.0, 29.0, 0.8)
    peaks = detect_r_peaks(filter_ecg(synthetic_ecg(beats), FS), FS)
    assert len(peaks) == len(beats)
    assert np.abs(peaks - beats * FS).max() <= 0.03 * FS  # within 30 ms


def test_detect_r_peaks_counts_two_spikes_closer_than_the_heart_rate_limit_once():
    """max_hr_bpm=200 means at least 0.3 s between detections; a second spike 0.15 s after the first is not a new beat.

    (Known limitation, deliberately not tested as a pass: thresholds are relative to the signal's own maximum, so on a
    signal with no beats at all the detector still reports noise peaks.)
    """
    beats = [1.0 + 0.8 * k for k in range(30)]
    doubled = sorted(beats + [b + 0.15 for b in beats])
    peaks = detect_r_peaks(filter_ecg(synthetic_ecg(doubled), FS), FS)
    assert len(peaks) == len(beats)
    assert np.diff(peaks).min() >= 0.3 * FS


def test_match_peaks_counts_tp_fp_fn_one_to_one():
    annotated = np.array([100, 300, 500, 700])
    detected = np.array([102, 298, 400, 700])  # 3 matches within tolerance, one spurious (400), one missed (500)
    res = match_peaks(detected, annotated, tolerance_samples=5)
    assert res["tp_annotated"].tolist() == [100, 300, 700]
    assert res["fp"].tolist() == [400]
    assert res["fn"].tolist() == [500]


def test_match_peaks_does_not_reuse_one_detection_for_two_annotations():
    res = match_peaks(np.array([100]), np.array([98, 102]), tolerance_samples=5)
    assert len(res["tp_detected"]) == 1 and len(res["fn"]) == 1


def test_detection_metrics_matches_hand_calculation():
    res = match_peaks(np.array([102, 298, 400, 700]), np.array([100, 300, 500, 700]), 5)
    m = detection_metrics(res)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 1)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)


def test_detection_metrics_is_zero_not_an_error_when_nothing_matches():
    m = detection_metrics(match_peaks(np.array([], dtype=int), np.array([], dtype=int), 5))
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0
