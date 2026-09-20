"""Tests for src/preprocessing.py. Synthetic signals only: no dataset download needed."""
import numpy as np
import pytest

from src.preprocessing import bandpass_filter, filter_ecg, get_ecg_lead

FS = 360


def sine(freq_hz, seconds=20):
    t = np.arange(seconds * FS) / FS
    return np.sin(2 * np.pi * freq_hz * t)


def steady_state_amplitude(x):
    """Amplitude away from the edges, where filter start-up effects live."""
    return np.abs(x[5 * FS:-5 * FS]).max()


def test_bandpass_keeps_ecg_band_and_removes_wander_and_noise():
    """10 Hz is inside 0.5-40 Hz and should survive; 0.05 Hz baseline wander and 120 Hz noise should not."""
    assert steady_state_amplitude(bandpass_filter(sine(10), 0.5, 40, FS)) == pytest.approx(1.0, abs=0.05)
    assert steady_state_amplitude(bandpass_filter(sine(0.05), 0.5, 40, FS)) < 0.05
    assert steady_state_amplitude(bandpass_filter(sine(120), 0.5, 40, FS)) < 0.05


def test_bandpass_removes_a_constant_offset():
    filtered = bandpass_filter(np.full(20 * FS, 3.0), 0.5, 40, FS)
    assert steady_state_amplitude(filtered) < 1e-3


def test_bandpass_is_zero_phase():
    """filtfilt must not shift waveforms in time: a symmetric pulse keeps its peak position."""
    t = np.arange(10 * FS)
    pulse = np.exp(-0.5 * ((t - 5 * FS) / 8) ** 2)
    assert np.argmax(bandpass_filter(pulse, 0.5, 40, FS)) == 5 * FS


def test_filter_ecg_uses_the_project_settings():
    x = np.random.default_rng(0).normal(size=10 * FS)
    assert np.array_equal(filter_ecg(x, FS), bandpass_filter(x, 0.5, 40, FS, order=4))


class FakeRecord:
    sig_name = ["MLII", "V1"]
    p_signal = np.array([[1.0, 10.0], [2.0, 20.0]])


def test_get_ecg_lead_picks_the_named_lead():
    signal, name = get_ecg_lead(FakeRecord(), "V1")
    assert name == "V1" and signal.tolist() == [10.0, 20.0]


def test_get_ecg_lead_fails_loudly_when_lead_is_missing():
    with pytest.raises(ValueError):
        get_ecg_lead(FakeRecord(), "V5")
