"""Tests for src/feature_extraction.py (and the leakage guard in src/splitting.py)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.feature_extraction import (add_record_relative_features, compute_rr_features, extract_beat_features,
                                    map_symbol_to_label, match_annotations, segment_beats)
from src.splitting import check_split

FS = 360
PRE, POST = 72, 144  # 200 ms before, 400 ms after the R peak


def test_label_mapping_follows_the_documented_scheme():
    assert map_symbol_to_label("N") == "Normal"
    assert map_symbol_to_label("R") == "Normal"       # bundle-branch beats are labelled Normal in this project
    assert map_symbol_to_label("V") == "Abnormal"
    assert map_symbol_to_label("A") == "Abnormal"
    assert map_symbol_to_label("/").startswith("Excluded")   # paced
    assert map_symbol_to_label("+").startswith("Excluded")   # rhythm-change marker, not a beat
    assert map_symbol_to_label("?").startswith("Excluded")   # unknown symbols are never guessed


def test_rr_features_are_seconds_to_the_neighbours_with_nan_at_the_ends():
    pre, post = compute_rr_features(np.array([0, 360, 900]), FS)
    assert np.isnan(pre[0]) and pre[1:].tolist() == [1.0, 1.5]
    assert post[:2].tolist() == [1.0, 1.5] and np.isnan(post[2])


def test_segment_beats_centres_the_r_peak_and_drops_beats_near_the_edges():
    signal = np.zeros(2000)
    peaks = np.array([10, 500, 1000, 1990])   # first and last windows would leave the signal
    signal[peaks] = 1.0
    waves, kept = segment_beats(signal, peaks, FS)
    assert kept.tolist() == [1, 2]
    assert waves.shape == (2, PRE + POST)
    assert (waves[:, PRE] == 1.0).all()       # the R peak sits at index 72 of every window


def annotation(samples, symbols):
    return SimpleNamespace(sample=np.array(samples), symbol=np.array(symbols))


def test_match_annotations_takes_the_nearest_beat_within_tolerance_and_ignores_non_beats():
    ann = annotation([100, 400, 405, 700], ["N", "+", "V", "N"])   # '+' at 400 is a rhythm marker, not a beat
    symbols, offsets = match_annotations(np.array([102, 401, 550]), ann, FS, tolerance_ms=50)
    assert symbols == ["N", "V", None]        # 401 matches the V at 405, not the marker; 550 is too far from anything
    assert offsets[0] == pytest.approx(2 / FS * 1000)
    assert np.isnan(offsets[2])


def test_extract_beat_features_on_a_beat_with_known_shape():
    beat = np.zeros(PRE + POST)
    beat[PRE] = 2.0
    beat[PRE + 10] = -0.5
    f = extract_beat_features(beat, FS, PRE, rr_pre_s=0.8, rr_post_s=1.0)
    assert f["rr_ratio"] == pytest.approx(0.8)
    assert f["heart_rate_bpm"] == pytest.approx(75.0)
    assert f["amp_max"] == 2.0 and f["amp_min"] == -0.5
    assert f["r_amplitude_mv"] == 2.0
    assert f["qrs_p2p_mv"] == pytest.approx(2.5)


def test_record_relative_features_cancel_each_patients_own_scale():
    """After dividing by the record's own median, a patient with double the amplitude and slower rate looks the same."""
    base = pd.DataFrame({"rr_pre_s": [0.8, 1.0, 0.9], "rr_post_s": [1.0, 0.9, 0.8],
                         "qrs_p2p_mv": [1.0, 1.2, 1.1], "amp_std": [0.3, 0.35, 0.32]})
    table = pd.concat([base.assign(record="A"), (base * 2.0).assign(record="B")], ignore_index=True)
    out = add_record_relative_features(table)
    for col in ["rr_pre_rel", "rr_post_rel", "qrs_p2p_rel", "amp_std_rel"]:
        a, b = out[out.record == "A"][col].to_numpy(), out[out.record == "B"][col].to_numpy()
        assert a == pytest.approx(b)
        assert np.median(a) == pytest.approx(1.0)


def split_table(train_records, test_records):
    rows = [(r, "train", i % 4) for i, r in enumerate(train_records)] + [(r, "test", -1) for r in test_records]
    t = pd.DataFrame(rows, columns=["record", "split", "cv_fold"])
    t["patient"] = t.record
    t["r_peak_sample"] = np.arange(len(t))
    return t


def test_leakage_guard_accepts_a_clean_split_and_rejects_a_shared_patient():
    check_split(split_table(["100", "101", "102"], ["200"]))
    leaky = split_table(["100", "101", "102"], ["200"])
    leaky.loc[leaky.record == "200", "patient"] = "100"     # same patient on both sides
    with pytest.raises(AssertionError):
        check_split(leaky)
