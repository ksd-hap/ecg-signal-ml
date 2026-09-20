"""From R peaks to a labelled table: label mapping, beat segmentation, per-beat features, the saved beat table."""
from collections import Counter

import numpy as np
import pandas as pd

from .peak_detection import detect_r_peaks
from .preprocessing import DEFAULT_DATA_DIR, filter_ecg, get_ecg_lead, load_filtered_record, load_record

DEFAULT_TABLE_PATH = DEFAULT_DATA_DIR.parent / "processed" / "beat_features.csv"

# AAMI EC57 grouping of MIT-BIH beat symbols, merged to Normal / Abnormal (see notebooks/05_beat_labeling.ipynb)
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
ABNORMAL_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F"}
PACED_SYMBOLS = {"/", "f", "Q"}
NON_BEAT_SYMBOLS = {"+", "~", "|", "x", "!", "[", "]", '"', "'"}
BEAT_SYMBOLS = NORMAL_SYMBOLS | ABNORMAL_SYMBOLS | PACED_SYMBOLS

FEATURE_COLUMNS = [
    "rr_pre_s", "rr_post_s", "rr_ratio", "heart_rate_bpm",
    "amp_mean", "amp_std", "amp_min", "amp_max", "energy_mv2s",
    "r_amplitude_mv", "dominant_deflection_mv", "qrs_p2p_mv", "qrs_fwhm_ms",
]


# Features used for modelling: the 13 extracted features minus four duplicates found in the EDA
# (heart_rate_bpm = 60/rr_pre_s; energy_mv2s ~ amp_std; r_amplitude_mv and dominant_deflection_mv ~ amp_max),
# plus four record-relative versions of the features whose absolute scale depends on the patient.
RELATIVE_SOURCES = {
    "rr_pre_rel": "rr_pre_s",
    "rr_post_rel": "rr_post_s",
    "qrs_p2p_rel": "qrs_p2p_mv",
    "amp_std_rel": "amp_std",
}
MODEL_FEATURES = [
    "rr_pre_s", "rr_post_s", "rr_ratio", "rr_pre_rel", "rr_post_rel",
    "amp_mean", "amp_std", "amp_min", "amp_max", "qrs_p2p_mv", "qrs_fwhm_ms",
    "qrs_p2p_rel", "amp_std_rel",
]


def add_record_relative_features(table):
    """Divide selected features by that record's own median, so a patient's scale (heart rate, electrode gain) cancels.

    Uses only the record's own beats and no labels, so it must be computed per record and never across a split:
    a record lives entirely in train or entirely in test, which keeps this leak-free.
    """
    table = table.copy()
    for new_name, source in RELATIVE_SOURCES.items():
        table[new_name] = table[source] / table.groupby("record", observed=True)[source].transform("median")
    return table


# Rhythm-context features (defined and pre-registered before any result was seen; see notebook 12).
RHYTHM_CONTEXT_FEATURES = ["rr_hist_cv", "rr_pre_vs_hist"]
MODEL_FEATURES_RHYTHM = MODEL_FEATURES + RHYTHM_CONTEXT_FEATURES


def add_rhythm_context_features(table, window=8, min_periods=4, bounds=(0.30, 2.0)):
    """Describe the rhythm *before* each beat, so one premature beat can be told from an irregular rhythm.

    Both features use the `window` RR intervals that precede the beat's own pre-interval (that is, the intervals
    ending at the previous `window` beats), clipped to fixed physiological bounds:
      rr_hist_cv     - coefficient of variation (std / mean) of those intervals: high in an irregular rhythm
      rr_pre_vs_hist - the beat's own pre-interval divided by their mean: well below 1 for a premature beat
    The first beats of a record have too little history; they get that record's own median. Uses only RR intervals
    of the same record (no labels, nothing fitted), and the table must be sorted by record and time.
    """
    table = table.copy()
    by_record = table["record"].astype(str)
    ordered = table.groupby(by_record)["r_peak_sample"].apply(lambda s: s.is_monotonic_increasing)
    assert ordered.all(), "table must be sorted by time within each record"

    rr = table["rr_pre_s"].clip(*bounds)
    rolling = rr.groupby(by_record).shift(1).groupby(by_record).rolling(window, min_periods=min_periods)
    mean = rolling.mean().reset_index(level=0, drop=True).reindex(table.index)
    std = rolling.std().reset_index(level=0, drop=True).reindex(table.index)
    table["rr_hist_cv"] = std / mean
    table["rr_pre_vs_hist"] = rr / mean
    for name in RHYTHM_CONTEXT_FEATURES:
        table[name] = table[name].fillna(table.groupby("record", observed=True)[name].transform("median"))
    return table


def map_symbol_to_label(symbol):
    """Map a MIT-BIH annotation symbol to our binary label, or an explicit exclusion reason."""
    if symbol in NORMAL_SYMBOLS:
        return "Normal"
    if symbol in ABNORMAL_SYMBOLS:
        return "Abnormal"
    if symbol in PACED_SYMBOLS:
        return "Excluded (paced/unclassifiable)"
    if symbol in NON_BEAT_SYMBOLS:
        return "Excluded (not a beat)"
    return "Excluded (unrecognized symbol)"


def compute_rr_features(r_peaks, fs):
    """Seconds to the previous / next R peak for every peak; NaN where there is no neighbour."""
    rr = np.diff(r_peaks) / fs
    return np.concatenate([[np.nan], rr]), np.concatenate([rr, [np.nan]])


def segment_beats(filtered_signal, r_peaks, fs, pre_ms=200, post_ms=400):
    """Cut a fixed window around each R peak; peaks whose window leaves the signal are dropped.

    Returns (waveforms, kept_idx) where kept_idx indexes into r_peaks.
    """
    pre = int(round(pre_ms / 1000 * fs))
    post = int(round(post_ms / 1000 * fs))
    keep = (r_peaks - pre >= 0) & (r_peaks + post <= len(filtered_signal))
    kept_idx = np.flatnonzero(keep)
    waveforms = np.array([filtered_signal[p - pre:p + post] for p in r_peaks[kept_idx]]).reshape(-1, pre + post)
    return waveforms, kept_idx


def match_annotations(peak_samples, annotation, fs, tolerance_ms=150):
    """Nearest reference *beat* annotation to each detected peak (symbol, |offset| in ms); (None, nan) if none within tolerance.

    Non-beat markers (rhythm change, noise, ...) never label a beat, so only beat symbols take part in matching.
    """
    ann_samples = np.asarray(annotation.sample)
    ann_symbols = np.asarray(annotation.symbol)
    is_beat = np.isin(ann_symbols, list(BEAT_SYMBOLS))
    beat_samples, beat_symbols = ann_samples[is_beat], ann_symbols[is_beat]
    tolerance = int(round(tolerance_ms / 1000 * fs))

    symbols, offsets_ms = [], []
    for p in peak_samples:
        j = np.searchsorted(beat_samples, p)
        candidates = [k for k in (j - 1, j) if 0 <= k < len(beat_samples)]
        best = min(candidates, key=lambda k: abs(int(beat_samples[k]) - int(p)), default=None)
        if best is not None and abs(int(beat_samples[best]) - int(p)) <= tolerance:
            symbols.append(str(beat_symbols[best]))
            offsets_ms.append(abs(int(beat_samples[best]) - int(p)) / fs * 1000)
        else:
            symbols.append(None)
            offsets_ms.append(np.nan)
    return symbols, np.array(offsets_ms)


def dominant_deflection(beat, fs, pre_samples, qrs_half_ms=100):
    """Largest-magnitude deflection in the QRS region, relative to the beat's median (baseline).

    Returns the baseline, the deflection's signed height, and its sample indices within `beat`:
    the extremum plus the left/right edges of its full width at half maximum (FWHM).
    """
    half = int(round(qrs_half_ms / 1000 * fs))
    start = pre_samples - half
    deviation = beat[start:pre_samples + half + 1] - np.median(beat)
    k = int(np.argmax(np.abs(deviation)))
    height = deviation[k]

    # FWHM: contiguous same-sign samples that stay at least half as far from baseline as the extremum
    above_half = (np.sign(deviation) == np.sign(height)) & (np.abs(deviation) >= 0.5 * abs(height))
    left = right = k
    while left > 0 and above_half[left - 1]:
        left -= 1
    while right < len(above_half) - 1 and above_half[right + 1]:
        right += 1
    return {"baseline": np.median(beat), "height": height,
            "peak_idx": start + k, "left_idx": start + left, "right_idx": start + right}


def extract_beat_features(beat, fs, pre_samples, rr_pre_s, rr_post_s, qrs_half_ms=100):
    """Feature dict for one beat waveform whose R peak sits at index `pre_samples`."""
    half = int(round(qrs_half_ms / 1000 * fs))
    qrs = beat[pre_samples - half:pre_samples + half + 1]
    defl = dominant_deflection(beat, fs, pre_samples, qrs_half_ms)

    return {
        "rr_pre_s": rr_pre_s,
        "rr_post_s": rr_post_s,
        "rr_ratio": rr_pre_s / rr_post_s,
        "heart_rate_bpm": 60.0 / rr_pre_s,
        "amp_mean": beat.mean(),
        "amp_std": beat.std(),
        "amp_min": beat.min(),
        "amp_max": beat.max(),
        "energy_mv2s": np.sum(beat ** 2) / fs,
        "r_amplitude_mv": beat[pre_samples],
        "dominant_deflection_mv": defl["height"],
        "qrs_p2p_mv": qrs.max() - qrs.min(),
        "qrs_fwhm_ms": (defl["right_idx"] - defl["left_idx"] + 1) / fs * 1000,
    }


def build_beat_table(record_name, filtered_signal, r_peaks, fs, annotation,
                     pre_ms=200, post_ms=400, label_tolerance_ms=150):
    """One row per labeled heartbeat: metadata + FEATURE_COLUMNS. Returns (DataFrame, summary dict)."""
    r_peaks = np.asarray(r_peaks)
    pre_samples = int(round(pre_ms / 1000 * fs))
    rr_pre, rr_post = compute_rr_features(r_peaks, fs)
    waveforms, kept_idx = segment_beats(filtered_signal, r_peaks, fs, pre_ms, post_ms)
    symbols, offsets_ms = match_annotations(r_peaks[kept_idx], annotation, fs, label_tolerance_ms)

    summary = Counter(n_detected=len(r_peaks), n_dropped_boundary=len(r_peaks) - len(kept_idx))
    rows = []
    for beat, idx, symbol, offset in zip(waveforms, kept_idx, symbols, offsets_ms):
        if np.isnan(rr_pre[idx]) or np.isnan(rr_post[idx]):
            summary["n_dropped_no_rr_neighbour"] += 1
            continue
        if symbol is None:
            summary["n_dropped_no_annotation"] += 1
            continue
        label = map_symbol_to_label(symbol)
        if label not in ("Normal", "Abnormal"):
            summary["n_dropped_paced_or_other"] += 1
            continue
        row = {
            "record": record_name,
            "r_peak_sample": int(r_peaks[idx]),
            "time_s": r_peaks[idx] / fs,
            "symbol": symbol,
            "label": label,
            "is_abnormal": int(label == "Abnormal"),
            "ann_offset_ms": offset,
        }
        row.update(extract_beat_features(beat, fs, pre_samples, rr_pre[idx], rr_post[idx]))
        rows.append(row)

    summary = dict(summary)
    summary["n_rows"] = len(rows)
    summary["unrecognized_symbols"] = sorted(set(annotation.symbol) - BEAT_SYMBOLS - NON_BEAT_SYMBOLS)
    return pd.DataFrame(rows), summary


def load_beat_table(path=DEFAULT_TABLE_PATH):
    """Load the saved beat table; record ids and symbols are read as strings, never guessed as numbers."""
    return pd.read_csv(path, dtype={"record": str, "symbol": str, "label": str})


def beat_annotations(annotation):
    """(sample, symbol) arrays of the annotations that mark beats (dropping rhythm changes, noise flags, etc.)."""
    is_beat = np.isin(np.asarray(annotation.symbol), list(BEAT_SYMBOLS))
    return np.asarray(annotation.sample)[is_beat], np.asarray(annotation.symbol)[is_beat]


def get_beat_waveform(record_name, r_peak_sample, pre_ms=200, post_ms=400):
    """(time_ms relative to R peak, filtered waveform) for one beat, for plotting individual beats."""
    filtered, fs = load_filtered_record(record_name)
    pre = int(round(pre_ms / 1000 * fs))
    post = int(round(post_ms / 1000 * fs))
    return np.arange(-pre, post) / fs * 1000, filtered[r_peak_sample - pre:r_peak_sample + post]


def process_record(record_name, data_dir=DEFAULT_DATA_DIR, **table_kwargs):
    """Full pipeline for one record: load -> filter -> detect R peaks -> segment -> label -> features."""
    record, annotation = load_record(record_name, data_dir)
    signal, _ = get_ecg_lead(record)
    filtered = filter_ecg(signal, record.fs)
    r_peaks = detect_r_peaks(filtered, record.fs)
    return build_beat_table(record_name, filtered, r_peaks, record.fs, annotation, **table_kwargs)
