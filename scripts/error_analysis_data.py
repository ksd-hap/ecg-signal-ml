"""Build the per-beat frame used by the error analysis (scripts/error_analysis.py).

For every labelled beat of the 17 evaluated patients:
  * held-out probability from each model: development patients via grouped out-of-fold predictions,
    test patients from models fit on all development patients (so every probability is from a model that
    never saw that patient);
  * signal-quality measures computed from the RAW signal (not the filtered one the model saw);
  * whether the R-peak detector landed within 50 ms of the annotation.

It also builds a table of every annotated beat with a `detected` flag, because beats the detector missed and
detector false positives never reach the classifier and are invisible in the beat table.

Run from the repo root:  python scripts/error_analysis_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import out_of_fold_probabilities
from src.feature_extraction import MODEL_FEATURES, beat_annotations, add_record_relative_features, load_beat_table
from src.models import make_gradient_boosting, make_logistic_regression, make_random_forest
from src.peak_detection import detect_r_peaks, match_peaks
from src.preprocessing import filter_ecg, get_ecg_lead, load_record
from src.splitting import apply_split, load_split

OUT_DIR = Path("data/processed")
MODELS = {"lr": make_logistic_regression, "rf": make_random_forest, "gb": make_gradient_boosting}
QRS_EXCLUDE_MS = 60  # samples this close to the R peak are left out of the noise estimate (QRS has real high frequencies)

table = add_record_relative_features(load_beat_table()).sort_values(["record", "r_peak_sample"]).reset_index(drop=True)
data = apply_split(table, load_split())
train, test = data[data.split == "train"], data[data.split == "test"]

# ---- held-out probabilities -------------------------------------------------------------------------------
for key, make in MODELS.items():
    proba = out_of_fold_probabilities(make, data)
    proba = pd.concat([proba, pd.Series(make().fit(train[MODEL_FEATURES], train.is_abnormal)
                                        .predict_proba(test[MODEL_FEATURES])[:, 1], index=test.index)])
    data[f"p_{key}"] = proba
    print("probabilities:", key, flush=True)
assert not data[[f"p_{k}" for k in MODELS]].isna().any().any()
data = data[data.record.isin(data.record.unique())].copy()

# ---- signal quality from the raw signal + detector outcome per annotated beat ---------------------------------
noise_rows = []
annotated_rows = []
for record_name in sorted(data.record.unique()):
    record, annotation = load_record(record_name)
    raw, _ = get_ecg_lead(record)
    fs = record.fs
    filtered = filter_ecg(raw, fs)
    high_freq = raw - sosfiltfilt(butter(4, 40, btype="low", fs=fs, output="sos"), raw)      # content above 40 Hz
    baseline = sosfiltfilt(butter(2, 0.5, btype="low", fs=fs, output="sos"), raw)            # content below 0.5 Hz

    beats = data[data.record == record_name]
    pre, post, excl = int(0.2 * fs), int(0.4 * fs), int(QRS_EXCLUDE_MS / 1000 * fs)
    keep = np.ones(pre + post, dtype=bool)
    keep[pre - excl:pre + excl + 1] = False
    for idx, sample in zip(beats.index, beats.r_peak_sample):
        hf = high_freq[sample - pre:sample + post][keep]
        bl = baseline[sample - pre:sample + post]
        noise_rows.append({"index": idx, "hf_noise_mv": hf.std(), "baseline_range_mv": np.ptp(bl)})

    ann_samples, ann_symbols = beat_annotations(annotation)
    peaks = detect_r_peaks(filtered, fs)
    match = match_peaks(peaks, ann_samples, int(round(0.05 * fs)))
    detected = np.isin(ann_samples, match["tp_annotated"])
    annotated_rows.append(pd.DataFrame({"record": record_name, "sample": ann_samples, "symbol": ann_symbols,
                                        "detected": detected}))
    annotated_rows.append(pd.DataFrame({"record": record_name, "sample": match["fp"], "symbol": "false_positive",
                                        "detected": True}))
    print("signal/detector:", record_name, flush=True)

noise = pd.DataFrame(noise_rows).set_index("index")
data = data.join(noise)
data["hf_noise_rel"] = data.hf_noise_mv / data.qrs_p2p_mv
annotated = pd.concat(annotated_rows, ignore_index=True)

data.to_pickle(OUT_DIR / "error_frame.pkl")
annotated.to_pickle(OUT_DIR / "error_annotated.pkl")
print(data.shape, annotated.shape)
