"""Survey every MIT-BIH record: annotation counts and R-peak detector quality.

Used to choose the record pool (see notebooks/06_feature_extraction.ipynb). Streams each record from
PhysioNet without caching, so it takes several minutes. Run from the repo root:

    python scripts/survey_records.py > results/metrics/record_survey.txt
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import wfdb

from src.feature_extraction import ABNORMAL_SYMBOLS, BEAT_SYMBOLS, NORMAL_SYMBOLS
from src.peak_detection import detect_r_peaks, detection_metrics, match_peaks
from src.preprocessing import filter_ecg

records = wfdb.get_record_list("mitdb")
print("records in mitdb:", len(records), flush=True)
print(f"{'rec':>4} {'leads':>10} {'normal':>6} {'abn':>5} {'paced':>5} | {'TP':>5} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}  secs", flush=True)

for name in records:
    t0 = time.time()
    try:
        rec = wfdb.rdrecord(name, pn_dir="mitdb")
        ann = wfdb.rdann(name, "atr", pn_dir="mitdb")
    except Exception as exc:
        print(f"{name:>4} LOAD ERROR {exc}", flush=True)
        continue

    syms = np.array(ann.symbol)
    samples = np.array(ann.sample)
    n_norm = int(np.isin(syms, list(NORMAL_SYMBOLS)).sum())
    n_abn = int(np.isin(syms, list(ABNORMAL_SYMBOLS)).sum())
    n_paced = int(np.isin(syms, ["/", "f"]).sum())

    line = f"{name:>4} {'+'.join(rec.sig_name):>10} {n_norm:>6} {n_abn:>5} {n_paced:>5} |"
    if "MLII" in rec.sig_name and n_paced == 0:
        fs = rec.fs
        signal = rec.p_signal[:, rec.sig_name.index("MLII")]
        filtered = filter_ecg(signal, fs)
        peaks = detect_r_peaks(filtered, fs)
        beats = samples[np.isin(syms, list(BEAT_SYMBOLS))]
        res = match_peaks(peaks, beats, int(round(0.05 * fs)))
        m = detection_metrics(res)
        line += f" {m['tp']:>5} {m['fp']:>4} {m['fn']:>4} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
    else:
        line += "  skipped (" + ("paced" if n_paced else "no MLII") + ")"
    print(line + f"  {time.time() - t0:.0f}", flush=True)

print("DONE", flush=True)
