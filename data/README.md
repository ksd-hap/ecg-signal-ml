# Data

This project uses the [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
from PhysioNet, accessed with the [`wfdb`](https://pypi.org/project/wfdb/) package. See the PhysioNet page
for the database's terms of use, and cite the papers listed in the top-level README if you use it.

**No data is committed to this repository.** Nothing needs to be downloaded by hand.

## How records are fetched

`src/preprocessing.py:load_record(name)` downloads the three files of a record (`.hea`, `.dat`, `.atr`)
from PhysioNet the first time the record is used and stores them in `data/mitdb/` (gitignored). Later
runs reuse the local copy. The exploratory notebooks `01`–`05` instead stream records directly with
`wfdb.rdrecord(name, pn_dir="mitdb")`, so they need internet each time they run.

Downloads occasionally fail with a server error (record 208 did during this project and was never
used); re-running continues from the files already cached.

## What is generated locally (also gitignored)

| Path | Created by | Contents |
|---|---|---|
| `data/mitdb/` | `load_record` | raw records used by the pipeline (the 17 modelling-pool records: 51 files, checked in a fresh clone) |
| `data/processed/beat_features.csv` | notebook `06_feature_extraction` | one row per labelled beat (40,940 rows, 20 columns) |
| `data/processed/error_frame*.pkl`, `error_annotated.pkl` | `scripts/error_analysis_data.py` and `error_analysis.py` | per-beat predictions and diagnostics for the error analysis |

The record pool and the patient split are **committed** (`results/metrics/record_survey.txt`,
`results/metrics/record_split.json`) so that everyone works with the same records and the same
train/test patients.
