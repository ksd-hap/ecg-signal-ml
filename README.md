# ECG Signal Processing & Arrhythmia Classification

**Status: work in progress — Stage 1 (project setup) complete.**

An educational engineering project that takes raw ECG recordings from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
through the full pipeline from signal to prediction: signal exploration,
filtering, R-peak detection, beat segmentation and labeling, feature
extraction, classical machine learning, and evaluation.

This is **not** a clinical or diagnostic tool. It is a portfolio project
demonstrating physiological signal processing and machine learning
engineering practice — no claims here should be read as medical advice or
validated diagnostic performance.

## Pipeline

```
Raw ECG (MIT-BIH)
  -> Signal exploration
  -> Bandpass filtering
  -> R-peak detection (Pan-Tompkins-style) + validation against annotations
  -> Heartbeat segmentation
  -> Beat labeling
  -> Feature extraction
  -> Exploratory data analysis
  -> Machine learning (record-level train/test split)
  -> Evaluation
  -> Error analysis
```

## Repository structure

```
ecg-signal-ml/
├── data/           # dataset access notes (raw data is not committed)
├── notebooks/      # exploration, experiments, visualization
├── src/            # reusable, tested pipeline code (added as the project matures)
├── results/        # figures and metrics produced by the notebooks
├── requirements.txt
└── learning_notes.md   # personal learning log, separate from this README
```

## Setup

Requires Python 3.14 (installed via [python.org](https://www.python.org/downloads/)
or the Windows Store).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

See [`data/README.md`](data/README.md). Data is fetched on demand via the
`wfdb` package rather than committed to the repository.

## Results

*(populated once models have actually been trained and evaluated — no
fabricated numbers)*

## Limitations

- Educational project, not a clinically validated diagnostic system.
- Uses a subset of MIT-BIH Arrhythmia Database records.
- Relies on classical, engineered-feature ML rather than deep learning on
  raw signal.

## Learning notes

See [`learning_notes.md`](learning_notes.md) for a running log of concepts
and decisions worked through at each stage.
