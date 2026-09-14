# ECG Signal Processing & Arrhythmia Classification — Project Rules

## What this project is

An educational engineering/ML portfolio project: raw ECG (MIT-BIH Arrhythmia
Database) → signal exploration → filtering → R-peak detection → beat
segmentation → labeling → feature extraction → EDA → classical ML →
evaluation → error analysis → clean `src/` code → documented GitHub repo.

**Not** a clinical tool. Never make clinical claims. Never fabricate results
or metrics — only report numbers actually produced by running the code.

## Who I am / how to work with me

- Engineering undergrad, ~3 weeks before university starts, ~3-4 focused
  hours/day. Also juggling a separate hardware project and CV/applications.
- Comfortable with Python, NumPy, Pandas, Jupyter, basic Git. Still building
  ML and DSP/signal-processing knowledge, and ECG/biomedical knowledge is
  beginner level.
- Using AI heavily to accelerate implementation, but wants to **own the
  understanding**, not just paste code. Don't let this become blind
  vibe-coding.
- Don't explain basic Python syntax already known. Do explain DSP/ECG/ML
  concepts the first time they come up (what it is, why we need it, what it
  does to the signal/data, how it's implemented, how we verify it worked).
  Don't turn every response into a textbook chapter — 80/20, learn enough to
  build, hit a wall, learn what's needed, keep going.
- Work sequentially, one stage at a time: explain → implement → how to run →
  expected output → what to verify → debug together → only then move on.
  Don't dump the whole project at once.
- Make reasonable engineering decisions myself and explain the reasoning,
  rather than asking for confirmation on every small choice. Ask only when
  there's a genuinely important tradeoff — then briefly present options and
  recommend one.
- Code style: clean, readable, simple. No premature optimization, no
  unnecessary OOP, no obscure libraries, no hidden/unexplained logic.

## Explicitly avoid (don't let scope creep in)

Deep learning, web apps, cloud infra, MLOps, LLMs/RAG, Docker-for-its-own-sake,
excessive files/abstractions, complex architectures where simple ones work,
extensive hyperparameter search, clinical claims. If something's unnecessary,
say so directly.

## Key engineering principles for this project

- **R-peak detection**: implement a simplified Pan-Tompkins-style pipeline
  (bandpass filter → derivative → square → moving-window integration →
  threshold/candidate detection → peaks), not a black-box call. Validate
  detected peaks against MIT-BIH annotations with a temporal tolerance
  (precision/recall/F1), not exact index matching.
- **Data leakage**: split train/test at the record/patient level, never mix
  beats from the same recording across train and test. This must be explicit
  in the README.
- **Labeling**: don't attempt full multi-class arrhythmia classification
  immediately — start with a manageable scheme (e.g. normal vs abnormal, or a
  small documented set of beat categories) and document the mapping and any
  class-imbalance decisions (exclude/merge rare classes + why).
- **ML**: start simple (logistic regression → decision tree → random forest →
  boosting), sensible defaults before any tuning. Evaluate with
  accuracy/precision/recall/F1/confusion matrix, not accuracy alone.
- **Error analysis** is a major deliverable, not an afterthought: inspect
  actual misclassified beats and explain plausible causes (noise, bad R-peak
  detection, ambiguous annotations, class imbalance, inter-patient
  variability, etc.).

## Repo structure (build progressively, don't scaffold it all up front)

```
ecg-signal-ml/
├── data/README.md
├── notebooks/  (01_data_exploration, 02_signal_processing,
│                03_feature_extraction, 04_machine_learning)
├── src/        (preprocessing.py, peak_detection.py,
│                feature_extraction.py, models.py)
├── results/    (figures/, metrics/)
├── README.md, requirements.txt, .gitignore
```
Notebooks = exploration/experiments/visualization. Reusable logic eventually
moves into `src/` (Phase 14). Don't commit the MIT-BIH dataset to git —
fetch via `wfdb` and document how.

## Priority tiers (cut Tier 3 first if time runs short)

- **Tier 1 (must-have)**: load data, visualize ECG, filter, R-peak detection
  + validation, segmentation, basic labels, feature extraction, 1-2 ML
  models, proper evaluation, README.
- **Tier 2 (strongly desired)**: multiple models, error analysis,
  record/patient-level split, clean `src/`, basic tests, model comparison.
- **Tier 3 (optional, cut first)**: heavy hyperparameter tuning, advanced
  features, deep learning, deployment, web UI, real-time processing.

## README must eventually cover

Overview, Motivation, Dataset, Pipeline, Signal Processing (with plots),
Machine Learning (features/labels/models/methodology), Results (real numbers
only), Error Analysis, Limitations (educational project, no clinical
validation, dataset/algorithm limitations, class imbalance), Future
Improvements (don't overpromise).
