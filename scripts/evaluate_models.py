"""Evaluate all three models with the same metrics, on the same split.

Task is BINARY (Normal = 0, Abnormal = 1), so precision / recall / F1 use the abnormal class as the positive
class ("binary" averaging). Macro-averaged F1 (mean of the two classes' F1) is added only as a reference.

Two evaluations:
  * development: grouped 4-fold cross-validation over the 12 development patients (out-of-fold predictions).
    This is the comparison that was allowed to inform model selection.
  * test: models fit on all 12 development patients, scored on the 5 held-out patients. The selection was frozen
    before this (results/metrics/frozen_protocol.json); the tree models are shown for context only.

Run from the repo root:  python scripts/evaluate_models.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import out_of_fold_probabilities, summarize
from src.feature_extraction import MODEL_FEATURES, add_record_relative_features, load_beat_table
from src.models import make_gradient_boosting, make_logistic_regression, make_random_forest
from src.splitting import apply_split, load_split

MODELS = {
    "Logistic regression": make_logistic_regression,
    "Random forest": make_random_forest,
    "Gradient boosting": make_gradient_boosting,
}
METRICS_DIR = Path("results/metrics")
FIGURE_PATH = Path("results/figures/27_confusion_matrices.png")

table = add_record_relative_features(load_beat_table()).sort_values(["record", "r_peak_sample"]).reset_index(drop=True)
data = apply_split(table, load_split())
train, test = data[data.split == "train"], data[data.split == "test"]


def row(y_true, proba):
    """Metrics at threshold 0.5 plus macro-F1 for reference."""
    out = summarize(y_true, proba)
    pred = (np.asarray(proba) >= 0.5).astype(int)
    out["macro_f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
    return out


def baseline_row(y_true):
    """Always predict Normal: the accuracy paradox reference."""
    y_true = np.asarray(y_true)
    return row(y_true, np.zeros(len(y_true)))


results = {"development_cv": {}, "test": {}}
for name, make in MODELS.items():
    oof = out_of_fold_probabilities(make, data)
    results["development_cv"][name] = row(train.loc[oof.index, "is_abnormal"], oof)
    fitted = make().fit(train[MODEL_FEATURES], train.is_abnormal)
    test_proba = fitted.predict_proba(test[MODEL_FEATURES])[:, 1]
    results["test"][name] = row(test.is_abnormal, test_proba)
results["development_cv"]["Always Normal (baseline)"] = baseline_row(train.is_abnormal)
results["test"]["Always Normal (baseline)"] = baseline_row(test.is_abnormal)

# Consistency check: the frozen logistic regression must reproduce the committed test numbers exactly.
saved = json.loads((METRICS_DIR / "test_results.json").read_text())["pooled"]
lr_test = results["test"]["Logistic regression"]
for key in ["tn", "fp", "fn", "tp"]:
    assert lr_test[key] == saved[key], f"{key}: {lr_test[key]} != saved {saved[key]}"

COLUMNS = ["accuracy", "precision", "recall", "f1", "macro_f1", "false_alarm_rate", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
frames = {part: pd.DataFrame(rows).T[COLUMNS] for part, rows in results.items()}
for part, frame in frames.items():
    frame.to_csv(METRICS_DIR / f"model_comparison_{part}.csv", float_format="%.4f")
    print(f"\n=== {part} ===")
    print(frame.round(3).to_string())

(METRICS_DIR / "model_comparison.json").write_text(json.dumps(
    {part: {m: {k: (None if pd.isna(v) else float(v)) for k, v in vals.items()} for m, vals in rows.items()}
     for part, rows in results.items()}, indent=2))

# Confusion matrices: rows = true class, columns = predicted class, counts.
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for r, (part, title) in enumerate([("development_cv", "Development CV (12 patients)"), ("test", "Test (5 patients)")]):
    for c, name in enumerate(MODELS):
        v = results[part][name]
        cm = np.array([[v["tn"], v["fp"]], [v["fn"], v["tp"]]])
        ax = axes[r, c]
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%} of row)", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred Normal", "pred Abnormal"])
        ax.set_yticks([0, 1], ["true Normal", "true Abnormal"])
        ax.set_title(f"{name}\n{title}", fontsize=10)
fig.suptitle("Confusion matrices at threshold 0.5 (shade = share of the true class)")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=130)
print("\nsaved", FIGURE_PATH)
