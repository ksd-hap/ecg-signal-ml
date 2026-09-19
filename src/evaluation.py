import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from .feature_extraction import MODEL_FEATURES
from .preprocessing import load_record


def out_of_fold_probabilities(make_model, data, features=MODEL_FEATURES):
    """Grouped cross-validation on the development set only.

    Every beat is scored by a model trained on the other folds, which never contain that beat's patient.
    The model (including any scaler) is rebuilt and refit inside every fold. Test beats are never touched.
    """
    dev = data[data.split == "train"]
    proba = pd.Series(np.nan, index=dev.index, name="proba")
    for fold in sorted(dev.cv_fold.unique()):
        fit_part, held_out = dev[dev.cv_fold != fold], dev[dev.cv_fold == fold]
        model = make_model().fit(fit_part[features], fit_part["is_abnormal"])
        proba.loc[held_out.index] = model.predict_proba(held_out[features])[:, 1]
    assert not proba.isna().any()
    return proba


def summarize(y_true, proba, threshold=0.5):
    """Threshold-based and threshold-free metrics, with 'abnormal' as the positive class."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else np.nan
    both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else np.nan,
        "pr_auc": average_precision_score(y_true, proba) if both_classes else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_by_group(data, proba, group_column, threshold=0.5):
    """One row of metrics per value of `group_column` (e.g. cv_fold or record)."""
    rows = {}
    for key, part in data.loc[proba.index].groupby(group_column):
        rows[key] = {"beats": len(part), "abnormal": int(part.is_abnormal.sum()),
                     **summarize(part.is_abnormal, proba.loc[part.index], threshold)}
    return pd.DataFrame(rows).T


def flagged_rate_by_symbol(data, proba, threshold=0.5):
    """Share of beats of each annotation symbol that the model calls abnormal.

    For abnormal symbols this is the recall; for normal symbols it is the false-alarm rate.
    """
    part = data.loc[proba.index].assign(flagged=(proba >= threshold).astype(int))
    table = part.groupby(part.symbol.astype(str)).agg(label=("label", "first"), beats=("flagged", "size"),
                                                      flagged=("flagged", "sum"))
    table["pct_flagged_abnormal"] = (100 * table.flagged / table.beats).round(1)
    return table.sort_values(["label", "beats"], ascending=[False, False])


def patient_bootstrap(data, proba_a, proba_b, n_boot=500, seed=0):
    """Resample whole patients with replacement and score two models on each resample.

    Beats within a patient are not independent, so patients (not beats) are the unit that is resampled.
    Returns one row per replicate with PR-AUC and F1 (threshold 0.5) for models a and b.
    """
    part = data.loc[proba_a.index]
    patients = part["patient"].to_numpy()
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    y, a, b = part["is_abnormal"].to_numpy(), proba_a.to_numpy(), proba_b.to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() == 0:
            continue
        rows.append({
            "pr_auc_a": average_precision_score(y[idx], a[idx]), "pr_auc_b": average_precision_score(y[idx], b[idx]),
            "roc_auc_a": roc_auc_score(y[idx], a[idx]), "roc_auc_b": roc_auc_score(y[idx], b[idx]),
            "f1_a": f1_score(y[idx], a[idx] >= 0.5, zero_division=0), "f1_b": f1_score(y[idx], b[idx] >= 0.5, zero_division=0),
        })
    return pd.DataFrame(rows)


def permutation_importance_by_fold(make_model, data, features=MODEL_FEATURES, n_repeats=3, seed=0):
    """How much does each feature matter for patients the model has NOT seen?

    In every cross-validation fold, fit on the other folds, then shuffle one feature at a time in the held-out fold
    and record the drop in PR-AUC. Development data only; returns features x folds (mean drop over repeats).
    """
    dev = data[data.split == "train"]
    rng = np.random.default_rng(seed)
    result = {}
    for fold in sorted(dev.cv_fold.unique()):
        fit_part, held_out = dev[dev.cv_fold != fold], dev[dev.cv_fold == fold]
        model = make_model().fit(fit_part[features], fit_part["is_abnormal"])
        y = held_out["is_abnormal"].to_numpy()
        base = average_precision_score(y, model.predict_proba(held_out[features])[:, 1])
        drops = {}
        for feature in features:
            scores = []
            for _ in range(n_repeats):
                shuffled = held_out[features].copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                scores.append(average_precision_score(y, model.predict_proba(shuffled)[:, 1]))
            drops[feature] = base - float(np.mean(scores))
        result[fold] = drops
    return pd.DataFrame(result)


def annotated_rhythm(data):
    """Rhythm annotation in force at each beat (e.g. '(N' sinus, '(AFIB'). For analysis only - never a model input."""
    rhythm = pd.Series("(unlabeled", index=data.index)
    for record, part in data.groupby("record"):
        _, annotation = load_record(str(record))
        symbols, samples, notes = np.asarray(annotation.symbol), np.asarray(annotation.sample), np.asarray(annotation.aux_note)
        is_change = symbols == "+"
        starts, names = samples[is_change], [n.strip("\x00").strip() for n in notes[is_change]]
        position = np.searchsorted(starts, part["r_peak_sample"].to_numpy(), side="right") - 1
        rhythm.loc[part.index] = [names[i] if i >= 0 else "(unlabeled" for i in position]
    return rhythm

def patient_bootstrap_metrics(data, proba, n_boot=2000, seed=0, threshold=0.5):
    """Metrics of one model on patient-level bootstrap resamples (whole patients drawn with replacement)."""
    part = data.loc[proba.index]
    patients = part["patient"].to_numpy()
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    y, scores = part["is_abnormal"].to_numpy(), proba.to_numpy()
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "precision", "recall", "f1", "false_alarm_rate", "roc_auc", "pr_auc"]
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        result = summarize(y[idx], scores[idx], threshold)
        rows.append({k: result[k] for k in keys})
    return pd.DataFrame(rows)
