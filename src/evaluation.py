import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .feature_extraction import MODEL_FEATURES


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
