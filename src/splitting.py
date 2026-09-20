"""Patient-level train/test split and cross-validation folds (the leakage guard)."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPLIT_PATH = Path(__file__).resolve().parents[1] / "results" / "metrics" / "record_split.json"

# Two recordings of the same patient: record 202's header says "taken from the same analog tape as record 201",
# and both list the same age, sex and medication. Splitting must group by patient, not just by recording.
PATIENT_OF_RECORD = {"201": "202"}


def patient_id(record):
    record = str(record)
    return PATIENT_OF_RECORD.get(record, record)


def _with_patient(table):
    return table.assign(patient=table["record"].astype(str).map(patient_id), symbol=table["symbol"].astype(str))


def patient_stats(table):
    """Beats, abnormal beats, and the two main abnormal types (V, A) per patient."""
    t = _with_patient(table)
    return pd.DataFrame({
        "beats": t.groupby("patient").size(),
        "abnormal": t.groupby("patient")["is_abnormal"].sum(),
        "V": t[t.symbol == "V"].groupby("patient").size(),
        "A": t[t.symbol == "A"].groupby("patient").size(),
    }).fillna(0).astype(int)


def single_source_patients(table, min_beats=100, min_share=0.8):
    """Patients supplying >= min_share of some beat type that has >= min_beats beats. Returns {patient: [symbols]}.

    A morphology that lives in one patient can be learned or tested, not both.
    """
    t = _with_patient(table)
    counts = t.groupby(["symbol", "patient"]).size().rename("n").reset_index()
    counts["total"] = counts.groupby("symbol")["n"].transform("sum")
    hits = counts[(counts.total >= min_beats) & (counts.n / counts.total >= min_share)]
    return {p: sorted(g.symbol) for p, g in hits.groupby("patient")}


def choose_test_patients(table, n_test=5, target_share=0.30, max_single_share=0.40):
    """Exhaustively pick the test patients whose share of beats / abnormal / V / A beats is closest to target_share.

    Deterministic (no randomness) and decided from metadata only - never from model results. Patients that are the
    sole source of a morphology stay in train; no patient may supply more than max_single_share of the abnormal
    beats on either side.
    """
    stats = patient_stats(table)
    totals = stats.sum()
    forced_train = single_source_patients(table)
    candidates = sorted(p for p in stats.index if p not in forced_train)

    best = None
    for test in combinations(candidates, n_test):
        in_test, in_train = stats.loc[list(test)], stats.drop(list(test))
        if in_test.abnormal.sum() == 0:
            continue
        if (in_test.abnormal.max() / in_test.abnormal.sum() > max_single_share
                or in_train.abnormal.max() / in_train.abnormal.sum() > max_single_share):
            continue
        shares = in_test.sum() / totals
        score = float(((shares - target_share) ** 2).sum())
        if best is None or score < best[0] - 1e-12:
            best = (score, test)
    if best is None:
        raise ValueError("no test set satisfies the constraints")
    return sorted(best[1]), best[0], forced_train


def make_cv_folds(table, dev_patients, n_folds=4):
    """Partition the development patients into equal-sized folds, balanced on beats / abnormal / V / A shares.

    Patients that are the sole source of a morphology inside the development set are spread over different folds,
    so no single fold is the one that has to extrapolate. Exhaustive and deterministic.
    """
    dev = sorted(dev_patients)
    size, remainder = divmod(len(dev), n_folds)
    if remainder:
        raise ValueError(f"{len(dev)} development patients cannot form {n_folds} equal folds")
    stats = patient_stats(table).loc[dev]
    totals = stats.sum()
    anchors = set(single_source_patients(_with_patient(table).query("patient in @dev")))

    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for mates in combinations(rest, size - 1):
            remaining = [x for x in rest if x not in mates]
            for tail in partitions(remaining):
                yield [(first,) + mates] + tail

    best = None
    for parts in partitions(dev):
        if any(len(set(group) & anchors) > 1 for group in parts):
            continue
        score = sum(float(((stats.loc[list(g)].sum() / totals - 1 / n_folds) ** 2).sum()) for g in parts)
        if best is None or score < best[0] - 1e-12:
            best = (score, parts)
    return sorted((sorted(g) for g in best[1]), key=lambda g: g[0]), sorted(anchors)


def build_split(table, n_test=5, n_folds=4):
    """Full split specification as a plain, JSON-serialisable dict."""
    test, score, forced_train = choose_test_patients(table, n_test=n_test)
    dev = sorted(set(patient_stats(table).index) - set(test))
    folds, cv_anchors = make_cv_folds(table, dev, n_folds=n_folds)
    return {
        "unit": "patient (recordings 201 and 202 share a patient)",
        "test_patients": test,
        "dev_patients": dev,
        "cv_folds": folds,
        "forced_to_train": forced_train,
        "cv_spread_across_folds": cv_anchors,
        "rules": {
            "n_test": n_test,
            "n_folds": n_folds,
            "target_test_share": 0.30,
            "max_single_patient_share_of_abnormal": 0.40,
            "single_source_min_beats": 100,
            "single_source_min_share": 0.8,
        },
    }


def apply_split(table, split):
    """Add `patient`, `split` ('train'/'test') and `cv_fold` (0..k-1 for train beats, -1 for test) columns."""
    t = table.copy()
    t["patient"] = t["record"].astype(str).map(patient_id)
    t["split"] = np.where(t["patient"].isin(split["test_patients"]), "test", "train")
    fold_of = {p: i for i, fold in enumerate(split["cv_folds"]) for p in fold}
    t["cv_fold"] = t["patient"].map(fold_of).fillna(-1).astype(int)
    check_split(t)
    return t


def check_split(t):
    """Fail loudly if any patient, recording or beat could appear on both sides of a split."""
    train, test = t[t.split == "train"], t[t.split == "test"]
    assert t.split.isin(["train", "test"]).all(), "a beat has no split"
    assert not set(train.patient) & set(test.patient), "a patient appears in both train and test"
    assert not set(train.record.astype(str)) & set(test.record.astype(str)), "a recording appears in both train and test"
    assert (test.cv_fold == -1).all() and (train.cv_fold >= 0).all(), "test beats must not belong to a CV fold"
    assert (t.groupby("patient").split.nunique() == 1).all(), "a patient is split across train and test"
    assert (train.groupby("patient").cv_fold.nunique() == 1).all(), "a patient is split across CV folds"
    assert not t.duplicated(["record", "r_peak_sample"]).any(), "a beat appears twice"


def save_split(split, path=DEFAULT_SPLIT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def load_split(path=DEFAULT_SPLIT_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))
