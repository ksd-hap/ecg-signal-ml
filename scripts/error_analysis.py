"""Error analysis of the three models. Needs data/processed/error_frame.pkl (scripts/error_analysis_data.py).

Every question below is answered with a measurement; the report at the end of the run lists what each number can
and cannot support. Primary model = the frozen logistic regression (threshold 0.5); the two tree models are
shown alongside. All predictions are from models that never saw the beat's patient.

Run from the repo root:  python scripts/error_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import out_of_fold_probabilities, summarize
from src.feature_extraction import MODEL_FEATURES
from src.preprocessing import load_filtered_record
from src.models import make_gradient_boosting, make_logistic_regression, make_random_forest

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
data = pd.read_pickle("data/processed/error_frame.pkl")
annotated = pd.read_pickle("data/processed/error_annotated.pkl")
NAMES = {"lr": "logistic regression", "rf": "random forest", "gb": "gradient boosting"}
data["part"] = np.where(data.split == "test", "test", "dev")
for k in NAMES:
    data[f"flag_{k}"] = (data[f"p_{k}"] >= 0.5).astype(int)
data["err_lr"] = np.where(data.is_abnormal == 1, data.flag_lr == 0, data.flag_lr == 1)


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ================================================================================================================
section("1. WHICH CLASSES ARE MISCLASSIFIED  (share of beats of each annotation symbol flagged ABNORMAL, %)")
for part in ["dev", "test"]:
    sub = data[data.part == part]
    rows = []
    for sym, g in sub.groupby("symbol"):
        rows.append({"symbol": sym, "label": g.label.iloc[0], "beats": len(g), "patients": g.patient.nunique(),
                     **{NAMES[k]: round(100 * g[f"flag_{k}"].mean(), 1) for k in NAMES}})
    print(f"\n{part} ({sub.patient.nunique()} patients)   [abnormal symbols: this is recall; normal symbols: false-alarm rate]")
    print(pd.DataFrame(rows).sort_values(["label", "beats"], ascending=[False, False]).to_string(index=False))

print("\nWhere the frozen logistic regression's ERRORS come from (share of all errors):")
err = data[data.err_lr]
by_sym = err.groupby(["part", "symbol"]).size().rename("errors").reset_index()
by_sym["share_of_part_errors_%"] = (100 * by_sym.errors / by_sym.groupby("part").errors.transform("sum")).round(1)
print(by_sym.sort_values(["part", "errors"], ascending=[True, False]).to_string(index=False))
print("\nErrors by patient (LR):")
by_pat = data.groupby(["part", "patient"]).agg(beats=("err_lr", "size"), errors=("err_lr", "sum"),
                                               abnormal=("is_abnormal", "sum")).reset_index()
by_pat["share_of_part_errors_%"] = (100 * by_pat.errors / by_pat.groupby("part").errors.transform("sum")).round(1)
print(by_pat.sort_values(["part", "errors"], ascending=[True, False]).to_string(index=False))

# ================================================================================================================
section("2. DOES NOISE CONTRIBUTE?  (noise measured on the RAW signal in each beat's own window)")
print("hf_noise_rel = std of >40 Hz content outside +-60 ms of R, divided by the beat's QRS peak-to-peak.")
print("baseline_range_mv = peak-to-peak of the <0.5 Hz component in the window. Ranked WITHIN each patient, so that")
print("'noisy' means 'noisier than that patient's usual', not 'a patient with a noisy recording'.\n")
for col in ["hf_noise_rel", "baseline_range_mv"]:
    data[f"{col}_pct"] = data.groupby("record")[col].rank(pct=True)
    data[f"{col}_q"] = pd.cut(data[f"{col}_pct"], [0, .25, .5, .75, .9, 1.0], labels=["Q1", "Q2", "Q3", "Q4-90", "top10%"])

for col in ["hf_noise_rel", "baseline_range_mv"]:
    print(f"\n-- {col}: error rate of the frozen LR by within-patient noise quartile --")
    for cls, name in [(0, "Normal beats (error = false alarm)"), (1, "Abnormal beats (error = missed)")]:
        sub = data[data.is_abnormal == cls]
        tab = sub.groupby(f"{col}_q", observed=True).err_lr.agg(["size", "mean"]).round(3)
        print(f"  {name}: " + "  ".join(f"{q}: {r['mean']:.3f} (n={int(r['size'])})" for q, r in tab.iterrows()))
    # sign consistency: does the noisiest 25% have a higher error rate than the quietest 25%, patient by patient?
    ups = downs = 0
    for _, g in data.groupby("record"):
        for cls in (0, 1):
            h = g[g.is_abnormal == cls]
            hi, lo = h[h[f"{col}_pct"] > .75], h[h[f"{col}_pct"] <= .25]
            if min(len(hi), len(lo)) >= 30:
                ups += hi.err_lr.mean() > lo.err_lr.mean()
                downs += hi.err_lr.mean() < lo.err_lr.mean()
    print(f"  patient x class cells (>=30 beats in both quartiles): noisiest quartile has MORE errors in {ups}, FEWER in {downs}")
    e = data[data.err_lr]
    print(f"  median within-patient noise percentile of ERRORS: {e[f'{col}_pct'].median():.2f} (0.50 = no relation to noise)")

rec = data.groupby("record").agg(hf=("hf_noise_rel", "median"), base=("baseline_range_mv", "median"),
                                 err=("err_lr", "mean"), abnormal_share=("is_abnormal", "mean"))
print("\nBetween patients (17 records): Spearman correlation of median noise with LR error rate:")
for col in ["hf", "base"]:
    r, p = spearmanr(rec[col], rec.err)
    print(f"  {col}: rho={r:+.2f} (p={p:.2f}, n=17)")
print("\nPer-record noise and error rate:")
print(rec.round(3).sort_values("hf", ascending=False).to_string())

# ================================================================================================================
section("3. DOES CLASS IMBALANCE CONTRIBUTE?")
dev = data[data.part == "dev"]
print(f"Abnormal share: dev {dev.is_abnormal.mean():.3f}, test {data[data.part == 'test'].is_abnormal.mean():.3f}")
print("\nRe-running development CV with class_weight='balanced' (same folds, same features):")
rows = {}
oof_bal = {}
for key, make in [("lr", make_logistic_regression), ("rf", make_random_forest)]:
    for weight in [None, "balanced"]:
        proba = dev[f"p_{key}"] if weight is None else out_of_fold_probabilities(lambda: make(class_weight=weight), data)
        if weight:
            oof_bal[key] = proba
        s = summarize(dev.is_abnormal, proba)
        rows[f"{NAMES[key]} / {weight or 'unweighted'}"] = {k: s[k] for k in
                                                          ["precision", "recall", "f1", "false_alarm_rate", "roc_auc", "pr_auc"]}
print(pd.DataFrame(rows).T.round(3).to_string())


def recall_at_false_alarm(y, proba, fa=0.05):
    fpr, tpr, _ = roc_curve(y, proba)
    return float(np.interp(fa, fpr, tpr))


print("\nRecall at a FIXED false-alarm rate (threshold-free test of whether weighting adds separability):")
for key in oof_bal:
    for fa in (0.02, 0.05, 0.10):
        print(f"  {NAMES[key]:>20}  FA={fa:.0%}: unweighted recall {recall_at_false_alarm(dev.is_abnormal, dev[f'p_{key}'], fa):.3f}"
              f"   balanced recall {recall_at_false_alarm(dev.is_abnormal, oof_bal[key], fa):.3f}")

print("\nRecall of each abnormal symbol vs how many examples the model had (dev patients only, frozen LR / balanced LR / balanced RF):")
rows = []
for sym in ["V", "F", "A", "J", "a", "j"]:
    g = dev[dev.symbol == sym]
    if len(g):
        rows.append({"symbol": sym, "beats": len(g), "patients": g.patient.nunique(),
                     "top_patient_share_%": round(100 * g.patient.value_counts(normalize=True).iloc[0]),
                     "LR_recall_%": round(100 * g.flag_lr.mean(), 1),
                     "LR_balanced_%": round(100 * (oof_bal["lr"].loc[g.index] >= .5).mean(), 1),
                     "RF_balanced_%": round(100 * (oof_bal["rf"].loc[g.index] >= .5).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

# ================================================================================================================
section("4. ARE THE FEATURES INSUFFICIENT?")
print("Test A - is the information in the features at all? Train and test WITHIN one patient (first half of the record")
print("in time -> second half, and back), gradient boosting on the same 13 features. Compare its ROC-AUC with the")
print("cross-patient (out-of-fold / test) ROC-AUC of the same model family for that patient.")
rows = []
for record_name, g in data.groupby("record"):
    g = g.sort_values("r_peak_sample")
    half = len(g) // 2
    a, b = g.iloc[:half], g.iloc[half:]
    ok = min(a.is_abnormal.sum(), b.is_abnormal.sum()) >= 30 and min((1 - a.is_abnormal).sum(), (1 - b.is_abnormal).sum()) >= 30
    if not ok:
        continue
    within = pd.concat([pd.Series(make_gradient_boosting().fit(a[MODEL_FEATURES], a.is_abnormal).predict_proba(b[MODEL_FEATURES])[:, 1], index=b.index),
                        pd.Series(make_gradient_boosting().fit(b[MODEL_FEATURES], b.is_abnormal).predict_proba(a[MODEL_FEATURES])[:, 1], index=a.index)])
    rows.append({"record": record_name, "abnormal": int(g.is_abnormal.sum()),
                 "within_patient_ROC": roc_auc_score(g.is_abnormal, within.loc[g.index]),
                 "cross_patient_GB_ROC": roc_auc_score(g.is_abnormal, g.p_gb),
                 "within_patient_PR": average_precision_score(g.is_abnormal, within.loc[g.index]),
                 "cross_patient_GB_PR": average_precision_score(g.is_abnormal, g.p_gb)})
within_table = pd.DataFrame(rows).set_index("record")
print(within_table.round(3).to_string())
print("mean:", within_table.mean().round(3).to_dict())
within_table.to_csv("results/metrics/within_vs_cross_patient.csv", float_format="%.4f")

print("\nTest B - is morphology missing? Add the (amplitude-normalised) beat waveform, 108 samples, to the 13 features;")
print("gradient boosting, same grouped CV on the 12 development patients.")
dev_idx = data.index[data.part == "dev"]
waves = np.zeros((len(dev_idx), 108))
for n, (i, row) in enumerate(data.loc[dev_idx].iterrows()):
    filtered, fs = load_filtered_record(row.record)
    w = filtered[int(row.r_peak_sample) - 72:int(row.r_peak_sample) + 144][::2]
    waves[n] = (w - np.median(w)) / max(row.qrs_p2p_mv, 1e-3)
wave_cols = [f"w{j}" for j in range(108)]
wave_frame = pd.DataFrame(waves, index=dev_idx, columns=wave_cols)
aug = data.loc[dev_idx].join(wave_frame)
oof_wave = out_of_fold_probabilities(make_gradient_boosting, aug, features=MODEL_FEATURES + wave_cols)
base_s = summarize(dev.is_abnormal, dev.p_gb)
wave_s = summarize(dev.is_abnormal, oof_wave)
print(pd.DataFrame({"13 features": base_s, "13 + waveform": wave_s}).loc[
    ["precision", "recall", "f1", "false_alarm_rate", "roc_auc", "pr_auc"]].round(3).to_string())
rows = []
for sym in ["V", "F", "A", "J", "N", "L", "R"]:
    g = dev[dev.symbol == sym]
    if len(g):
        rows.append({"symbol": sym, "beats": len(g), "flagged_13_%": round(100 * g.flag_gb.mean(), 1),
                     "flagged_+waveform_%": round(100 * (oof_wave.loc[g.index] >= .5).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
data.loc[dev_idx, "p_gb_wave"] = oof_wave

# ================================================================================================================
section("5. COULD R-PEAK DETECTION / SEGMENTATION BE RESPONSIBLE?")
ann = annotated[annotated.symbol != "false_positive"]
print("Detector outcome on the 17 evaluated records (annotated beats missed = never reach the classifier):")
tab = ann.groupby("symbol").detected.agg(beats="size", missed=lambda s: int((~s).sum()))
tab["missed_%"] = (100 * tab.missed / tab.beats).round(2)
print(tab.sort_values("beats", ascending=False).to_string())
print(f"\nDetector false-positive peaks (no beat annotation within 50 ms): {int((annotated.symbol == 'false_positive').sum())}")
per_rec = ann.groupby("record").detected.agg(beats="size", missed=lambda s: int((~s).sum()))
per_rec["fp"] = annotated[annotated.symbol == "false_positive"].groupby("record").size().reindex(per_rec.index).fillna(0).astype(int)
print(per_rec.assign(**{"missed_%": (100 * per_rec.missed / per_rec.beats).round(2)}).to_string())

is_abn_sym = ann.symbol.isin(["V", "A", "a", "J", "j", "F", "L", "R"][:0] + ["V", "A", "a", "J", "j", "F"])
print(f"\nMissed-beat rate: abnormal symbols {100 * (~ann[is_abn_sym].detected).mean():.2f}%  vs  normal-type symbols "
      f"{100 * (~ann[~is_abn_sym].detected).mean():.2f}%")

print("\nR-peak localisation of the beats that DO reach the classifier (detected sample minus annotated sample, ms):")
data["abs_offset_ms"] = data.ann_offset_ms.abs()
print(data.groupby("symbol").abs_offset_ms.agg(["median", lambda s: s.quantile(.95), "max"]).round(1)
      .rename(columns={"<lambda_0>": "p95"}).to_string())
far = data.abs_offset_ms > 50
print(f"\nBeats matched to a label but located >50 ms from the annotation: {int(far.sum())} of {len(data)}; "
      f"LR error rate {data[far].err_lr.mean():.3f} vs {data[~far].err_lr.mean():.3f} otherwise")
print("Error rate by offset band (ms):")
band = pd.cut(data.abs_offset_ms, [-1, 10, 20, 50, 150])
print(data.groupby(band, observed=True).err_lr.agg(["size", "mean"]).round(3).to_string())

print("\nSegmentation: the window is 200 ms before / 400 ms after R. A neighbouring R peak inside the window means")
print("the 'beat' also contains part of another beat (short RR).")
contaminated = (data.rr_pre_s < 0.2 + 0.05) | (data.rr_post_s < 0.4)
print(f"Beats whose window contains another beat's R peak (rr_post < 0.4 s): {int(contaminated.sum())} of {len(data)}")
for cls, name in [(0, "Normal"), (1, "Abnormal")]:
    g = data[data.is_abnormal == cls]
    c = contaminated.loc[g.index]
    print(f"  {name}: error rate with contaminated window {g[c].err_lr.mean():.3f} (n={int(c.sum())}) vs clean {g[~c].err_lr.mean():.3f}")
print("\nWindows cut at the record boundary or without RR neighbours are dropped by the pipeline (see notebook 05/06).")

data.to_pickle("data/processed/error_frame_analysed.pkl")
