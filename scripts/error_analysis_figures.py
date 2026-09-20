"""Figures for the error analysis. Needs data/processed/error_frame_analysed.pkl (scripts/error_analysis.py).

Examples are NOT hand-picked for how convincing they look: within each error category the patients with the most
errors are used and one beat per patient is drawn at random (fixed seed).

Run from the repo root:  python scripts/error_analysis_figures.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extraction import _filtered_record
from src.preprocessing import get_ecg_lead, load_record

FIG_DIR = Path("results/figures")
d = pd.read_pickle("data/processed/error_frame_analysed.pkl")
annotated = pd.read_pickle("data/processed/error_annotated.pkl")
rng = np.random.default_rng(0)
COLORS = {"lr": "#1f77b4", "rf": "#2ca02c", "gb": "#d62728"}
NAMES = {"lr": "logistic regression", "rf": "random forest", "gb": "gradient boosting"}


def raw_signal(record_name):
    record, _ = load_record(record_name)
    return get_ecg_lead(record)[0]


# ---- 28: which classes / which patients -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.1, 0.7, 1.2]})
for ax, part in zip(axes[:2], ["dev", "test"]):
    sub = d[d.part == part]
    symbols = [s for s in ["V", "F", "A", "J", "N", "R", "L"] if s in set(sub.symbol) and (sub.symbol == s).sum() >= 50]
    x = np.arange(len(symbols))
    for i, key in enumerate(NAMES):
        ax.bar(x + (i - 1) * 0.27, [100 * sub[sub.symbol == s][f"flag_{key}"].mean() for s in symbols], 0.27,
               color=COLORS[key], label=NAMES[key])
    ax.set_xticks(x, [f"{s}\n(n={int((sub.symbol == s).sum()):,})" for s in symbols])
    ax.set_ylabel("% of beats flagged abnormal")
    ax.set_title(f"{'Development CV, 12 patients' if part == 'dev' else 'Test, 5 patients'}\n"
                 "V, F, A, J: higher is better (recall)   N, R, L: lower is better")
    ax.axvline(len([s for s in symbols if s in "VFAJ"]) - 0.5, color="gray", ls=":")
axes[0].legend(fontsize=8)
counts = d.groupby(["part", "patient"]).err_lr.sum().reset_index().sort_values("err_lr", ascending=False)
counts["share"] = 100 * counts.err_lr / counts.groupby("part").err_lr.transform("sum")
counts = counts[counts.share >= 1.0].sort_values(["part", "share"], ascending=[True, False])
ax = axes[2]
ax.bar([f"{r.patient}\n{r.part}" for r in counts.itertuples()], counts.share,
       color=["#9ecae1" if r.part == "dev" else "#fdae6b" for r in counts.itertuples()])
ax.set_ylabel("% of that part's errors (frozen LR)")
ax.set_title("Errors are concentrated in a few patients\n(blue = development, orange = test; patients under 1% omitted)")
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "28_error_breakdown.png", dpi=130)
plt.close(fig)


# ---- 29: representative misclassified beats ---------------------------------------------------------------------
def pick(mask, n=3):
    """The patients with most errors in this category, one random beat each."""
    pool = d[mask]
    patients = pool.patient.value_counts().index[:n]
    return [pool[pool.patient == p].sample(1, random_state=int(rng.integers(1e6))).iloc[0] for p in patients]


categories = [
    ("MISSED atrial beats (A, p<0.5)", (d.symbol == "A") & (d.flag_lr == 0)),
    ("MISSED fusion beats (F)", (d.symbol == "F") & (d.flag_lr == 0)),
    ("MISSED ventricular beats (V)", (d.symbol == "V") & (d.flag_lr == 0)),
    ("FALSE ALARMS on N beats", (d.symbol == "N") & (d.flag_lr == 1)),
    ("FALSE ALARMS on R beats (right bundle branch block, labelled Normal)", (d.symbol == "R") & (d.flag_lr == 1)),
]
fig, axes = plt.subplots(len(categories), 3, figsize=(17, 3.1 * len(categories)))
for row, (title, mask) in enumerate(categories):
    beats = pick(mask)
    for col in range(3):
        ax = axes[row, col]
        if col >= len(beats):
            ax.axis("off")
            continue
        b = beats[col]
        filtered, fs = _filtered_record(b.record)
        raw = raw_signal(b.record)
        lo, hi = int(b.r_peak_sample - 1.2 * fs), int(b.r_peak_sample + 1.2 * fs)
        t = (np.arange(lo, hi) - b.r_peak_sample) / fs
        ax.plot(t, raw[lo:hi], color="lightgray", lw=0.8, label="raw")
        ax.plot(t, filtered[lo:hi], color="black", lw=0.9, label="filtered (model input)")
        ax.axvspan(-0.2, 0.4, color="#fdd49e", alpha=0.35)
        near = d[(d.record == b.record) & d.r_peak_sample.between(lo, hi)]
        for nb in near.itertuples():
            tt = (nb.r_peak_sample - b.r_peak_sample) / fs
            ax.text(tt, filtered[nb.r_peak_sample] + 0.15, nb.symbol, ha="center", fontsize=8,
                    color="red" if nb.Index == b.name else "dimgray", fontweight="bold" if nb.Index == b.name else None)
        ax.set_title(f"rec {b.record} '{b.symbol}' (true {b.label}) | P(abn): LR {b.p_lr:.2f} RF {b.p_rf:.2f} GB {b.p_gb:.2f}\n"
                     f"RR before {b.rr_pre_s:.2f}s after {b.rr_post_s:.2f}s | {b.part}", fontsize=8)
        ax.set_xlabel("s from this R peak (orange = the 600 ms window the model saw)", fontsize=7)
        if col == 0:
            ax.set_ylabel(title, fontsize=8, fontweight="bold")
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Misclassified beats by the frozen logistic regression: one random beat from each of the patients with most such errors "
             "(letters = annotation of each detected beat)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "29_error_examples.png", dpi=110)
plt.close(fig)

# ---- 30: noise --------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
q_labels = ["Q1", "Q2", "Q3", "Q4-90", "top10%"]
d["noise_q"] = pd.cut(d.groupby("record").hf_noise_rel.rank(pct=True), [0, .25, .5, .75, .9, 1.0], labels=q_labels)
ax = axes[0, 0]
for sym, color in [("N", "#1f77b4"), ("A", "#ff7f0e"), ("V", "#d62728")]:
    rate = d[d.symbol == sym].groupby("noise_q", observed=True).err_lr.mean()
    ax.plot(q_labels, 100 * rate.values, "o-", color=color, label=f"{sym} (n={int((d.symbol == sym).sum()):,})")
ax.set_ylabel("% misclassified (frozen LR)")
ax.set_xlabel("within-patient high-frequency noise quartile")
ax.set_title("Error rate vs noise, per beat type\n(ranked inside each patient)")
ax.legend()
rec = d.groupby("record").agg(noise=("hf_noise_rel", "median"), err=("err_lr", "mean"), part=("part", "first"))
ax = axes[0, 1]
ax.scatter(100 * rec.noise, 100 * rec.err, c=np.where(rec.part == "test", "#fdae6b", "#9ecae1"), edgecolor="k")
for name, r in rec.iterrows():
    ax.annotate(name, (100 * r.noise, 100 * r.err), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("patient's median HF noise (% of QRS peak-to-peak)")
ax.set_ylabel("% of patient's beats misclassified")
ax.set_title("Between patients (blue dev, orange test)\nRho +0.39, p=0.12, n=17")
axes[0, 2].hist(np.log10(d.hf_noise_rel.clip(lower=1e-4)), bins=60, color="gray")
axes[0, 2].set_xlabel("log10 HF noise / QRS p2p")
axes[0, 2].set_title("Distribution of the noise measure (all beats):\ntwo humps = between-patient differences; thin right tail")
noisy = d.sort_values("hf_noise_rel", ascending=False).drop_duplicates("record").head(3)
for col, b in enumerate(noisy.itertuples()):
    ax = axes[1, col]
    filtered, fs = _filtered_record(b.record)
    raw = raw_signal(b.record)
    lo, hi = int(b.r_peak_sample - 0.6 * fs), int(b.r_peak_sample + 0.8 * fs)
    t = (np.arange(lo, hi) - b.r_peak_sample) / fs
    ax.plot(t, raw[lo:hi], color="lightgray", lw=0.9, label="raw")
    ax.plot(t, filtered[lo:hi], color="black", lw=0.9, label="filtered")
    ax.axvspan(-0.2, 0.4, color="#fdd49e", alpha=0.35)
    ax.set_title(f"noisiest beat of rec {b.record} ('{b.symbol}', true {b.label}), P(abn) LR {b.p_lr:.2f}", fontsize=9)
    ax.set_xlabel("s from R peak")
axes[1, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "30_noise.png", dpi=120)
plt.close(fig)

# ---- 31: within-patient vs cross-patient separability ---------------------------------------------------------
w = pd.read_csv("results/metrics/within_vs_cross_patient.csv", dtype={"record": str}).rename(columns={
    "within_patient_ROC": "roc_within", "cross_patient_GB_ROC": "roc_cross",
    "within_patient_PR": "pr_within", "cross_patient_GB_PR": "pr_cross"})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
x = np.arange(len(w))
for ax, (a, b, name) in zip(axes, [("roc_within", "roc_cross", "ROC-AUC"), ("pr_within", "pr_cross", "PR-AUC")]):
    ax.bar(x - 0.2, w[a], 0.4, label="trained on the SAME patient (other half of the record)", color="#54a24b")
    ax.bar(x + 0.2, w[b], 0.4, label="trained on OTHER patients (gradient boosting)", color="#e45756")
    ax.set_xticks(x, w.record)
    ax.set_ylabel(name)
    ax.set_xlabel("record")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Same 13 features, same model family: separable inside a patient, less so across patients (mean ROC "
             f"{w.roc_within.mean():.3f} vs {w.roc_cross.mean():.3f}; record 232 is inverted across patients)")
fig.tight_layout()
fig.savefig(FIG_DIR / "31_within_vs_cross_patient.png", dpi=120)
plt.close(fig)

# ---- 32: detector misses -----------------------------------------------------------------------------------------
ann = annotated[annotated.symbol != "false_positive"]
fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))
tab = ann.groupby("symbol").detected.agg(beats="size", missed=lambda s: int((~s).sum()))
tab = tab[tab.beats >= 20].sort_values("beats", ascending=False)
ax = axes[0, 0]
ax.bar(tab.index, 100 * tab.missed / tab.beats, color="#e45756")
for i, (s, r) in enumerate(tab.iterrows()):
    ax.text(i, 100 * r.missed / r.beats + 0.3, f"{int(r.missed)}/{int(r.beats)}", ha="center", fontsize=7)
ax.set_ylabel("% of annotated beats the detector missed")
ax.set_title("Detector misses by beat type\n(missed beats never reach the classifier)")
missed_v = ann[(ann.symbol == "V") & (~ann.detected)]
picks = missed_v.groupby("record").sample(1, random_state=0).sample(7, random_state=0)
detected_all = d.groupby("record").r_peak_sample.apply(np.array)
for ax, b in zip(axes.flat[1:], picks.itertuples()):
    filtered, fs = _filtered_record(b.record)
    lo, hi = int(b.sample - 1.0 * fs), int(b.sample + 1.0 * fs)
    t = (np.arange(lo, hi) - b.sample) / fs
    ax.plot(t, filtered[lo:hi], color="black", lw=0.9)
    ax.axvline(0, color="red", ls="--", lw=1)
    peaks = detected_all[b.record]
    peaks = peaks[(peaks > lo) & (peaks < hi)]
    ax.plot((peaks - b.sample) / fs, filtered[peaks], "v", color="#1f77b4", ms=6)
    ax.set_title(f"rec {b.record}: annotated V (red) not detected", fontsize=9)
    ax.set_xlabel("s (blue triangles = beats the pipeline kept)", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "32_detector_misses.png", dpi=120)
plt.close(fig)

# ---- 33: record 232 ------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axes[0]
groups = {"232, labelled A (abnormal)": d[(d.record == "232") & (d.symbol == "A")],
          "232, labelled R (normal)": d[(d.record == "232") & (d.symbol == "R")],
          "other dev patients, labelled N": d[(d.part == "dev") & (d.record != "232") & (d.symbol == "N")]}
for (name, g), color in zip(groups.items(), ["#ff7f0e", "#1f77b4", "black"]):
    sample = g.sample(min(200, len(g)), random_state=0)
    from src.feature_extraction import get_beat_waveform
    waves = np.array([get_beat_waveform(b.record, int(b.r_peak_sample))[1] for b in sample.itertuples()])
    t = get_beat_waveform(sample.iloc[0].record, int(sample.iloc[0].r_peak_sample))[0]
    ax.plot(t, np.median(waves, axis=0), color=color, lw=2, label=f"{name} (median of {len(sample)})")
ax.set_xlabel("ms from R peak")
ax.set_ylabel("mV (filtered)")
ax.set_title("Median beat shapes: 232's 'A' and 'R' beats look alike, and unlike\nthe training patients' normal beats")
ax.legend(fontsize=8)
r232 = d[d.record == "232"].sort_values("r_peak_sample")
ax = axes[1]
seg = r232.iloc[500:560]
ax.plot(seg.time_s, seg.rr_pre_s, "o-", color="gray", lw=0.8, ms=3)
for sym, color in [("A", "#ff7f0e"), ("R", "#1f77b4")]:
    s = seg[seg.symbol == sym]
    ax.plot(s.time_s, s.rr_pre_s, "o", color=color, ms=6, label=f"'{sym}'")
ax.set_xlabel("time in record (s)")
ax.set_ylabel("RR interval before the beat (s)")
ax.set_title("Record 232: rhythm is what separates its A and R beats (RR before the beat), not shape")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "33_record_232.png", dpi=120)
plt.close(fig)
print("saved figures 28-33")
