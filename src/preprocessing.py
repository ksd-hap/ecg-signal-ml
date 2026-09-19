from pathlib import Path

import wfdb
from scipy.signal import butter, filtfilt

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mitdb"


def load_record(record_name, data_dir=DEFAULT_DATA_DIR):
    """Load a MIT-BIH record and its reference annotations, caching the files locally (gitignored)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = [f"{record_name}.{ext}" for ext in ("hea", "dat", "atr")]
    missing = [f for f in wanted if not (data_dir / f).exists()]
    if missing:
        wfdb.dl_files("mitdb", str(data_dir), missing)
    base = str(data_dir / record_name)
    return wfdb.rdrecord(base), wfdb.rdann(base, "atr")


def get_ecg_lead(record, lead_name="MLII"):
    """Return (signal, lead_name); fail loudly rather than silently falling back to a different lead."""
    if lead_name not in record.sig_name:
        raise ValueError(f"Lead {lead_name} not in record (has {record.sig_name})")
    return record.p_signal[:, record.sig_name.index(lead_name)], lead_name


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Zero-phase Butterworth bandpass filter for ECG."""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)
