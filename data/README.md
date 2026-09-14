# Data

This project uses the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/),
accessed via the [`wfdb`](https://pypi.org/project/wfdb/) Python package.

The raw signal files are **not** committed to this repository (they're
redistributable but add unnecessary bloat to a code repo). Instead, records
are streamed directly from PhysioNet on demand, e.g.:

```python
import wfdb

record = wfdb.rdrecord("100", pn_dir="mitdb")
annotation = wfdb.rdann("100", "atr", pn_dir="mitdb")
```

The first read for a given record downloads it locally (see `.gitignore`)
and reuses the local copy on subsequent runs.
