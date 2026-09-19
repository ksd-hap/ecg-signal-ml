from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fixed, physiologically motivated bounds - chosen by reasoning, not fitted to any data, so they cannot leak.
# RR intervals: 0.3 s (200 bpm) to 2.0 s (30 bpm). Ratios of intervals: at most 3x faster or slower than the reference.
RR_SECONDS_BOUNDS = (0.30, 2.0)
RR_RATIO_BOUNDS = (1 / 3, 3.0)
DEFAULT_CLIP_BOUNDS = {
    "rr_pre_s": RR_SECONDS_BOUNDS,
    "rr_post_s": RR_SECONDS_BOUNDS,
    "rr_ratio": RR_RATIO_BOUNDS,
    "rr_pre_rel": RR_RATIO_BOUNDS,
    "rr_post_rel": RR_RATIO_BOUNDS,
}


class FixedClipper(BaseEstimator, TransformerMixin):
    """Clip the named columns of a DataFrame to fixed bounds. Learns nothing from the data."""

    def __init__(self, bounds=None):
        self.bounds = bounds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column, (low, high) in (self.bounds or {}).items():
            X[column] = X[column].clip(low, high)
        return X


def make_logistic_regression(scale=True, class_weight=None, C=1.0):
    """clip RR outliers (fixed bounds) -> standardise (fitted on the training data only) -> L2 logistic regression."""
    steps = [("clip", FixedClipper(DEFAULT_CLIP_BOUNDS))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)))
    return Pipeline(steps)
