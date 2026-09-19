from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
    "rr_pre_vs_hist": RR_RATIO_BOUNDS,
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
            if column in X.columns:
                X[column] = X[column].clip(low, high)
        return X


def make_logistic_regression(scale=True, class_weight=None, C=1.0):
    """clip RR outliers (fixed bounds) -> standardise (fitted on the training data only) -> L2 logistic regression."""
    steps = [("clip", FixedClipper(DEFAULT_CLIP_BOUNDS))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)))
    return Pipeline(steps)


def make_random_forest(class_weight=None, n_estimators=300, random_state=42):
    """Random forest on the raw features: trees only compare a feature to thresholds, so no scaling or clipping.

    Everything else is left at scikit-learn defaults (fully grown trees, sqrt(n_features) tried per split).
    """
    return RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight,
                                  random_state=random_state, n_jobs=-1)

def make_gradient_boosting(class_weight=None, max_iter=100, random_state=42):
    """Histogram-based gradient boosting on the raw features (no scaling or clipping needed).

    scikit-learn defaults (learning rate 0.1, up to 31 leaves per tree, min 20 beats per leaf) with a fixed number of
    rounds. Early stopping is switched off: 'auto' would hold out a random 10% of the *training beats* as validation,
    mixing patients between fit and validation inside the training data.
    """
    return HistGradientBoostingClassifier(max_iter=max_iter, class_weight=class_weight, early_stopping=False,
                                          random_state=random_state)
