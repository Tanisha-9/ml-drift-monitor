"""Builds the simulated production stream and injects drift."""
import numpy as np
from sklearn.datasets import make_classification

from . import config as C


def scenario_of(t):
    """Ground-truth scenario of evaluation window t (used only for scoring)."""
    if C.HARMLESS_START <= t < C.HARMLESS_END:
        return "harmless"
    if t >= C.IMPACT_START:
        return "impactful"
    return "clean"


def apply_drift(X, t, impact_cols=(0, 1)):
    """Return a copy of window X with the drift for window t injected."""
    X = X.copy()
    s = scenario_of(t)
    if s == "harmless":
        X[:, C.N_INFORMATIVE:] += 1.0            # shift the noise features
    elif s == "impactful":
        k = t - C.IMPACT_START + 1               # 1, 2, ... grows every window
        X[:, list(impact_cols)] += 0.12 * k     # gradually shift two informative features
    return X


def make_stream(seed=C.SEED, window_size=C.WINDOW_SIZE):
    """Draw one big i.i.d. dataset, then split it into train / reference / windows.

    Because everything comes from the same generator, all windows are clean
    until apply_drift() changes them.
    """
    n_train, n_ref = 5000, 3000
    total = n_train + n_ref + window_size * (C.N_CALIB + C.N_WINDOWS)
    X, y = make_classification(
        n_samples=total, n_features=C.N_FEATURES, n_informative=C.N_INFORMATIVE,
        n_redundant=0, n_clusters_per_class=2, flip_y=0.02,
        shuffle=False, random_state=seed)
    p = np.random.RandomState(seed).permutation(total)
    X, y = X[p], y[p]

    i = 0
    out = {}
    out["X_train"], out["y_train"] = X[i:i + n_train], y[i:i + n_train]; i += n_train
    out["X_ref"], out["y_ref"] = X[i:i + n_ref], y[i:i + n_ref]; i += n_ref
    out["calib"], out["win_X"], out["win_y"] = [], [], []
    for _ in range(C.N_CALIB):
        out["calib"].append(X[i:i + window_size]); i += window_size
    for _ in range(C.N_WINDOWS):
        out["win_X"].append(X[i:i + window_size])
        out["win_y"].append(y[i:i + window_size]); i += window_size
    return out
