"""The three drift checks used in this project."""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def feature_ks_pvalues(ref, win):
    """A. Kolmogorov-Smirnov test on every feature separately."""
    return np.array([ks_2samp(ref[:, j], win[:, j]).pvalue for j in range(ref.shape[1])])


def classifier_auc(ref, win, seed=0):
    """B. Can a classifier tell reference rows from new rows? AUC ~0.5 means no."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ref), size=len(win), replace=False)
    X = np.vstack([ref[idx], win])
    y = np.r_[np.zeros(len(win)), np.ones(len(win))]
    clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=seed, n_jobs=1)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def prediction_ks(ref_scores, win_scores):
    """D. Has the distribution of the model's predicted probabilities changed?"""
    r = ks_2samp(ref_scores, win_scores)
    return float(r.statistic), float(r.pvalue)
