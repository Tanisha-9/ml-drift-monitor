"""Scores a monitor run against the known ground-truth scenarios."""
import numpy as np

from . import config as C

METHODS = ["baseline", "A_feature_tests", "B_classifier", "D_prediction", "system"]
METHOD_LABELS = {
    "baseline": "Baseline: KS on top feature",
    "A_feature_tests": "A: per-feature KS tests",
    "B_classifier": "B: classifier detector",
    "D_prediction": "D: prediction monitor",
    "system": "Combined system",
}


def label_confirmation(run):
    """Window at which delayed labels first *show* an accuracy drop (or None)."""
    for r in run["records"]:
        if r["accuracy"] < run["ref_acc"] - C.ACC_DROP_CONFIRM:
            return r["t"] + C.LABEL_DELAY
    return None


def score(run):
    recs = run["records"]
    scen = np.array([r["scenario"] for r in recs])
    confirmed_at = label_confirmation(run)
    out = {}
    for m in METHODS:
        alert = np.array([r["alerts"][m] for r in recs])
        imp = scen == "impactful"
        n_alerts = alert.sum()
        first = next((r["t"] for r in recs if r["t"] >= C.IMPACT_START and r["alerts"][m]), None)
        out[m] = {
            "recall_impactful": float(alert[imp].mean()),
            "false_alarm_clean": float(alert[scen == "clean"].mean()),
            "false_alarm_harmless": float(alert[scen == "harmless"].mean()),
            "precision": float(alert[imp].sum() / n_alerts) if n_alerts else float("nan"),
            "delay_from_drift_start": None if first is None else first - C.IMPACT_START,
            "lead_over_labels": None if (first is None or confirmed_at is None)
                                else confirmed_at - first,
        }
    acc = {s: float(np.mean([r["accuracy"] for r in recs if r["scenario"] == s]))
           for s in ["clean", "harmless", "impactful"]}
    return {"methods": out, "accuracy_by_scenario": acc,
            "labels_confirm_at": confirmed_at, "ref_acc": run["ref_acc"]}
