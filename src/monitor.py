"""Runs the monitor over the simulated stream and records what it saw."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from .data import make_stream, apply_drift, scenario_of
from .detectors import feature_ks_pvalues, classifier_auc, prediction_ks


def run(seed=C.SEED, window_size=C.WINDOW_SIZE, impact_cols=(0, 1)):
    d = make_stream(seed, window_size)

    # 1. The "deployed" model
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)
    model.fit(d["X_train"], d["y_train"])
    ref_acc = float(model.score(d["X_ref"], d["y_ref"]))
    ref_scores = model.predict_proba(d["X_ref"])[:, 1]
    importance = model.feature_importances_
    top1 = int(np.argmax(importance))
    top3 = [int(j) for j in np.argsort(importance)[::-1][:3]]

    # 2. Calibrate detector B on clean windows that are never used for evaluation
    calib_auc = [classifier_auc(d["X_ref"], w, seed + i) for i, w in enumerate(d["calib"])]
    auc_threshold = float(max(0.55, np.mean(calib_auc) + 3 * np.std(calib_auc)))

    # 3. Walk through the stream one window at a time
    records = []
    for t, (raw, y) in enumerate(zip(d["win_X"], d["win_y"])):
        X = apply_drift(raw, t, impact_cols)
        pv = feature_ks_pvalues(d["X_ref"], X)
        drifted = pv < C.ALPHA_FEATURES / C.N_FEATURES
        auc = classifier_auc(d["X_ref"], X, seed + 100 + t)
        ks_stat, ks_p = prediction_ks(ref_scores, model.predict_proba(X)[:, 1])
        acc = float(model.score(X, y))

        flags = {
            "baseline": bool(pv[top1] < C.ALPHA_BASELINE),
            "A_feature_tests": bool(drifted.any()),
            "B_classifier": bool(auc > auc_threshold),
            "D_prediction": bool(ks_p < C.ALPHA_PREDICTION and ks_stat >= C.PRED_KS_MIN_EFFECT),
        }
        input_drift = flags["A_feature_tests"] or flags["B_classifier"]
        important_drift = bool(drifted[top3].any())
        flags["system"] = bool(input_drift and (flags["D_prediction"] or important_drift))

        if flags["system"]:
            level = "relevant"      # input drift that touches what the model relies on
        elif input_drift:
            level = "info"          # input changed, but predictions/important features did not
        else:
            level = "ok"

        records.append({
            "t": t, "scenario": scenario_of(t), "accuracy": acc,
            "auc": auc, "pred_ks": ks_stat, "pred_p": ks_p,
            "drifted_features": [int(j) for j in np.where(drifted)[0]],
            "flags": flags, "level": level,
        })

    # 4. Turn per-window flags into alerts: flagged in N consecutive windows
    for m in records[0]["flags"]:
        for t, r in enumerate(records):
            recent = [records[k]["flags"][m] for k in range(max(0, t - C.PERSISTENCE + 1), t + 1)]
            r.setdefault("alerts", {})[m] = bool(len(recent) == C.PERSISTENCE and all(recent))

    return {"seed": seed, "window_size": window_size, "ref_acc": ref_acc,
            "importance": [float(v) for v in importance], "top1": top1, "top3": top3,
            "auc_threshold": auc_threshold, "impact_cols": list(impact_cols), "records": records}
