"""Runs every experiment and regenerates all tables, figures and the dashboard data.

    python run_all.py            # full run (about 12 minutes on one CPU core)
    python run_all.py --quick    # small run to check that everything works
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.evaluate import METHODS, METHOD_LABELS, score
from src.monitor import run

os.makedirs("results", exist_ok=True)
os.makedirs("web", exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

MAIN_SEEDS = list(range(100, 103 if args.quick else 115))        # held-out: never used for design
DIFF_SEEDS = list(range(200, 202 if args.quick else 210))
SIZE_SEEDS = list(range(300, 301 if args.quick else 305))
SIZES = [250, 500] if args.quick else [250, 500, 1000, 2000]


def log(msg):
    print(msg, flush=True)


def aggregate(scores):
    """Average the per-seed scores of each method."""
    rows = []
    for m in METHODS:
        get = lambda k: [s["methods"][m][k] for s in scores]
        nanmean = lambda v: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
        delays = [d for d in get("delay_from_drift_start") if d is not None]
        leads = [d for d in get("lead_over_labels") if d is not None]
        rows.append({
            "method": METHOD_LABELS[m], "key": m,
            "recall_impactful": np.mean(get("recall_impactful")),
            "recall_std": np.std(get("recall_impactful")),
            "false_alarm_clean": np.mean(get("false_alarm_clean")),
            "false_alarm_harmless": np.mean(get("false_alarm_harmless")),
            "false_alarm_harmless_std": np.std(get("false_alarm_harmless")),
            "precision": nanmean(np.array(get("precision"), dtype=float)),
            "delay_windows": float(np.mean(delays)) if delays else float("nan"),
            "lead_over_labels": float(np.mean(leads)) if leads else float("nan"),
            "detected_in_seeds": f"{len(delays)}/{len(scores)}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E1-E5: main experiment
log(f"Main experiment: {len(MAIN_SEEDS)} held-out seeds")
main_scores = []
for s in MAIN_SEEDS:
    main_scores.append(score(run(s)))
    log(f"  seed {s} done")
main = aggregate(main_scores)
main.to_csv("results/summary_multi_seed.csv", index=False)
acc_rows = pd.DataFrame(
    [{"ref_acc": s["ref_acc"], **s["accuracy_by_scenario"]} for s in main_scores]).mean().to_frame("mean_accuracy").T
acc_rows.to_csv("results/accuracy_by_scenario.csv", index=False)
log(main.round(3).to_string())
log(acc_rows.round(3).to_string())

# ---------------------------------------------------------------- Drift hits other features
log("Different-feature experiment")
diff_scores = [score(run(s, impact_cols=(2, 3))) for s in DIFF_SEEDS]
diff = aggregate(diff_scores)
diff.to_csv("results/drift_on_other_features.csv", index=False)
log(diff.round(3).to_string())

# ---------------------------------------------------------------- Window-size sensitivity
log("Window-size experiment")
size_rows = []
for size in SIZES:
    sc = [score(run(s, window_size=size)) for s in SIZE_SEEDS]
    agg = aggregate(sc)
    agg.insert(0, "window_size", size)
    size_rows.append(agg)
    log(f"  size {size} done")
sizes = pd.concat(size_rows)
sizes.to_csv("results/window_size.csv", index=False)
log(sizes[["window_size", "key", "recall_impactful", "false_alarm_clean", "false_alarm_harmless"]].round(3).to_string())

# ---------------------------------------------------------------- Demo run for the dashboard
log("Demo run (seed 42)")
demo = run(C.SEED)
demo_score = score(demo)
json.dump({"run": demo, "score": demo_score}, open("results/demo_run.json", "w"), indent=1)

def clean(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))

payload = {
    "config": {"label_delay": C.LABEL_DELAY, "window_size": C.WINDOW_SIZE,
               "impact_start": C.IMPACT_START, "harmless": [C.HARMLESS_START, C.HARMLESS_END],
               "n_windows": C.N_WINDOWS, "persistence": C.PERSISTENCE,
               "acc_drop_confirm": C.ACC_DROP_CONFIRM},
    "methods": [{"key": m, "label": METHOD_LABELS[m]} for m in METHODS],
    "demo": demo, "demo_score": demo_score,
    "summary": clean(main), "other_features": clean(diff), "window_size": clean(sizes),
    "accuracy_by_scenario": json.loads(acc_rows.to_json(orient="records"))[0],
    "n_seeds": len(MAIN_SEEDS),
}
with open("web/results.js", "w") as f:
    f.write("window.DRIFT = " + json.dumps(payload) + ";\n")

# ---------------------------------------------------------------- Figures
COL = {"clean": "#ffffff", "harmless": "#fde9c4", "impactful": "#f7d0cc"}
recs = demo["records"]
t = np.arange(len(recs))

def shade(ax):
    for r in recs:
        ax.axvspan(r["t"] - 0.5, r["t"] + 0.5, color=COL[r["scenario"]], lw=0)

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2.2]})
shade(ax[0]); ax[0].plot(t, [r["accuracy"] for r in recs], color="#1f3a5f", marker="o", ms=3)
ax[0].axhline(demo["ref_acc"], color="gray", ls="--", lw=1, label="reference accuracy")
if demo_score["labels_confirm_at"] is not None and demo_score["labels_confirm_at"] < len(recs):
    ax[0].axvline(demo_score["labels_confirm_at"], color="#b03a2e", lw=1.2, label="labels confirm the drop")
ax[0].set_ylabel("True accuracy"); ax[0].legend(loc="lower left", fontsize=8)
ax[0].set_title("Simulated production stream (white = clean, amber = harmless shift, red = impactful drift)", fontsize=10)

shade(ax[1])
ax[1].plot(t, [r["auc"] for r in recs], color="#0f766e", marker="o", ms=3, label="B: classifier AUC")
ax[1].axhline(demo["auc_threshold"], color="#0f766e", ls=":", lw=1)
ax[1].set_ylabel("Classifier AUC")
ax2 = ax[1].twinx(); ax2.plot(t, [r["pred_ks"] for r in recs], color="#c77d0a", marker="s", ms=3, label="D: prediction KS")
ax2.axhline(C.PRED_KS_MIN_EFFECT, color="#c77d0a", ls=":", lw=1); ax2.set_ylabel("Prediction KS distance")
ax[1].set_ylim(0.4, 1.0)

shade(ax[2])
for i, m in enumerate(METHODS):
    for r in recs:
        if r["alerts"][m]:
            ax[2].add_patch(plt.Rectangle((r["t"] - 0.45, i - 0.4), 0.9, 0.8, color="#b03a2e"))
ax[2].set_yticks(range(len(METHODS))); ax[2].set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
ax[2].set_ylim(-0.6, len(METHODS) - 0.4); ax[2].invert_yaxis(); ax[2].set_xlabel("Window"); ax[2].set_xlim(-0.5, len(recs) - 0.5)
plt.tight_layout(); plt.savefig("results/fig1_timeline.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(METHODS)); w = 0.27
ax.bar(x - w, main["recall_impactful"], w, yerr=main["recall_std"], color="#1f3a5f", label="Detected impactful windows (higher is better)")
ax.bar(x, main["false_alarm_harmless"], w, yerr=main["false_alarm_harmless_std"], color="#c77d0a", label="Alerts on harmless-shift windows (lower is better)")
ax.bar(x + w, main["false_alarm_clean"], w, color="#9aa5b1", label="Alerts on clean windows (lower is better)")
ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS[m].replace(": ", ":\n") for m in METHODS], fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel("Fraction of windows"); ax.legend(fontsize=8, loc="upper center", ncol=1)
ax.set_title(f"Mean over {len(MAIN_SEEDS)} held-out seeds", fontsize=10)
plt.tight_layout(); plt.savefig("results/fig2_detection_vs_false_alarms.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for m, c in [("baseline", "#9aa5b1"), ("A_feature_tests", "#c77d0a"), ("B_classifier", "#0f766e"), ("system", "#1f3a5f")]:
    sub = sizes[sizes.key == m]
    axs[0].plot(sub.window_size, sub.recall_impactful, marker="o", color=c, label=METHOD_LABELS[m])
    axs[1].plot(sub.window_size, sub.false_alarm_harmless, marker="o", color=c, label=METHOD_LABELS[m])
axs[0].set_title("Detected impactful windows", fontsize=10); axs[1].set_title("Alerts on harmless-shift windows", fontsize=10)
for a in axs: a.set_xlabel("Window size (rows)"); a.set_ylim(-0.02, 1.05); a.set_xscale("log"); a.set_xticks(SIZES); a.set_xticklabels(SIZES)
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/fig3_window_size.png", dpi=150); plt.close()
log("All done.")
