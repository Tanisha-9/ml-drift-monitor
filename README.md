# Drift Monitor: detecting when a deployed model's data has quietly changed

Learn Depth Academy LLP · Track 2 Advanced ML Research Internship · Project 01 · Problem ID **ML-T2-060**

A small, readable monitoring system for a deployed ML model. It watches incoming data window by window and
answers one question: **did the data change in a way that will hurt the model, or is it a harmless wiggle?**

**Live demo:** https://ml-drift-monitor-web.vercel.app/· **Report:** [REPORT.md](REPORT.md)

## What is in this repo

| Path | What it is |
|---|---|
| `src/data.py` | Builds the simulated production stream and injects drift |
| `src/detectors.py` | The three drift checks (per-feature KS test, classifier detector, prediction monitor) |
| `src/monitor.py` | Trains the "deployed" model, runs the checks on every window, applies the alert rule |
| `src/evaluate.py` | Scores alerts against the known ground truth |
| `src/config.py` | Every setting in one file |
| `run_all.py` | Runs all experiments, writes tables, figures and the dashboard data |
| `web/` | Static dashboard (`index.html` + `results.js`). This is what you deploy on Vercel |
| `results/` | CSV tables and PNG figures used in the report |
| `REPORT.md` | Technical report |
| `docs/VIDEO_SCRIPT.md` | Script for the demo video |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --quick      # about 1 minute, checks everything works
python run_all.py              # full run, about 12 minutes, regenerates every number in the report
python -m http.server 8000 --directory web             # then open http://localhost:8000
```

Tested with Python 3.12, numpy 2.4, scipy 1.17, scikit-learn 1.8, pandas 3.0, matplotlib 3.10. All randomness is seeded.
The committed `web/results.js` and `results/` files come from the full run, so you can deploy without running anything.

## How it works (short version)

1. A random forest is trained on a synthetic tabular dataset (10 features, 5 informative, 5 noise).
2. The stream has 40 windows of 1,000 rows: clean, then a **harmless** shift (noise features move), then clean again,
   then an **impactful** drift (two important features slowly shift and accuracy falls).
3. Every window is checked three ways:
   - **A**: a Kolmogorov-Smirnov test on each feature (Bonferroni corrected)
   - **B**: a classifier that tries to tell reference rows from new rows (high AUC means drift)
   - **D**: a KS test on the model's predicted probabilities, with a minimum effect size
4. **Combined system**: alert only if (A or B) fires **and** either an important feature moved or the predictions moved,
   in two windows in a row.
5. True labels are hidden for 3 windows to copy real delayed feedback. The main metric is how many windows before
   the labels would have shown the damage the alert fires.

## Headline results (15 held-out seeds)

| Method | Detects impactful drift | Alerts on harmless shift | Alerts that were right |
|---|---|---|---|
| Baseline: KS on top feature | 30% | 0% | 86% |
| A: per-feature KS | 90% | 88% | 67% |
| B: classifier detector | 83% | 88% | 66% |
| D: prediction monitor | 59% | 0% | 100% |
| **Combined system** | **87%** | **0%** | **100%** |

Model accuracy: 91.5% on clean windows, 91.5% on harmless-shift windows, 83.2% on impactful-drift windows.
The combined system fires about 8 windows before delayed labels confirm a 5-point accuracy drop.
See [REPORT.md](REPORT.md) for the full experiments and limitations.

## Push to GitHub

```bash
git init
git add .
git commit -m "Drift monitor: ML-T2-060"
git branch -M main
git remote add origin https://github.com/<your-username>/drift-monitor.git
git push -u origin main
```

## Deploy the dashboard on Vercel

1. Go to vercel.com, click **Add New → Project**, and import the GitHub repo.
2. Set **Root Directory** to `web`.
3. Set **Framework Preset** to `Other`. Leave Build Command and Output Directory empty.
4. Click **Deploy**. Paste the link at the top of this README.

The dashboard is plain HTML, CSS and JavaScript with no build step and no external scripts.
