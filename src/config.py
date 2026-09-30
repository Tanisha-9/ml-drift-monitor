"""All experiment settings in one place."""

SEED = 42            # seed used for the demo run shown on the dashboard
N_FEATURES = 10      # f0-f4 informative, f5-f9 pure noise
N_INFORMATIVE = 5
WINDOW_SIZE = 1000   # requests per evaluation window
N_WINDOWS = 40       # windows in the simulated production stream
N_CALIB = 8          # clean windows used only to calibrate the classifier threshold
LABEL_DELAY = 3      # true labels for window t become visible at window t + 3
ACC_DROP_CONFIRM = 0.05   # label-based confirmation: accuracy fell >= 5 points below reference

# Simulated production timeline (window index -> scenario)
#   0-9   clean      nothing changes
#   10-17 harmless   noise features shift (model does not really use them)
#   18-23 clean      back to normal
#   24-39 impactful  important features drift gradually
HARMLESS_START, HARMLESS_END = 10, 18
IMPACT_START = 24

ALPHA_BASELINE = 0.05      # single-feature KS test, no correction
ALPHA_FEATURES = 0.05      # per-feature KS, Bonferroni-corrected by N_FEATURES
ALPHA_PREDICTION = 0.01    # KS test on predicted probabilities
PERSISTENCE = 2            # alert only if flagged in this many consecutive windows

# Effect-size floor for the prediction monitor. A KS test on 1000 rows flags tiny,
# statistically "significant" wiggles; requiring a KS distance >= 0.10 makes the
# monitor react only to changes in predictions that are big enough to matter.
PRED_KS_MIN_EFFECT = 0.10
