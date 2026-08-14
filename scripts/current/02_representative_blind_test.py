# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:59:52 2026

@author: Metin Zontul
"""

# -*- coding: utf-8 -*-
"""
Representative blind-test experiment with leakage-free hyperparameter tuning,
parameter-matched SNN/Classical NN ablation, empirical baselines, and
node-level SNN explainability.

Protocol
--------
Months 1--72   : trend fit for hyperparameter search
Months 73--96  : validation for independent SNN/Classical NN grid searches
Months 1--96   : refit/final training after hyperparameters are fixed
Months 97--120 : untouched blind test
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONSTANTS AND SPLIT
# ============================================================
SEED = 42
MONTHS_TOTAL = 120
FIT_SIZE = 72
VALIDATION_SIZE = 24
TRAIN_SIZE = FIT_SIZE + VALIDATION_SIZE   # 96
TEST_SIZE = MONTHS_TOTAL - TRAIN_SIZE     # 24

MONTHS = np.arange(1, MONTHS_TOTAL + 1)
X_NODES = -1.0 + 2.0 * (MONTHS - 1) / (MONTHS_TOTAL - 1)

X_FIT = X_NODES[:FIT_SIZE]
X_VAL = X_NODES[FIT_SIZE:TRAIN_SIZE]
X_TRAIN = X_NODES[:TRAIN_SIZE]
X_TEST = X_NODES[TRAIN_SIZE:]

N_GRID = [16, 32, 64]
T_GRID = [0.5, 2.0, 3.0]   # t=1 excluded: degenerate reciprocal case
XI_GRID = [0.5, 1.0, 2.0]
TIE_TOL = 1e-12

OUTPUT_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. DATA GENERATION
# ============================================================
rng = np.random.RandomState(SEED)
T_DATA = (
    15.0
    + 10.0 * np.sin(2.0 * np.pi * MONTHS / 12.0 - np.pi / 2.0)
    + 0.05 * MONTHS
    + rng.normal(0.0, 1.2, MONTHS_TOTAL)
)

FIT_DATA = T_DATA[:FIT_SIZE]
VAL_DATA = T_DATA[FIT_SIZE:TRAIN_SIZE]
TRAIN_DATA = T_DATA[:TRAIN_SIZE]
TEST_DATA = T_DATA[TRAIN_SIZE:]

# For normalized reporting only
Y_MIN_REPORT = np.min(T_DATA)
Y_MAX_REPORT = np.max(T_DATA)


# ============================================================
# 3. COMMON FUNCTIONS
# ============================================================
def x_to_month(x):
    return ((x + 1.0) / 2.0) * (MONTHS_TOTAL - 1) + 1.0


def seasonal_trend_func(x, A, w, phi, B, gamma):
    m = x_to_month(x)
    return A * np.sin(w * m + phi) + B + gamma * m


def k_activation(x, t, xi):
    z = np.clip(-xi * x, -np.inf, 200.0)
    return 1.0 / (1.0 + t * np.exp(z))


def k_sym(x, t, xi):
    return 0.5 * (k_activation(x, t, xi) + k_activation(x, 1.0 / t, xi))


def aleph_density(x, t, xi):
    return 0.5 * (k_activation(x + 1.0, t, xi) - k_activation(x - 1.0, t, xi))


def F_density(x, t, xi):
    return 0.5 * (k_sym(x + 1.0, t, xi) - k_sym(x - 1.0, t, xi))


def generalized_operator(x_eval, n, density_func, t, xi, popt, return_weights=False):
    a_n = float(n)
    b_n = np.sqrt(n**2 + 2.0)
    m_vals = np.arange(int(np.ceil(-b_n)), int(np.floor(b_n)) + 1)

    f_nodes = seasonal_trend_func(m_vals / b_n, *popt)

    z = a_n * np.asarray(x_eval)[:, None] - m_vals[None, :]
    weights = density_func(z, t=t, xi=xi)
    den = np.sum(weights, axis=1)

    if np.any(den <= 0.0):
        raise FloatingPointError("Encountered a non-positive normalization denominator.")

    preds = (weights @ f_nodes) / den

    if return_weights:
        return preds, weights, m_vals, b_n
    return preds


def get_metrics(preds, true_vals):
    preds_norm = (preds - Y_MIN_REPORT) / (Y_MAX_REPORT - Y_MIN_REPORT)
    true_norm = (true_vals - Y_MIN_REPORT) / (Y_MAX_REPORT - Y_MIN_REPORT)

    return {
        "NMAE": mean_absolute_error(true_norm, preds_norm),
        "NRMSE": np.sqrt(mean_squared_error(true_norm, preds_norm)),
        "R2": r2_score(true_vals, preds),
    }


def run_grid_search(val_x, val_y, popt, density_func, model_name):
    records = []

    for n in N_GRID:
        for t in T_GRID:
            for xi in XI_GRID:
                preds = generalized_operator(val_x, n, density_func, t, xi, popt)
                records.append({
                    "Model": model_name,
                    "n": n,
                    "t": t,
                    "xi": xi,
                    "Validation_R2": r2_score(val_y, preds),
                })

    df = pd.DataFrame(records)
    best_score = df["Validation_R2"].max()

    tied = df[
        np.isclose(df["Validation_R2"], best_score, rtol=0.0, atol=TIE_TOL)
    ].copy()

    tied = tied.sort_values(
        ["n", "t", "xi"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    return df, tied


# ============================================================
# 4. LEAKAGE-FREE INDEPENDENT GRID SEARCH
#    Fit on months 1--72, validate on months 73--96
# ============================================================
p0 = [10.0, 2.0 * np.pi / 12.0, -np.pi / 2.0, 15.0, 0.05]

popt_tuning, _ = curve_fit(
    seasonal_trend_func,
    X_FIT,
    FIT_DATA,
    p0=p0,
    maxfev=20000
)

snn_grid, snn_ties = run_grid_search(X_VAL, VAL_DATA, popt_tuning, F_density, "SNN")
cnn_grid, cnn_ties = run_grid_search(X_VAL, VAL_DATA, popt_tuning, aleph_density, "Classical NN")

# Classical NN unique validation-best representative
cnn_best = cnn_ties.iloc[0][["n", "t", "xi"]].to_dict()

# SNN may have reciprocal-equivalent ties, so we choose the matched representative
matched_mask = (
    (snn_ties["n"] == cnn_best["n"])
    & np.isclose(snn_ties["t"], cnn_best["t"], atol=TIE_TOL, rtol=0.0)
    & np.isclose(snn_ties["xi"], cnn_best["xi"], atol=TIE_TOL, rtol=0.0)
)

if not matched_mask.any():
    raise RuntimeError(
        "No parameter-matched SNN optimum exists for the Classical NN optimum; "
        "the experiment cannot be described as a matched ablation."
    )

snn_selected = snn_ties.loc[matched_mask].iloc[0][["n", "t", "xi"]].to_dict()
cnn_selected = cnn_best

for key in ("n", "t", "xi"):
    if not np.isclose(float(snn_selected[key]), float(cnn_selected[key]), atol=TIE_TOL, rtol=0.0):
        raise RuntimeError("SNN and Classical NN parameters are not matched.")

print("\n=== Selected Parameter-Matched Configuration ===")
print(f"SNN          : n={int(snn_selected['n'])}, t={snn_selected['t']}, xi={snn_selected['xi']}")
print(f"Classical NN : n={int(cnn_selected['n'])}, t={cnn_selected['t']}, xi={cnn_selected['xi']}")
print("\nSNN tied validation optima:")
print(snn_ties[["n", "t", "xi", "Validation_R2"]].to_string(index=False))

pd.concat([snn_grid, cnn_grid], ignore_index=True).to_csv(
    OUTPUT_DIR / "Example3_GridSearch_Validation_Results.csv",
    index=False
)


# ============================================================
# 5. REFIT ON MONTHS 1--96, THEN BLIND TEST ON MONTHS 97--120
# ============================================================
popt_final, _ = curve_fit(
    seasonal_trend_func,
    X_TRAIN,
    TRAIN_DATA,
    p0=p0,
    maxfev=20000
)

opt_n = int(snn_selected["n"])
opt_t = float(snn_selected["t"])
opt_xi = float(snn_selected["xi"])

snn_preds, snn_weights_matrix, m_vals, opt_b_n = generalized_operator(
    X_TEST,
    opt_n,
    F_density,
    opt_t,
    opt_xi,
    popt_final,
    return_weights=True
)

cnn_preds = generalized_operator(
    X_TEST,
    opt_n,
    aleph_density,
    opt_t,
    opt_xi,
    popt_final
)


# ============================================================
# 6. EMPIRICAL BASELINES: MLP AND SEASONAL ARIMA
# ============================================================
X_train_mlp = np.column_stack((
    MONTHS[:TRAIN_SIZE],
    np.sin(2.0 * np.pi * MONTHS[:TRAIN_SIZE] / 12.0),
    np.cos(2.0 * np.pi * MONTHS[:TRAIN_SIZE] / 12.0),
))

X_test_mlp = np.column_stack((
    MONTHS[TRAIN_SIZE:],
    np.sin(2.0 * np.pi * MONTHS[TRAIN_SIZE:] / 12.0),
    np.cos(2.0 * np.pi * MONTHS[TRAIN_SIZE:] / 12.0),
))

scaler = StandardScaler()
X_train_mlp_scaled = scaler.fit_transform(X_train_mlp)
X_test_mlp_scaled = scaler.transform(X_test_mlp)

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=2000,
    random_state=SEED,
    solver="lbfgs",
)

mlp.fit(X_train_mlp_scaled, TRAIN_DATA)
mlp_preds = mlp.predict(X_test_mlp_scaled)

arima = ARIMA(
    TRAIN_DATA,
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 12),
).fit()

arima_preds = arima.forecast(steps=TEST_SIZE)


# ============================================================
# 7. BLIND-TEST METRICS
# ============================================================
metrics = {
    "SNN": get_metrics(snn_preds, TEST_DATA),
    "Classical NN": get_metrics(cnn_preds, TEST_DATA),
    "Seasonal ARIMA": get_metrics(arima_preds, TEST_DATA),
    "MLP": get_metrics(mlp_preds, TEST_DATA),
}

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(OUTPUT_DIR / "Example3_Representative_BlindTest_Metrics.csv")

print("\n=== Representative Blind-Test Performance (Seed 42; Months 97--120) ===")
print(metrics_df.round(4).to_string())


# ============================================================
# 8. FIGURE 7: BLIND-TEST PREDICTION COMPARISON
# ============================================================
plt.figure(figsize=(12, 5))

plt.plot(
    MONTHS[TRAIN_SIZE:],
    TEST_DATA,
    "ko-",
    linewidth=2.5,
    markersize=6,
    label="Actual Data"
)

plt.plot(
    MONTHS[TRAIN_SIZE:],
    snn_preds,
    "r^-",
    linewidth=2.0,
    markersize=8,
    label=fr"SNN ($n={opt_n},\, t={opt_t},\, \xi={opt_xi}$)"
)

plt.plot(
    MONTHS[TRAIN_SIZE:],
    cnn_preds,
    "ms--",
    linewidth=2.0,
    markersize=6,
    alpha=0.7,
    label=fr"Classical NN ($n={opt_n},\, t={opt_t},\, \xi={opt_xi}$)"
)

plt.plot(
    MONTHS[TRAIN_SIZE:],
    arima_preds,
    "b--",
    linewidth=2.0,
    label=r"Seasonal ARIMA $(1,0,1)(0,1,1)_{12}$"
)

plt.plot(
    MONTHS[TRAIN_SIZE:],
    mlp_preds,
    "g-.",
    linewidth=2.0,
    label="MLP (100, 50)"
)

plt.title("Blind-Test Prediction Comparison (Months 97--120)", fontsize=14)
plt.xlabel("Month Index", fontsize=12)
plt.ylabel("Temperature-like Value", fontsize=12)
plt.xticks(np.arange(97, 121, 2))
plt.legend(fontsize=9, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure_Example3_Baseline_Predictions_600dpi.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. FIGURE: NODE-LEVEL EXPLAINABILITY FOR MONTH 98
# ============================================================
target_idx = 1   # second blind-test observation = month 98
target_month = TRAIN_SIZE + target_idx + 1

normalized_weights = snn_weights_matrix[target_idx] / np.sum(snn_weights_matrix[target_idx])
mapped_months = x_to_month(m_vals / opt_b_n)

plt.figure(figsize=(12, 4))
plt.bar(
    mapped_months,
    normalized_weights,
    edgecolor="black",
    alpha=0.8,
    width=1.5
)

plt.axvline(
    x=target_month,
    linestyle="--",
    linewidth=2.5,
    label=f"Target Prediction (Month {target_month})",
)

plt.title(
    f"SNN Node-Level Explainability: Kernel Weight Distribution for Target Month {target_month}",
    fontsize=14,
)

plt.xlabel("SNN Sampling Nodes (Mapped to Global Month Scale)", fontsize=12)
plt.ylabel("Normalized SNN Kernel Weights", fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure_Example3_Node_Explainability_Month98_600dpi.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FINAL OUTPUT MESSAGES
# ============================================================
print("\nSaved publication-ready outputs:")
print("- Example3_GridSearch_Validation_Results.csv")
print("- Example3_Representative_BlindTest_Metrics.csv")
print("- Figure_Example3_Baseline_Predictions_600dpi.png")
print("- Figure_Example3_Node_Explainability_Month98_600dpi.png")