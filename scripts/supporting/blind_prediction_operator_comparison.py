# -*- coding: utf-8 -*-
"""
Blind prediction experiment for a seasonal time series with trend.

This script compares two compact-interval kernel operators in the same tables:

1. Classical NN : Classical neural network operator based on the classical
                  density kernel aleph_{t,xi};
2. SNN          : Symmetrized neural network operator based on the symmetric
                  density kernel F.

The comparison is performed under the same leakage-free validation protocol.
Each operator is allowed to select its own best hyperparameter configuration
from the same search grid.

Created on Mon Jun 15 13:35:11 2026

Author: Metin Zontul
Updated: English comments and outputs; Classical NN vs SNN comparison added;
         kernel coefficient aligned with the manuscript.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. Mathematical kernel functions
# ============================================================
def k_func(x, t, xi, S):
    """
    Deformation-dependent sigmoidal activation function.

    k_{t,xi}(x) = 1 / (1 + t S^{-xi x})
    """
    return 1.0 / (1.0 + t * (S ** (-xi * x)))


def aleph_func(x, t, xi, S):
    """
    Classical activation-induced density kernel.

    aleph_{t,xi}(x) = 1/2 [k_{t,xi}(x+1) - k_{t,xi}(x-1)]

    The coefficient 0.5 is used to match the theoretical definition
    in the manuscript.
    """
    return 0.5 * (k_func(x + 1, t, xi, S) - k_func(x - 1, t, xi, S))


def F_func(x, t, xi, S):
    """
    Symmetric activation-induced density kernel.

    F(x) = 1/2 [aleph_{t,xi}(x) + aleph_{1/t,xi}(x)]
    """
    return (aleph_func(x, t, xi, S) + aleph_func(x, 1.0 / t, xi, S)) / 2.0


# ============================================================
# 2. Generalized compact-interval operators
# ============================================================
def L_Generalized_Operator(
    x,
    n,
    t,
    xi,
    S,
    f_func,
    a_n,
    b_n,
    kernel_func,
    a=-1,
    b=1,
):
    """
    Generalized compact-interval normalized kernel operator.

    Parameters
    ----------
    x : float
        Evaluation point.
    n : int
        Scale parameter.
    t : float
        Deformation coefficient.
    xi : float
        Shape parameter.
    S : float
        Exponential base parameter.
    f_func : callable
        Continuous fitted target function.
    a_n : float
        Kernel scaling sequence.
    b_n : float
        Sampling-node scaling sequence.
    kernel_func : callable
        aleph_func for the classical NN operator;
        F_func for the symmetrized SNN operator.
    a, b : float
        Compact interval endpoints.

    Returns
    -------
    float
        Normalized operator value.
    """
    m_min = int(np.ceil(b_n * a))
    m_max = int(np.floor(b_n * b))

    numerator = 0.0
    denominator = 0.0

    for m in range(m_min, m_max + 1):
        node = m / b_n
        f_val = f_func(node)
        weight = kernel_func(a_n * x - m, t, xi, S)

        numerator += f_val * weight
        denominator += weight

    if denominator == 0:
        return 0.0

    return numerator / denominator


def L_Classical_NN(x, n, t, xi, S, f_func, a_n, b_n, a=-1, b=1):
    """
    Classical NN operator based on the non-symmetrized density kernel aleph_{t,xi}.
    """
    return L_Generalized_Operator(
        x=x,
        n=n,
        t=t,
        xi=xi,
        S=S,
        f_func=f_func,
        a_n=a_n,
        b_n=b_n,
        kernel_func=aleph_func,
        a=a,
        b=b,
    )


def L_SNN(x, n, t, xi, S, f_func, a_n, b_n, a=-1, b=1):
    """
    Symmetrized NN operator based on the symmetric density kernel F.
    """
    return L_Generalized_Operator(
        x=x,
        n=n,
        t=t,
        xi=xi,
        S=S,
        f_func=f_func,
        a_n=a_n,
        b_n=b_n,
        kernel_func=F_func,
        a=a,
        b=b,
    )


# ============================================================
# 3. Natural time-scaled baseline function
# ============================================================
def x_to_month(x, total_months=60):
    """
    Map x in [-1, 1] to the monthly scale [1, total_months].
    """
    return ((x + 1.0) / 2.0) * (total_months - 1) + 1.0


def hybrid_func_natural(x, amplitude, frequency, phase, baseline, trend):
    """
    Seasonal sinusoidal model with a linear trend.

    Parameters
    ----------
    amplitude : float
        Seasonal amplitude.
    frequency : float
        Monthly angular frequency.
    phase : float
        Phase shift.
    baseline : float
        Baseline level.
    trend : float
        Linear trend coefficient.
    """
    month = x_to_month(x)
    return amplitude * np.sin(frequency * month + phase) + baseline + trend * month


# ============================================================
# 4. Synthetic seasonal time series generation
# ============================================================
np.random.seed(42)

months_total = 60
months_arr = np.arange(1, months_total + 1)

# Synthetic temperature-like signal:
# - baseline level: 15
# - seasonal amplitude: 10
# - annual period: 12 months
# - warming trend: 0.05 per month
# - Gaussian noise with standard deviation 1.5
T_data = (
    15.0
    + 10.0 * np.sin(2.0 * np.pi * months_arr / 12.0 - np.pi / 2.0)
    + 0.05 * months_arr
    + np.random.normal(0.0, 1.5, months_total)
)

# Normalized input nodes on [-1, 1]
x_nodes = -1.0 + 2.0 * (months_arr - 1) / (months_total - 1)

T_min, T_max = np.min(T_data), np.max(T_data)

# First 48 months: training and validation
# Last 12 months: blind test
x_train_full = x_nodes[:48]
y_train_full = T_data[:48]

x_test_blind = x_nodes[48:]
y_test_blind = T_data[48:]


# ============================================================
# 5. Leakage-free validation split
# ============================================================
# Subtraining set: months 1-36
# Validation set: months 37-48
x_sub_train = x_train_full[:36]
y_sub_train = y_train_full[:36]

x_val = x_train_full[36:]
y_val = y_train_full[36:]

# Initial values and bounds for stable nonlinear curve fitting.
# The frequency is constrained around 2*pi/12 to reflect annual seasonality.
p0 = [10.0, 2.0 * np.pi / 12.0, 0.0, 15.0, 0.0]

lower_bounds = [5.0, 0.4, -np.pi, 0.0, -2.0]
upper_bounds = [20.0, 0.6, np.pi, 30.0, 2.0]

popt_sub, _ = curve_fit(
    hybrid_func_natural,
    x_sub_train,
    y_sub_train,
    p0=p0,
    bounds=(lower_bounds, upper_bounds),
    method="trf",
)


def f_continuous_val(x):
    """
    Continuous fitted function used during validation.
    """
    return hybrid_func_natural(x, *popt_sub)


# ============================================================
# 6. Hyperparameter grids
# ============================================================
n_values = [8, 16, 32]
t_values = [0.5, 1.0, 2.0]
xi_values = [0.5, 1.0, 2.0]
S_val = np.e


def validation_search(operator_name, operator_func):
    """
    Perform leakage-free validation search for a given operator.

    The same grid is used for the classical NN and SNN operators.
    """
    best_val_mae = float("inf")
    best_params = {}

    for n, t, xi in itertools.product(n_values, t_values, xi_values):
        a_n = n
        b_n = np.sqrt(n**2 + 2.0)

        preds_val = [
            operator_func(
                x=x,
                n=n,
                t=t,
                xi=xi,
                S=S_val,
                f_func=f_continuous_val,
                a_n=a_n,
                b_n=b_n,
            )
            for x in x_val
        ]

        current_mae = mean_absolute_error(y_val, preds_val)

        if current_mae < best_val_mae:
            best_val_mae = current_mae
            best_params = {
                "operator": operator_name,
                "n": n,
                "t": t,
                "xi": xi,
                "a_n": a_n,
                "b_n": b_n,
                "validation_mae": best_val_mae,
            }

    return best_params


print("Phase 1: Leakage-free validation search for the classical NN and SNN operators...")

best_params_classical = validation_search(
    operator_name="Classical NN",
    operator_func=L_Classical_NN,
)

best_params_snn = validation_search(
    operator_name="SNN",
    operator_func=L_SNN,
)

config_df = pd.DataFrame(
    [
        best_params_classical,
        best_params_snn,
    ]
)

print("\n--- Best Validation Configurations ---")
print(config_df.to_string(index=False))


# ============================================================
# 7. Final blind-test evaluation
# ============================================================
print("\nPhase 2: Final blind-test comparison on the last 12 months...")

# Refit the continuous seasonal baseline using all available training data
# before the blind-test period.
popt_full, _ = curve_fit(
    hybrid_func_natural,
    x_train_full,
    y_train_full,
    p0=p0,
    bounds=(lower_bounds, upper_bounds),
    method="trf",
)


def f_continuous_final(x):
    """
    Continuous fitted function used for the final blind-test evaluation.
    """
    return hybrid_func_natural(x, *popt_full)


def predict_blind(operator_func, params):
    """
    Compute blind-test predictions for a selected operator and configuration.
    """
    return np.array(
        [
            operator_func(
                x=x,
                n=params["n"],
                t=params["t"],
                xi=params["xi"],
                S=S_val,
                f_func=f_continuous_final,
                a_n=params["a_n"],
                b_n=params["b_n"],
            )
            for x in x_test_blind
        ]
    )


classical_preds = predict_blind(L_Classical_NN, best_params_classical)
snn_preds = predict_blind(L_SNN, best_params_snn)

# Normalized values are reported to make the errors comparable
# relative to the full data range.
y_test_norm = (y_test_blind - T_min) / (T_max - T_min)
classical_norm = (classical_preds - T_min) / (T_max - T_min)
snn_norm = (snn_preds - T_min) / (T_max - T_min)


def compute_metrics(y_true, y_pred, y_true_norm, y_pred_norm):
    """
    Compute raw and normalized blind-test performance metrics.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Normalized MAE": mean_absolute_error(y_true_norm, y_pred_norm),
        "Normalized RMSE": np.sqrt(mean_squared_error(y_true_norm, y_pred_norm)),
        "R2": r2_score(y_true, y_pred),
    }


metrics_classical = compute_metrics(
    y_test_blind,
    classical_preds,
    y_test_norm,
    classical_norm,
)

metrics_snn = compute_metrics(
    y_test_blind,
    snn_preds,
    y_test_norm,
    snn_norm,
)


# ============================================================
# 8. Prediction comparison table
# ============================================================
prediction_comparison_df = pd.DataFrame(
    {
        "Month_Index": range(49, 61),
        "Actual_Value": np.round(y_test_blind, 2),
        "Classical_NN_Prediction": np.round(classical_preds, 2),
        "SNN_Prediction": np.round(snn_preds, 2),
        "Classical_NN_Absolute_Error": np.round(
            np.abs(y_test_blind - classical_preds), 4
        ),
        "SNN_Absolute_Error": np.round(np.abs(y_test_blind - snn_preds), 4),
    }
)


# ============================================================
# 9. Performance metric comparison table
# ============================================================
metric_comparison_df = pd.DataFrame(
    {
        "Metric": ["MAE", "RMSE", "Normalized MAE", "Normalized RMSE", "R2"],
        "Classical_NN": [
            metrics_classical["MAE"],
            metrics_classical["RMSE"],
            metrics_classical["Normalized MAE"],
            metrics_classical["Normalized RMSE"],
            metrics_classical["R2"],
        ],
        "SNN": [
            metrics_snn["MAE"],
            metrics_snn["RMSE"],
            metrics_snn["Normalized MAE"],
            metrics_snn["Normalized RMSE"],
            metrics_snn["R2"],
        ],
    }
)

# For error metrics, positive values mean that SNN reduces the error.
# For R2, positive values mean that SNN increases the score.
metric_comparison_df["SNN_Improvement_Percent"] = [
    100.0
    * (metrics_classical["MAE"] - metrics_snn["MAE"])
    / metrics_classical["MAE"],
    100.0
    * (metrics_classical["RMSE"] - metrics_snn["RMSE"])
    / metrics_classical["RMSE"],
    100.0
    * (metrics_classical["Normalized MAE"] - metrics_snn["Normalized MAE"])
    / metrics_classical["Normalized MAE"],
    100.0
    * (metrics_classical["Normalized RMSE"] - metrics_snn["Normalized RMSE"])
    / metrics_classical["Normalized RMSE"],
    100.0
    * (metrics_snn["R2"] - metrics_classical["R2"])
    / abs(metrics_classical["R2"]),
]


# ============================================================
# 10. Save outputs as CSV files
# ============================================================
config_df.to_csv(
    "seasonal_trend_classical_vs_snn_configurations.csv",
    index=False,
)

prediction_comparison_df.to_csv(
    "seasonal_trend_classical_vs_snn_predictions.csv",
    index=False,
)

metric_comparison_df.to_csv(
    "seasonal_trend_classical_vs_snn_metrics.csv",
    index=False,
)


# ============================================================
# 11. Print final comparison tables
# ============================================================
print("\n--- Classical NN vs SNN Blind-Test Predictions in the Same Table ---")
print(prediction_comparison_df.to_string(index=False))

print("\n--- Classical NN vs SNN Performance Metrics in the Same Table ---")
print(metric_comparison_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

print("\nCSV files saved:")
print("1. seasonal_trend_classical_vs_snn_configurations.csv")
print("2. seasonal_trend_classical_vs_snn_predictions.csv")
print("3. seasonal_trend_classical_vs_snn_metrics.csv")
