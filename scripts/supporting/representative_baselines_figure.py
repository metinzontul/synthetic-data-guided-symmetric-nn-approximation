# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:01:07 2026

@author: Metin Zontul
"""

# -*- coding: utf-8 -*-

# ============================================================
# FIGURE 7
# Blind-Test Prediction Comparison (Months 97--120)
#
# Final parameter-matched configuration:
# SNN          : n=64, t=0.5, xi=2.0
# Classical NN : n=64, t=0.5, xi=2.0
#
# Seed = 42
# Months 1--96   : final model fitting
# Months 97--120 : untouched blind test
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

SEED = 42

MONTHS_TOTAL = 120
TRAIN_SIZE = 96
TEST_SIZE = 24

MONTHS = np.arange(1, MONTHS_TOTAL + 1)

# Map months 1,...,120 to [-1,1]
X_NODES = (
    -1.0
    + 2.0 * (MONTHS - 1)
    / (MONTHS_TOTAL - 1)
)

X_TRAIN = X_NODES[:TRAIN_SIZE]
X_TEST = X_NODES[TRAIN_SIZE:]


# ============================================================
# 2. SYNTHETIC DATA GENERATION
# ============================================================

rng = np.random.RandomState(SEED)

T_DATA = (
    15.0
    + 10.0
    * np.sin(
        2.0 * np.pi * MONTHS / 12.0
        - np.pi / 2.0
    )
    + 0.05 * MONTHS
    + rng.normal(
        loc=0.0,
        scale=1.2,
        size=MONTHS_TOTAL
    )
)

TRAIN_DATA = T_DATA[:TRAIN_SIZE]
TEST_DATA = T_DATA[TRAIN_SIZE:]


# ============================================================
# 3. CONTINUOUS SEASONAL-TREND FUNCTION
# ============================================================

def x_to_month(x):
    return (
        ((x + 1.0) / 2.0)
        * (MONTHS_TOTAL - 1)
        + 1.0
    )


def seasonal_trend_func(
    x,
    A,
    omega,
    phi,
    B,
    gamma
):
    m = x_to_month(x)

    return (
        A * np.sin(omega * m + phi)
        + B
        + gamma * m
    )


# ============================================================
# 4. ACTIVATION FUNCTIONS
# ============================================================

def k_activation(x, t, xi):

    z = np.clip(
        -xi * x,
        -np.inf,
        200.0
    )

    return (
        1.0
        / (1.0 + t * np.exp(z))
    )


def k_sym(x, t, xi):

    return 0.5 * (
        k_activation(
            x,
            t,
            xi
        )
        +
        k_activation(
            x,
            1.0 / t,
            xi
        )
    )


# ============================================================
# 5. DENSITY KERNELS
# ============================================================

def aleph_density(x, t, xi):

    return 0.5 * (
        k_activation(
            x + 1.0,
            t,
            xi
        )
        -
        k_activation(
            x - 1.0,
            t,
            xi
        )
    )


def F_density(x, t, xi):

    return 0.5 * (
        k_sym(
            x + 1.0,
            t,
            xi
        )
        -
        k_sym(
            x - 1.0,
            t,
            xi
        )
    )


# ============================================================
# 6. GENERALIZED OPERATOR
# ============================================================

def generalized_operator(
    x_eval,
    n,
    density_func,
    t,
    xi,
    popt
):

    a_n = float(n)

    b_n = np.sqrt(
        n**2 + 2.0
    )

    m_vals = np.arange(
        int(np.ceil(-b_n)),
        int(np.floor(b_n)) + 1
    )

    f_nodes = seasonal_trend_func(
        m_vals / b_n,
        *popt
    )

    z = (
        a_n
        * np.asarray(x_eval)[:, None]
        - m_vals[None, :]
    )

    weights = density_func(
        z,
        t=t,
        xi=xi
    )

    denominator = np.sum(
        weights,
        axis=1
    )

    if np.any(denominator <= 0.0):
        raise FloatingPointError(
            "Non-positive normalization denominator."
        )

    predictions = (
        weights @ f_nodes
    ) / denominator

    return predictions


# ============================================================
# 7. FINAL PARAMETER-MATCHED CONFIGURATION
#
# These values have already been selected by the
# leakage-free validation/grid-search procedure.
# ============================================================

OPT_N = 64
OPT_T = 0.5
OPT_XI = 2.0


# ============================================================
# 8. REFIT CONTINUOUS TREND ON MONTHS 1--96
# ============================================================

p0 = [
    10.0,
    2.0 * np.pi / 12.0,
    -np.pi / 2.0,
    15.0,
    0.05
]

popt_final, _ = curve_fit(
    seasonal_trend_func,
    X_TRAIN,
    TRAIN_DATA,
    p0=p0,
    maxfev=20000
)


# ============================================================
# 9. SNN BLIND-TEST PREDICTIONS
# ============================================================

snn_preds = generalized_operator(
    X_TEST,
    OPT_N,
    F_density,
    OPT_T,
    OPT_XI,
    popt_final
)


# ============================================================
# 10. CLASSICAL NN BLIND-TEST PREDICTIONS
# ============================================================

cnn_preds = generalized_operator(
    X_TEST,
    OPT_N,
    aleph_density,
    OPT_T,
    OPT_XI,
    popt_final
)


# ============================================================
# 11. MLP BASELINE
# ============================================================

X_train_mlp = np.column_stack(
    (
        MONTHS[:TRAIN_SIZE],

        np.sin(
            2.0
            * np.pi
            * MONTHS[:TRAIN_SIZE]
            / 12.0
        ),

        np.cos(
            2.0
            * np.pi
            * MONTHS[:TRAIN_SIZE]
            / 12.0
        )
    )
)

X_test_mlp = np.column_stack(
    (
        MONTHS[TRAIN_SIZE:],

        np.sin(
            2.0
            * np.pi
            * MONTHS[TRAIN_SIZE:]
            / 12.0
        ),

        np.cos(
            2.0
            * np.pi
            * MONTHS[TRAIN_SIZE:]
            / 12.0
        )
    )
)

scaler = StandardScaler()

X_train_mlp_scaled = scaler.fit_transform(
    X_train_mlp
)

X_test_mlp_scaled = scaler.transform(
    X_test_mlp
)

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    solver="lbfgs",
    max_iter=2000,
    random_state=SEED
)

mlp.fit(
    X_train_mlp_scaled,
    TRAIN_DATA
)

mlp_preds = mlp.predict(
    X_test_mlp_scaled
)


# ============================================================
# 12. SEASONAL ARIMA BASELINE
# ============================================================

arima_model = ARIMA(
    TRAIN_DATA,
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 12)
)

arima_fit = arima_model.fit()

arima_preds = arima_fit.forecast(
    steps=TEST_SIZE
)


# ============================================================
# 13. FIGURE 7
# ============================================================

blind_months = MONTHS[TRAIN_SIZE:]

fig, ax = plt.subplots(
    figsize=(14, 5.5)
)


# Actual data
ax.plot(
    blind_months,
    TEST_DATA,
    "ko-",
    linewidth=2.5,
    markersize=6,
    label="Actual Data"
)


# SNN
ax.plot(
    blind_months,
    snn_preds,
    "r^-",
    linewidth=2.0,
    markersize=8,
    label=r"SNN ($n=64,\ t=0.5,\ \xi=2.0$)"
)


# Classical NN
ax.plot(
    blind_months,
    cnn_preds,
    "ms--",
    linewidth=2.0,
    markersize=6,
    alpha=0.75,
    label=r"Classical NN ($n=64,\ t=0.5,\ \xi=2.0$)"
)


# Seasonal ARIMA
ax.plot(
    blind_months,
    arima_preds,
    "b--",
    linewidth=2.0,
    label=r"Seasonal ARIMA $(1,0,1)(0,1,1)_{12}$"
)


# MLP
ax.plot(
    blind_months,
    mlp_preds,
    "g-.",
    linewidth=2.0,
    label="MLP (100, 50)"
)


# ============================================================
# 14. GRAPH FORMATTING
# ============================================================

ax.set_title(
    "Blind-Test Prediction Comparison (Months 97--120)",
    fontsize=15
)

ax.set_xlabel(
    "Month Index",
    fontsize=12
)

ax.set_ylabel(
    "Temperature-like Value",
    fontsize=12
)

ax.set_xticks(
    np.arange(
        97,
        121,
        2
    )
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.5
)


# ============================================================
# 15. LEGEND OUTSIDE GRAPH
#
# This prevents the legend from covering any prediction curves.
# ============================================================

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    fontsize=10,
    frameon=True,
    borderaxespad=0.0
)

# Leave sufficient space on the right
# for the external legend
fig.subplots_adjust(
    right=0.72,
    left=0.08,
    bottom=0.15,
    top=0.90
)


# ============================================================
# 16. SAVE 600 DPI
# ============================================================

FIGURE_NAME = (
    "Figure_Example3_Baseline_Predictions_600dpi.png"
)

plt.savefig(
    FIGURE_NAME,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


print("\nFigure 7 successfully generated.")
print(f"Saved as: {FIGURE_NAME}")
print("Parameter-matched configuration:")
print("SNN          : n=64, t=0.5, xi=2.0")
print("Classical NN : n=64, t=0.5, xi=2.0")