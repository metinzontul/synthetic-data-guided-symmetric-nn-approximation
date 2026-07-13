# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:54:48 2026

@author: Seda Karateke & Metin Zontul
"""
"""
Figures for Example 3:
Figure 16: Blind-test prediction comparison
Figure 17: Absolute blind-test error comparison

This script does not automatically save PNG files.
It only displays the figures.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Blind-test data from Example 3
# ============================================================

months = np.arange(49, 61)

actual_values = np.array([
    9.31, 9.86, 18.04, 22.02, 25.29, 28.62,
    27.96, 24.20, 16.59, 12.44, 9.79, 9.46
])

classical_nn_predictions = np.array([
    9.53, 13.24, 17.88, 22.22, 25.10, 25.77,
    24.05, 20.41, 15.84, 11.60, 8.76, 7.51
])

snn_predictions = np.array([
    8.72, 11.90, 16.37, 20.94, 24.40, 25.82,
    24.84, 21.73, 17.33, 12.84, 9.48, 7.78
])


# ============================================================
# 2. Absolute errors
# ============================================================

classical_nn_errors = np.abs(actual_values - classical_nn_predictions)
snn_errors = np.abs(actual_values - snn_predictions)


# ============================================================
# 3. Figure 16: Blind-test prediction comparison
# ============================================================

plt.figure(figsize=(8.8, 5.2))

plt.plot(
    months,
    actual_values,
    marker="o",
    linewidth=2.2,
    label="Actual value"
)

plt.plot(
    months,
    classical_nn_predictions,
    marker="s",
    linewidth=2.0,
    linestyle="--",
    label="Classical NN prediction"
)

plt.plot(
    months,
    snn_predictions,
    marker="^",
    linewidth=2.0,
    linestyle="-.",
    label="SNN prediction"
)

plt.xlabel("Month index", fontsize=12)
plt.ylabel("Temperature-like value", fontsize=12)
plt.title("Blind-test prediction comparison", fontsize=13)

plt.xticks(months)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()


# ============================================================
# 4. Figure 17: Absolute blind-test error comparison
# ============================================================

plt.figure(figsize=(8.8, 5.2))

plt.plot(
    months,
    classical_nn_errors,
    marker="s",
    linewidth=2.0,
    linestyle="--",
    label="Classical NN absolute error"
)

plt.plot(
    months,
    snn_errors,
    marker="^",
    linewidth=2.0,
    linestyle="-.",
    label="SNN absolute error"
)

plt.xlabel("Month index", fontsize=12)
plt.ylabel("Absolute error", fontsize=12)
plt.title("Absolute blind-test error comparison", fontsize=13)

plt.xticks(months)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()