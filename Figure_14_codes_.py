# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:39:39 2026

@author: Seda Karateke & Metin Zontul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Example 2: Clear MAE and RMSE decay plot
# Fractional Taylor remainder metrics
# Standalone code
# No PDF saving
# ============================================================


# ------------------------------------------------------------
# Numerical data from Example 2
# ------------------------------------------------------------

data = {
    "n": [5, 10, 20, 40, 80, 160],

    "MAE_R_classical": [
        7.826561e-01,
        3.559276e-01,
        1.155819e-01,
        3.134558e-02,
        8.004473e-03,
        2.011791e-03
    ],

    "MAE_R_symmetrized": [
        7.583284e-01,
        3.510170e-01,
        1.152179e-01,
        3.132077e-02,
        8.002127e-03,
        2.011519e-03
    ],

    "RMSE_R_classical": [
        9.117034e-01,
        3.893587e-01,
        1.268757e-01,
        3.480415e-02,
        8.907060e-03,
        2.238959e-03
    ],

    "RMSE_R_symmetrized": [
        8.719729e-01,
        3.722172e-01,
        1.244010e-01,
        3.460783e-02,
        8.894005e-03,
        2.238127e-03
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})


# ============================================================
# Two-panel figure: MAE and RMSE separately
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14.5, 5.6),
    sharey=True
)


# ------------------------------------------------------------
# Panel 1: MAE decay
# ------------------------------------------------------------

axes[0].plot(
    df["n"],
    df["MAE_R_classical"],
    color="#D62728",
    marker="o",
    linewidth=2.8,
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2.0,
    label=r"Classical MAE$_{\mathcal{R}}$"
)

axes[0].plot(
    df["n"],
    df["MAE_R_symmetrized"],
    color="#1F77B4",
    marker="s",
    linewidth=2.8,
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2.0,
    linestyle="--",
    label=r"Symmetrized MAE$_{\mathcal{R}}$"
)

axes[0].set_yscale("log")
axes[0].set_xlabel(r"$n$")
axes[0].set_ylabel(r"Remainder-based error")
axes[0].set_title(r"MAE decay")
axes[0].grid(True, which="both", alpha=0.35)
axes[0].legend(loc="upper right")


# ------------------------------------------------------------
# Panel 2: RMSE decay
# ------------------------------------------------------------

axes[1].plot(
    df["n"],
    df["RMSE_R_classical"],
    color="#FF7F0E",
    marker="D",
    linewidth=2.8,
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2.0,
    label=r"Classical RMSE$_{\mathcal{R}}$"
)

axes[1].plot(
    df["n"],
    df["RMSE_R_symmetrized"],
    color="#2CA02C",
    marker="^",
    linewidth=2.8,
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2.0,
    linestyle="--",
    label=r"Symmetrized RMSE$_{\mathcal{R}}$"
)

axes[1].set_yscale("log")
axes[1].set_xlabel(r"$n$")
axes[1].set_title(r"RMSE decay")
axes[1].grid(True, which="both", alpha=0.35)
axes[1].legend(loc="upper right")


# ------------------------------------------------------------
# General title and layout
# ------------------------------------------------------------

fig.suptitle(
    r"MAE and RMSE decay for fractional Taylor remainders",
    fontsize=17,
    y=1.03
)

plt.tight_layout()
plt.show()