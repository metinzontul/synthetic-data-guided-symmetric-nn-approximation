# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:03:57 2026

@author: Seda Karateke & Metin Zontul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Example 1: Half-order fractional approximation
# All approximations for n = 5, 10, 20, 40, 80, 160
# Classical NN operator versus symmetrized SNN operator
# ============================================================

# Parameters
a, b = 0.0, 1.0
t = 2.0
xi = 1.0
S = np.e

alpha = 0.5
tau = 0.5

n_values = [5, 10, 20, 40, 80, 160]

# Evaluation grid
N_grid = 4001
x_grid = np.linspace(a, b, N_grid)


# ------------------------------------------------------------
# Test function
# ------------------------------------------------------------

def f1(x):
    """
    One-dimensional symmetric test function on [0,1].
    """
    return np.cos(2.0 * np.pi * x) + x * (1.0 - x)


# ------------------------------------------------------------
# Activation and density kernels
# ------------------------------------------------------------

def k_activation(x, t=t, xi=xi):
    """
    Deformation-dependent sigmoidal activation:
        k_{t,xi}(x) = 1 / (1 + t exp(-xi x)).
    """
    return 1.0 / (1.0 + t * np.exp(-xi * x))


def aleph_density(x, t=t, xi=xi):
    """
    Classical density kernel:
        aleph_{t,xi}(x)
        =
        1/2 [k_{t,xi}(x+1) - k_{t,xi}(x-1)].
    """
    return 0.5 * (
        k_activation(x + 1.0, t=t, xi=xi)
        -
        k_activation(x - 1.0, t=t, xi=xi)
    )


def F_sym_density(x, t=t, xi=xi):
    """
    Symmetrized density kernel:
        F(x)
        =
        1/2 [aleph_{t,xi}(x) + aleph_{1/t,xi}(x)].
    """
    return 0.5 * (
        aleph_density(x, t=t, xi=xi)
        +
        aleph_density(x, t=1.0 / t, xi=xi)
    )


# ------------------------------------------------------------
# Compact-interval normalized operators
# ------------------------------------------------------------

def compact_operator(f, x_eval, n, density_function):
    """
    Computes the compact-interval normalized operator

        sum f(m/n) K(nx-m)
        ------------------
        sum K(nx-m)

    on [a,b]=[0,1].
    """
    m_min = int(np.ceil(n * a))
    m_max = int(np.floor(n * b))

    m = np.arange(m_min, m_max + 1)
    nodes = m / n
    f_nodes = f(nodes)

    z = n * x_eval[:, None] - m[None, :]
    weights = density_function(z)

    numerator = weights @ f_nodes
    denominator = np.sum(weights, axis=1)

    return numerator / denominator


def classical_operator(f, x_eval, n):
    return compact_operator(f, x_eval, n, aleph_density)


def symmetrized_operator(f, x_eval, n):
    return compact_operator(f, x_eval, n, F_sym_density)


# ------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------

def uniform_error(y_pred, y_true):
    return np.max(np.abs(y_pred - y_true))


def mae_error(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def rmse_error(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def r2_score(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# ------------------------------------------------------------
# Compute approximations and metrics
# ------------------------------------------------------------

y_true = f1(x_grid)

classical_approximations = {}
symmetrized_approximations = {}

rows = []

for n in n_values:
    y_classical = classical_operator(f1, x_grid, n)
    y_sym = symmetrized_operator(f1, x_grid, n)

    classical_approximations[n] = y_classical
    symmetrized_approximations[n] = y_sym

    E_inf_c = uniform_error(y_classical, y_true)
    E_inf_s = uniform_error(y_sym, y_true)

    MAE_c = mae_error(y_classical, y_true)
    MAE_s = mae_error(y_sym, y_true)

    RMSE_c = rmse_error(y_classical, y_true)
    RMSE_s = rmse_error(y_sym, y_true)

    R2_c = r2_score(y_classical, y_true)
    R2_s = r2_score(y_sym, y_true)

    I_inf = 100.0 * (E_inf_c - E_inf_s) / E_inf_c

    MAE_ratio = MAE_s / MAE_c
    RMSE_ratio = RMSE_s / RMSE_c

    rows.append({
        "n": n,
        "E_inf_classical": E_inf_c,
        "E_inf_symmetrized": E_inf_s,
        "Improvement_percent": I_inf,
        "MAE_classical": MAE_c,
        "MAE_symmetrized": MAE_s,
        "MAE_ratio": MAE_ratio,
        "RMSE_classical": RMSE_c,
        "RMSE_symmetrized": RMSE_s,
        "RMSE_ratio": RMSE_ratio,
        "R2_classical": R2_c,
        "R2_symmetrized": R2_s
    })

df = pd.DataFrame(rows)


# ------------------------------------------------------------
# Table 1: Complete numerical metrics
# ------------------------------------------------------------

pd.set_option("display.precision", 8)

print("\nTable 1. Numerical metrics for Example 1")
print(df.to_string(index=False))


# ------------------------------------------------------------
# Table 2: Manuscript-style compact table
# ------------------------------------------------------------

df_compact = df[[
    "n",
    "E_inf_classical",
    "E_inf_symmetrized",
    "Improvement_percent",
    "MAE_classical",
    "MAE_symmetrized",
    "RMSE_classical",
    "RMSE_symmetrized",
    "R2_classical",
    "R2_symmetrized"
]]

print("\nTable 2. Compact table for manuscript")
print(df_compact.to_string(index=False))


# ------------------------------------------------------------
# LaTeX table rows: uniform error comparison
# ------------------------------------------------------------

print("\nLaTeX table rows: uniform-error comparison")
for _, row in df.iterrows():
    print(
        f"{int(row['n'])} & "
        f"${row['E_inf_classical']:.6e}$ & "
        f"${row['E_inf_symmetrized']:.6e}$ & "
        f"${row['Improvement_percent']:.2f}$ \\\\"
    )


# ------------------------------------------------------------
# LaTeX table rows: extended metrics
# ------------------------------------------------------------

print("\nLaTeX table rows: extended metrics")
for _, row in df.iterrows():
    print(
        f"{int(row['n'])} & "
        f"${row['MAE_classical']:.6e}$ & "
        f"${row['MAE_symmetrized']:.6e}$ & "
        f"${row['RMSE_classical']:.6e}$ & "
        f"${row['RMSE_symmetrized']:.6e}$ & "
        f"${row['R2_classical']:.6f}$ & "
        f"${row['R2_symmetrized']:.6f}$ \\\\"
    )


# ============================================================
# Figure 1: All approximations on the same graph
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x_grid,
    y_true,
    linewidth=3.0,
    label=r"$f_1(x)$"
)

for n in n_values:
    plt.plot(
        x_grid,
        classical_approximations[n],
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=rf"$L_{{{n}}}(f_1,x)$"
    )

for n in n_values:
    plt.plot(
        x_grid,
        symmetrized_approximations[n],
        linestyle=":",
        linewidth=1.8,
        alpha=0.85,
        label=rf"$L_{{{n}}}^s(f_1,x)$"
    )

plt.xlabel(r"$x$")
plt.ylabel("Approximation value")
plt.title(
    r"Approximations of $f_1(x)=\cos(2\pi x)+x(1-x)$ "
    r"for $n=5,10,20,40,80,160$"
)
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 2: Uniform error decay
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["n"],
    df["E_inf_classical"],
    marker="o",
    label=r"$E_{\infty,n}^{c}(f_1)$"
)

plt.plot(
    df["n"],
    df["E_inf_symmetrized"],
    marker="s",
    label=r"$E_{\infty,n}^{s}(f_1)$"
)

plt.yscale("log")
plt.xlabel(r"$n$")
plt.ylabel("Uniform error")
plt.title("Log-scale decay of uniform approximation errors")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()


# ============================================================
# Figure 3: MAE and RMSE decay
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["n"],
    df["MAE_classical"],
    marker="o",
    linestyle="--",
    label="Classical MAE"
)

plt.plot(
    df["n"],
    df["MAE_symmetrized"],
    marker="s",
    linestyle="--",
    label="Symmetrized MAE"
)

plt.plot(
    df["n"],
    df["RMSE_classical"],
    marker="o",
    linestyle=":",
    label="Classical RMSE"
)

plt.plot(
    df["n"],
    df["RMSE_symmetrized"],
    marker="s",
    linestyle=":",
    label="Symmetrized RMSE"
)

plt.yscale("log")
plt.xlabel(r"$n$")
plt.ylabel("Error")
plt.title("Log-scale decay of MAE and RMSE")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()


# ============================================================
# Figure 4: Relative improvement
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["n"],
    df["Improvement_percent"],
    marker="o",
    label=r"$I_{\infty,n}$"
)

plt.xlabel(r"$n$")
plt.ylabel("Improvement percentage")
plt.title("Relative improvement of the symmetrized SNN operator")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()