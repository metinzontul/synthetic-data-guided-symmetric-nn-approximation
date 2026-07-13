# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:55:28 2026

@author: Seda Karateke & Metin Zontul
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.special import expit
except ImportError:
    expit = None

try:
    from numpy import trapezoid
except ImportError:
    from scipy.integrate import trapezoid


# ============================================================
# Numerical verification of the activation-symmetry mechanism
# Full corrected version
# ============================================================

output_dir = "activation_symmetry_results_final"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

t = 2.0
xi = 1.0
S = np.e

R = 8.0          # plotting interval [-R, R]
N = 4001        # number of grid points
M = 80          # truncation parameter for partition of unity

x = np.linspace(-R, R, N)


# ------------------------------------------------------------
# Stable activation functions
# ------------------------------------------------------------

def sigmoid_stable(z):
    """
    Stable logistic function.
    """
    if expit is not None:
        return expit(z)
    return 1.0 / (1.0 + np.exp(-z))


def k_activation(x, t_param=t, xi=xi, S=S):
    """
    k_{t,xi}(x) = 1 / (1 + t S^{-xi x})

    Stable equivalent form:
    k_{t,xi}(x) = expit(xi x log(S) - log(t)).
    """
    z = xi * x * np.log(S) - np.log(t_param)
    return sigmoid_stable(z)


def k_sym(x, t=t, xi=xi, S=S):
    """
    Symmetrized reciprocal-deformation activation:

    k^s_{t,xi}(x)
    =
    1/2 [ k_{t,xi}(x) + k_{1/t,xi}(x) ].
    """
    return 0.5 * (
        k_activation(x, t_param=t, xi=xi, S=S)
        +
        k_activation(x, t_param=1.0 / t, xi=xi, S=S)
    )


def aleph(x, t_param=t, xi=xi, S=S):
    """
    Density kernel induced by k_{t,xi}:

    aleph_{t,xi}(x)
    =
    1/2 [ k_{t,xi}(x+1) - k_{t,xi}(x-1) ].
    """
    return 0.5 * (
        k_activation(x + 1.0, t_param=t_param, xi=xi, S=S)
        -
        k_activation(x - 1.0, t_param=t_param, xi=xi, S=S)
    )


def F_density(x, t=t, xi=xi, S=S):
    """
    Symmetric activation-induced density kernel:

    F(x)
    =
    1/2 [ k^s_{t,xi}(x+1) - k^s_{t,xi}(x-1) ].

    Equivalently,

    F(x)
    =
    1/2 [ aleph_{t,xi}(x) + aleph_{1/t,xi}(x) ].
    """
    return 0.5 * (
        k_sym(x + 1.0, t=t, xi=xi, S=S)
        -
        k_sym(x - 1.0, t=t, xi=xi, S=S)
    )


# ------------------------------------------------------------
# Function values
# ------------------------------------------------------------

k_t = k_activation(x, t_param=t)
k_inv = k_activation(x, t_param=1.0 / t)
k_s = k_sym(x)

F_x = F_density(x)
F_minus_x = F_density(-x)


# ------------------------------------------------------------
# Numerical diagnostics
# ------------------------------------------------------------

# 1. Activation-level central symmetry:
# k^s(x) + k^s(-x) = 1

E_act_x = np.abs(k_s + k_sym(-x) - 1.0)
E_act = np.max(E_act_x)


# 2. Density-level evenness:
# F(x) = F(-x)

E_den_x = np.abs(F_x - F_minus_x)
E_den = np.max(E_den_x)


# 3. Integral normalization:
# int F(x) dx = 1

integral_F = trapezoid(F_x, x)
E_int = np.abs(integral_F - 1.0)


# 4. Partition of unity:
# sum_i F(x-i) = 1

x_pu = np.linspace(-1.0, 1.0, 1001)
i_vals = np.arange(-M, M + 1)

pu_values = np.array([
    np.sum(F_density(xx - i_vals))
    for xx in x_pu
])

pu_error = np.abs(pu_values - 1.0)
E_PU = np.max(pu_error)


# ------------------------------------------------------------
# Tail decay experiment
# ------------------------------------------------------------

varsigma = 0.5
n_values = np.array([10, 20, 40, 80, 160])
x0 = 0.37

W = 2.0 * max(t, 1.0 / t)

tail_values = []
bound_values = []

U = 300

for n in n_values:
    center = n * x0

    m_min = int(np.floor(center - U))
    m_max = int(np.ceil(center + U))
    m_vals = np.arange(m_min, m_max + 1)

    u_vals = center - m_vals

    mask = np.abs(u_vals) >= n ** (1.0 - varsigma)

    tail = np.sum(F_density(u_vals[mask]))
    bound = W * S ** (-xi * (n ** (1.0 - varsigma) - 2.0))

    tail_values.append(tail)
    bound_values.append(bound)

tail_values = np.array(tail_values)
bound_values = np.array(bound_values)


# ------------------------------------------------------------
# Save numerical results
# ------------------------------------------------------------

diagnostics = pd.DataFrame({
    "Metric": [
        "Activation symmetry error E_act",
        "Density symmetry error E_den",
        "Partition of unity error E_PU",
        "Integral of F over [-R,R]",
        "Integral normalization error E_int"
    ],
    "Value": [
        E_act,
        E_den,
        E_PU,
        integral_F,
        E_int
    ]
})

tail_df = pd.DataFrame({
    "n": n_values,
    "Numerical tail mass T_n": tail_values,
    "Theoretical bound B_n": bound_values,
    "Ratio T_n / B_n": tail_values / bound_values
})

diagnostics_path = os.path.join(
    output_dir,
    "diagnostics_activation_symmetry.csv"
)

tail_path = os.path.join(
    output_dir,
    "tail_decay_results.csv"
)

diagnostics.to_csv(diagnostics_path, index=False)
tail_df.to_csv(tail_path, index=False)

print("\nNumerical diagnostics")
print("---------------------")
print(diagnostics.to_string(index=False))

print("\nTail decay results")
print("------------------")
print(tail_df.to_string(index=False))

print("\nSaved files")
print("-----------")
print(diagnostics_path)
print(tail_path)


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11
})

markevery_main = 220


# ------------------------------------------------------------
# Figure 1: Activation functions
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    x,
    k_t,
    linewidth=2.6,
    linestyle="-",
    marker="o",
    markevery=markevery_main,
    markersize=4,
    label=r"$k_{t,\xi}(x)$"
)

ax.plot(
    x,
    k_inv,
    linewidth=2.6,
    linestyle="--",
    marker="s",
    markevery=markevery_main,
    markersize=4,
    label=r"$k_{1/t,\xi}(x)$"
)

ax.plot(
    x,
    k_s,
    linewidth=3.0,
    linestyle="-.",
    marker="^",
    markevery=markevery_main,
    markersize=4,
    label=r"$k^s_{t,\xi}(x)$"
)

ax.set_xlabel(r"$x$")
ax.set_ylabel("Activation value")
ax.set_title(r"Activation-level reciprocal-deformation symmetry")
ax.set_xlim(-6, 6)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, "Figure_1_activation_symmetry.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Figure 2: Central symmetry of k^s and error
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].plot(
    x,
    k_s,
    linewidth=2.8,
    label=r"$k^s_{t,\xi}(x)$"
)

axes[0].plot(
    x,
    1.0 - k_sym(-x),
    linewidth=2.4,
    linestyle="--",
    label=r"$1-k^s_{t,\xi}(-x)$"
)

axes[0].set_title(r"Central symmetry of $k^s_{t,\xi}$")
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel("Value")
axes[0].set_xlim(-6, 6)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", frameon=True)

axes[1].semilogy(
    x,
    E_act_x + 1e-18,
    linewidth=2.4
)

axes[1].set_title(r"Activation-symmetry error")
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$|k^s_{t,\xi}(x)+k^s_{t,\xi}(-x)-1|$")
axes[1].set_xlim(-6, 6)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, "Figure_2_activation_symmetry_error.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Figure 3: Density symmetry with zoom
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].plot(
    x,
    F_x,
    linewidth=2.8,
    label=r"$F(x)$"
)

axes[0].plot(
    x,
    F_minus_x,
    linewidth=2.2,
    linestyle="--",
    label=r"$F(-x)$"
)

axes[0].set_title(r"Even density kernel induced by $k^s_{t,\xi}$")
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel("Density value")
axes[0].set_xlim(-6, 6)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", frameon=True)

zoom_mask = (x >= -2.5) & (x <= 2.5)

axes[1].plot(
    x[zoom_mask],
    F_x[zoom_mask],
    linewidth=2.8,
    label=r"$F(x)$"
)

axes[1].plot(
    x[zoom_mask],
    F_minus_x[zoom_mask],
    linewidth=2.2,
    linestyle="--",
    label=r"$F(-x)$"
)

axes[1].set_title(r"Zoomed view")
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel("Density value")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, "Figure_3_density_symmetry.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Figure 4: Density symmetry error
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(
    x,
    E_den_x + 1e-18,
    linewidth=2.4
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$|F(x)-F(-x)|$")
ax.set_title(r"Numerical density-symmetry error")
ax.set_xlim(-6, 6)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, "Figure_4_density_symmetry_error.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Figure 5: Partition of unity and deviation from 1
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].plot(
    x_pu,
    pu_values,
    linewidth=2.6,
    label=r"$\sum_{i=-M}^{M}F(x-i)$"
)

axes[0].axhline(
    1.0,
    linestyle="--",
    linewidth=2.0,
    label=r"$1$"
)

axes[0].set_title(r"Numerical partition of unity")
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$\sum_{i=-M}^{M}F(x-i)$")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", frameon=True)

axes[1].semilogy(
    x_pu,
    pu_error + 1e-18,
    linewidth=2.4
)

axes[1].set_title(r"Deviation from unity")
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$|\sum_{i=-M}^{M}F(x-i)-1|$")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, "Figure_5_partition_unity.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Figure 6: Tail decay
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(
    n_values,
    tail_values,
    linewidth=2.6,
    marker="o",
    markersize=7,
    label=r"Numerical tail mass $T_n$"
)

ax.semilogy(
    n_values,
    bound_values,
    linewidth=2.6,
    linestyle="--",
    marker="s",
    markersize=7,
    label=r"Theoretical bound $B_n$"
)

ax.set_xlabel(r"$n$")
ax.set_ylabel("Tail mass")
ax.set_title(r"Tail decay of the symmetric activation-induced density kernel")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, "Figure_6_tail_decay.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

