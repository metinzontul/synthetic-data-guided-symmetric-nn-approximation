# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:29:30 2026

@author: sedak
"""

# -*- coding: utf-8 -*-

"""
Vector-valued approximation in Y = R^2.

Figure 19:
Image of the exact vector-valued function and the classical NN
approximants.

Figure 20:
Image of the exact vector-valued function and the SNN
approximants.

The two panels use identical coordinate limits.
The legends are placed outside the plotting regions so that
they do not overlap the images of the approximants.

@author: sedak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. NUMERICAL PARAMETERS
# ============================================================

t = 2.0
xi = 1.0
S = np.e

N = 4001
x_grid = np.linspace(0.0, 1.0, N)

n_values = [5, 10, 20, 40, 80, 160]


# ============================================================
# 2. DEFORMATION-DEPENDENT ACTIVATION FUNCTIONS
# ============================================================

def k_activation(x, t, xi, S=np.e):
    """
    Deformation-dependent sigmoidal activation

        k_{t,xi}(x)
        = 1 / (1 + t S^{-xi x}).
    """

    return 1.0 / (
        1.0 + t * S ** (-xi * x)
    )


def k_sym(x, t, xi, S=np.e):
    """
    Activation used in the SNN construction

        k^s_{t,xi}(x)
        = 1/2 [
            k_{t,xi}(x)
            + k_{1/t,xi}(x)
          ].
    """

    return 0.5 * (
        k_activation(x, t, xi, S)
        +
        k_activation(x, 1.0 / t, xi, S)
    )


# ============================================================
# 3. CLASSICAL NN AND SNN DENSITY KERNELS
# ============================================================

def aleph_density(x, t, xi, S=np.e):
    """
    Classical NN density kernel

        aleph_{t,xi}(x)
        = 1/2 [
            k_{t,xi}(x+1)
            - k_{t,xi}(x-1)
          ].
    """

    return 0.5 * (
        k_activation(x + 1.0, t, xi, S)
        -
        k_activation(x - 1.0, t, xi, S)
    )


def F_density(x, t, xi, S=np.e):
    """
    SNN density kernel

        F(x)
        = 1/2 [
            k^s_{t,xi}(x+1)
            - k^s_{t,xi}(x-1)
          ].
    """

    return 0.5 * (
        k_sym(x + 1.0, t, xi, S)
        -
        k_sym(x - 1.0, t, xi, S)
    )


# ============================================================
# 4. Y-VALUED TARGET FUNCTION
#
#    f : [0,1] -> R^2
#    Y = R^2 endowed with the Euclidean norm
# ============================================================

def f_vector(x):
    """
    R^2-valued target function

              [ cos(2*pi*x) + x(1-x) ]
        f(x)= [                         ].
              [ sin(2*pi*x) + x^2      ]
    """

    x = np.asarray(x)

    values = np.empty(
        x.shape + (2,),
        dtype=float
    )

    values[..., 0] = (
        np.cos(2.0 * np.pi * x)
        + x * (1.0 - x)
    )

    values[..., 1] = (
        np.sin(2.0 * np.pi * x)
        + x**2
    )

    return values


# ============================================================
# 5. Y-VALUED COMPACT-INTERVAL APPROXIMATION OPERATOR
# ============================================================

def vector_operator(
    x_eval,
    n,
    density_function,
    t,
    xi,
    S=np.e
):
    """
    Computes the Y-valued normalized approximation operator

                    sum f(m/n) K(nx-m)
        L_n(f,x) = ---------------------
                       sum K(nx-m),

    where f(m/n) belongs to Y = R^2.

    The numerator is a finite scalar-weighted linear
    combination of elements of Y.
    """

    x_eval = np.asarray(x_eval)

    # Sampling indices
    m_values = np.arange(
        0,
        n + 1,
        dtype=float
    )

    # Sampling nodes m/n
    sampling_nodes = m_values / n

    # Y-valued samples
    #
    # Each row is one element of R^2.
    # Shape: (n+1, 2)
    f_nodes = f_vector(
        sampling_nodes
    )

    # Kernel arguments
    #
    # Shape:
    # (number of evaluation points, n+1)
    kernel_arguments = (
        n * x_eval[:, None]
        - m_values[None, :]
    )

    # Scalar kernel coefficients
    kernel_values = density_function(
        kernel_arguments,
        t,
        xi,
        S
    )

    # Normalization denominator
    denominator = np.sum(
        kernel_values,
        axis=1,
        keepdims=True
    )

    # Finite scalar-weighted linear combination
    # of Y-valued samples
    numerator = (
        kernel_values @ f_nodes
    )

    # Y-valued approximant
    return (
        numerator / denominator
    )


# ============================================================
# 6. UNIFORM APPROXIMATION ERROR IN C([0,1], R^2)
# ============================================================

def uniform_R2_error(
    approximation,
    exact_values
):
    """
    Computes

        ||L_n(f) - f||_{infinity,2}
        =
        max_{x in [0,1]}
        ||L_n(f,x) - f(x)||_2.
    """

    # Error vectors in Y = R^2
    error_vectors = (
        approximation
        - exact_values
    )

    # Euclidean norm at each evaluation point
    pointwise_norm_errors = np.linalg.norm(
        error_vectors,
        ord=2,
        axis=1
    )

    # Uniform approximation error
    return np.max(
        pointwise_norm_errors
    )


# ============================================================
# 7. EXACT Y-VALUED FUNCTION
# ============================================================

true_values = f_vector(
    x_grid
)


# ============================================================
# 8. CLASSICAL NN AND SNN APPROXIMANTS
# ============================================================

classical_results = {}
snn_results = {}

table_rows = []


for n in n_values:

    # --------------------------------------------------------
    # Classical NN approximant
    # --------------------------------------------------------

    classical_approx = vector_operator(
        x_eval=x_grid,
        n=n,
        density_function=aleph_density,
        t=t,
        xi=xi,
        S=S
    )

    # --------------------------------------------------------
    # SNN approximant
    # --------------------------------------------------------

    snn_approx = vector_operator(
        x_eval=x_grid,
        n=n,
        density_function=F_density,
        t=t,
        xi=xi,
        S=S
    )

    classical_results[n] = (
        classical_approx
    )

    snn_results[n] = (
        snn_approx
    )

    # --------------------------------------------------------
    # Uniform errors in C([0,1], R^2)
    # --------------------------------------------------------

    E_classical = uniform_R2_error(
        classical_approx,
        true_values
    )

    E_snn = uniform_R2_error(
        snn_approx,
        true_values
    )

    # --------------------------------------------------------
    # Relative reduction
    # --------------------------------------------------------

    I_V_n = (
        100.0
        * (E_classical - E_snn)
        / E_classical
    )

    table_rows.append({
        "n": n,
        "E_classical": E_classical,
        "E_SNN": E_snn,
        "I_V_n_percent": I_V_n
    })


# ============================================================
# 9. TABLE 10 CONSISTENCY CHECK
# ============================================================

df = pd.DataFrame(
    table_rows
)

print(
    "\nTable 10 consistency check:\n"
)

print(
    df.to_string(
        index=False,
        formatters={
            "E_classical":
                lambda value: f"{value:.6f}",

            "E_SNN":
                lambda value: f"{value:.6f}",

            "I_V_n_percent":
                lambda value: f"{value:.2f}"
        }
    )
)


# ============================================================
# 10. COMMON AXIS LIMITS
#
#     Both panels use exactly the same coordinate limits.
# ============================================================

all_images = [
    true_values
]

for n in n_values:

    all_images.append(
        classical_results[n]
    )

    all_images.append(
        snn_results[n]
    )


all_points = np.vstack(
    all_images
)


# f1-coordinate limits
f1_min = np.min(
    all_points[:, 0]
)

f1_max = np.max(
    all_points[:, 0]
)


# f2-coordinate limits
f2_min = np.min(
    all_points[:, 1]
)

f2_max = np.max(
    all_points[:, 1]
)


# Add small margins around the images
f1_margin = (
    0.05
    * (f1_max - f1_min)
)

f2_margin = (
    0.05
    * (f2_max - f2_min)
)


common_xlim = (
    f1_min - f1_margin,
    f1_max + f1_margin
)

common_ylim = (
    f2_min - f2_margin,
    f2_max + f2_margin
)


# ============================================================
# 11. FIGURES 19 AND 20
#     SIDE-BY-SIDE REPRESENTATION
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(15, 7.8)
)

ax1, ax2 = axes


# ============================================================
# FIGURE 19
# CLASSICAL NN APPROXIMANTS
# ============================================================

ax1.plot(
    true_values[:, 0],
    true_values[:, 1],
    linewidth=3.0,
    color="black",
    label=r"Exact image $\mathbf{f}([0,1])$"
)


for n in n_values:

    classical_approx = (
        classical_results[n]
    )

    ax1.plot(
        classical_approx[:, 0],
        classical_approx[:, 1],
        linewidth=1.5,
        label=rf"$n={n}$"
    )


ax1.set_xlabel(
    r"$f_1$-coordinate",
    fontsize=12
)

ax1.set_ylabel(
    r"$f_2$-coordinate",
    fontsize=12
)

ax1.set_title(
    r"(a) Classical NN approximants in "
    r"$Y=\mathbb{R}^2$",
    fontsize=12
)


ax1.set_xlim(
    common_xlim
)

ax1.set_ylim(
    common_ylim
)


ax1.set_aspect(
    "equal",
    adjustable="box"
)


ax1.grid(
    True,
    alpha=0.30
)


# ------------------------------------------------------------
# Legend is outside the plotting region,
# below the right-hand side of Figure 19.
# ------------------------------------------------------------

ax1.legend(
    loc="upper right",
    bbox_to_anchor=(
        1.0,
        -0.13
    ),
    ncol=2,
    fontsize=8.5,
    frameon=True,
    borderaxespad=0.0
)


# ============================================================
# FIGURE 20
# SNN APPROXIMANTS
# ============================================================

ax2.plot(
    true_values[:, 0],
    true_values[:, 1],
    linewidth=3.0,
    color="black",
    label=r"Exact image $\mathbf{f}([0,1])$"
)


for n in n_values:

    snn_approx = (
        snn_results[n]
    )

    ax2.plot(
        snn_approx[:, 0],
        snn_approx[:, 1],
        linewidth=1.5,
        label=rf"$n={n}$"
    )


ax2.set_xlabel(
    r"$f_1$-coordinate",
    fontsize=12
)

ax2.set_ylabel(
    r"$f_2$-coordinate",
    fontsize=12
)

ax2.set_title(
    r"(b) SNN approximants in "
    r"$Y=\mathbb{R}^2$",
    fontsize=12
)


ax2.set_xlim(
    common_xlim
)

ax2.set_ylim(
    common_ylim
)


ax2.set_aspect(
    "equal",
    adjustable="box"
)


ax2.grid(
    True,
    alpha=0.30
)


# ------------------------------------------------------------
# Legend is outside the plotting region,
# below the right-hand side of Figure 20.
# ------------------------------------------------------------

ax2.legend(
    loc="upper right",
    bbox_to_anchor=(
        1.0,
        -0.13
    ),
    ncol=2,
    fontsize=8.5,
    frameon=True,
    borderaxespad=0.0
)


# ============================================================
# 12. LAYOUT
#
#     Extra space is reserved below the two panels for the
#     legends, preventing any overlap with the approximants.
# ============================================================

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.91,
    bottom=0.27,
    wspace=0.25
)


# ============================================================
# 13. SAVE FIGURES 19 AND 20
# ============================================================

output_file = (
    "Figures19_20_Vector_Valued_Approximation_R2.png"
)


plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print(
    "\nFigures 19 and 20 saved as:"
)

print(
    output_file
)