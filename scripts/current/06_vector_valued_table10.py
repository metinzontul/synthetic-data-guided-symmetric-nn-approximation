# -*- coding: utf-8 -*-

"""
Vector-Valued Approximation in Y = R^2

Numerical evaluation of the classical NN and symmetrized SNN
operators for a single R^2-valued target function.

The approximation error is evaluated in the Euclidean norm of Y = R^2,
and the corresponding uniform error is computed over [0,1].

Created on Wed Aug 12 18:32:06 2026

@author: sedak
"""

import numpy as np
import pandas as pd


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
    Reciprocal-deformation symmetrized activation

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
# 3. CLASSICAL AND SYMMETRIZED DENSITY KERNELS
# ============================================================

def aleph_density(x, t, xi, S=np.e):
    """
    Classical activation-induced density kernel

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
    Symmetrized activation-induced density kernel

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
    Single R^2-valued target function

              [ cos(2*pi*x) + x(1-x) ]
        f(x)= [                         ].
              [ sin(2*pi*x) + x^2      ]

    The returned array represents elements of Y = R^2.
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
    Computes the normalized compact-interval approximation

        L_n(f,x)
        =
        sum_{m=0}^n f(m/n) K(nx-m)
        --------------------------------
        sum_{m=0}^n K(nx-m),

    where

        f(m/n) belongs to Y = R^2,

    and K is either

        aleph_{t,xi}   (classical NN)

    or

        F              (symmetrized SNN).

    Each numerator is therefore a finite scalar-weighted
    linear combination of vectors in Y.
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

    # Y-valued samples:
    #
    # f_nodes[m] is one element of R^2.
    #
    # Shape:
    # (n+1, 2)
    f_nodes = f_vector(sampling_nodes)

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

    # Finite normalization denominator
    denominator = np.sum(
        kernel_values,
        axis=1,
        keepdims=True
    )

    # Scalar-weighted linear combination
    # of Y-valued samples
    #
    # Shape:
    # (number of evaluation points, 2)
    numerator = kernel_values @ f_nodes

    # Y-valued approximant
    return numerator / denominator


# ============================================================
# 6. BANACH-SPACE UNIFORM ERROR
# ============================================================

def uniform_R2_error(approximation, exact_values):
    """
    Computes

        max_{x in [0,1]}
        || approximation(x) - f(x) ||_2.

    The pointwise error is therefore measured directly
    in the norm of Y = R^2.
    """

    # Error vectors in Y = R^2
    error_vectors = (
        approximation
        - exact_values
    )

    # Euclidean norm in R^2 at every evaluation point
    pointwise_norm_errors = np.linalg.norm(
        error_vectors,
        ord=2,
        axis=1
    )

    # Uniform norm over the computational grid
    return np.max(pointwise_norm_errors)


# ============================================================
# 7. EXACT Y-VALUED FUNCTION ON THE COMPUTATIONAL GRID
# ============================================================

true_values = f_vector(x_grid)


# ============================================================
# 8. CLASSICAL AND SYMMETRIZED APPROXIMATION ERRORS
# ============================================================

results = []

for n in n_values:

    # --------------------------------------------------------
    # Classical NN approximant
    # --------------------------------------------------------

    classical_approximation = vector_operator(
        x_eval=x_grid,
        n=n,
        density_function=aleph_density,
        t=t,
        xi=xi,
        S=S
    )

    # --------------------------------------------------------
    # Symmetrized SNN approximant
    # --------------------------------------------------------

    snn_approximation = vector_operator(
        x_eval=x_grid,
        n=n,
        density_function=F_density,
        t=t,
        xi=xi,
        S=S
    )

    # --------------------------------------------------------
    # Uniform approximation errors in C([0,1], R^2)
    # --------------------------------------------------------

    E_classical = uniform_R2_error(
        classical_approximation,
        true_values
    )

    E_snn = uniform_R2_error(
        snn_approximation,
        true_values
    )

    # --------------------------------------------------------
    # Relative reduction in the vector-valued uniform error
    # --------------------------------------------------------

    I_V_n = (
        100.0
        * (E_classical - E_snn)
        / E_classical
    )

    results.append({
        "n": n,
        "E_inf_classical_R2": E_classical,
        "E_inf_SNN_R2": E_snn,
        "I_V_n_percent": I_V_n
    })


# ============================================================
# 9. TABLE 10
# ============================================================

df = pd.DataFrame(results)

print(
    "\n"
    "Table 10. Vector-Valued Uniform Approximation "
    "Errors in Y = R^2\n"
)

print(
    df.to_string(
        index=False,
        formatters={
            "E_inf_classical_R2":
                lambda value: f"{value:.6f}",

            "E_inf_SNN_R2":
                lambda value: f"{value:.6f}",

            "I_V_n_percent":
                lambda value: f"{value:.2f}"
        }
    )
)


# ============================================================
# 10. OPTIONAL NUMERICAL CONSISTENCY CHECK
# ============================================================

assert np.all(
    df["E_inf_classical_R2"] > 0
)

assert np.all(
    df["E_inf_SNN_R2"] > 0
)

assert np.all(
    df["I_V_n_percent"] > 0
)

print(
    "\nAll tested sampling levels give "
    "E_inf_SNN_R2 < E_inf_classical_R2."
)


# ============================================================
# 11. SAVE TABLE 10 DATA
# ============================================================

output_file = "Table10_Vector_Valued_R2_Approximation.csv"

df.to_csv(
    output_file,
    index=False
)

print(
    f"\nTable 10 data saved to: {output_file}"
)