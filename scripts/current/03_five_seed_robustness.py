# -*- coding: utf-8 -*-
"""
Author: Metin Zontul

Table 5 + Figure 8
Five-seed robustness analysis with leakage-free one-time hyperparameter
selection and a frozen parameter-matched SNN/Classical NN configuration.

Protocol
--------
Hyperparameter selection (seed 42 only):
    Months 1--72   : fit continuous seasonal trend
    Months 73--96  : validation for independent SNN/Classical NN grid search
    Months 97--120 : untouched during model selection

Five-seed robustness evaluation:
    Hyperparameters are frozen after the seed-42 validation stage.
    For each seed, months 1--96 are used for final fitting and months 97--120
    are used exclusively for blind-test evaluation.

Final matched operator configuration:
    n = 64, t = 0.5, xi = 2.0, S = e

Notes
-----
- For the SNN, t=0.5 and t=2.0 are reciprocal-equivalent optima because
  the symmetrized density kernel is invariant under t <-> 1/t. The t=0.5
  representative is used to match the Classical NN optimum exactly.
- Seasonal ARIMA failures are not silently replaced with artificial forecasts.
- NMAE/NRMSE use the realization-wide range only as a post-prediction
  reporting scale, consistent with the manuscript's metric definition.
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
# 1. CONSTANTS
# ============================================================
MONTHS_TOTAL = 120
FIT_SIZE = 72
VALIDATION_SIZE = 24
TRAIN_SIZE = FIT_SIZE + VALIDATION_SIZE      # 96
TEST_SIZE = MONTHS_TOTAL - TRAIN_SIZE        # 24

MONTHS = np.arange(1, MONTHS_TOTAL + 1)
X_NODES = -1.0 + 2.0 * (MONTHS - 1) / (MONTHS_TOTAL - 1)
X_FIT = X_NODES[:FIT_SIZE]
X_VAL = X_NODES[FIT_SIZE:TRAIN_SIZE]
X_TRAIN = X_NODES[:TRAIN_SIZE]
X_TEST = X_NODES[TRAIN_SIZE:]

N_GRID = [16, 32, 64]
T_GRID = [0.5, 2.0, 3.0]   # t=1 omitted: degenerate reciprocal case
XI_GRID = [0.5, 1.0, 2.0]
TIE_TOL = 1e-12

SELECTION_SEED = 42
SEEDS = [42, 123, 777, 2026, 9999]

OUTPUT_DIR = Path(__file__).resolve().parent

# ============================================================
# 2. COMMON FUNCTIONS
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
    return 0.5 * (
        k_activation(x, t, xi)
        + k_activation(x, 1.0 / t, xi)
    )


def aleph_density(x, t, xi):
    return 0.5 * (
        k_activation(x + 1.0, t, xi)
        - k_activation(x - 1.0, t, xi)
    )


def F_density(x, t, xi):
    return 0.5 * (
        k_sym(x + 1.0, t, xi)
        - k_sym(x - 1.0, t, xi)
    )


def generalized_operator(x_eval, n, density_func, t, xi, popt):
    a_n = float(n)
    b_n = np.sqrt(n**2 + 2.0)

    m_vals = np.arange(
        int(np.ceil(-b_n)),
        int(np.floor(b_n)) + 1
    )

    f_nodes = seasonal_trend_func(m_vals / b_n, *popt)
    z = a_n * np.asarray(x_eval)[:, None] - m_vals[None, :]
    weights = density_func(z, t=t, xi=xi)
    denominator = np.sum(weights, axis=1)

    if np.any(denominator <= 0.0):
        raise FloatingPointError(
            "Encountered a non-positive normalization denominator."
        )

    return (weights @ f_nodes) / denominator


def get_metrics(preds, true_vals, y_min, y_max):
    if y_max <= y_min:
        raise ValueError("Normalization range must be positive.")

    preds_norm = (preds - y_min) / (y_max - y_min)
    true_norm = (true_vals - y_min) / (y_max - y_min)

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
                preds = generalized_operator(
                    val_x, n, density_func, t, xi, popt
                )
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
        np.isclose(
            df["Validation_R2"],
            best_score,
            rtol=0.0,
            atol=TIE_TOL
        )
    ].copy()

    tied = tied.sort_values(
        ["n", "t", "xi"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    return df, tied


def generate_series(seed):
    rng = np.random.RandomState(seed)
    return (
        15.0
        + 10.0 * np.sin(2.0 * np.pi * MONTHS / 12.0 - np.pi / 2.0)
        + 0.05 * MONTHS
        + rng.normal(0.0, 1.2, MONTHS_TOTAL)
    )


# ============================================================
# 3. ONE-TIME LEAKAGE-FREE HYPERPARAMETER SELECTION
#    Seed 42: months 1--72 fit / 73--96 validation
# ============================================================
selection_data = generate_series(SELECTION_SEED)
fit_data = selection_data[:FIT_SIZE]
val_data = selection_data[FIT_SIZE:TRAIN_SIZE]

p0 = [
    10.0,
    2.0 * np.pi / 12.0,
    -np.pi / 2.0,
    15.0,
    0.05
]

popt_tuning, _ = curve_fit(
    seasonal_trend_func,
    X_FIT,
    fit_data,
    p0=p0,
    maxfev=20000
)

snn_grid, snn_ties = run_grid_search(
    X_VAL, val_data, popt_tuning, F_density, "SNN"
)

cnn_grid, cnn_ties = run_grid_search(
    X_VAL, val_data, popt_tuning, aleph_density, "Classical NN"
)

# Classical NN optimum
cnn_best = cnn_ties.iloc[0][["n", "t", "xi"]].to_dict()

# Select the reciprocal-equivalent SNN optimum matching Classical NN exactly.
matched_mask = (
    (snn_ties["n"] == cnn_best["n"])
    & np.isclose(
        snn_ties["t"], cnn_best["t"],
        atol=TIE_TOL, rtol=0.0
    )
    & np.isclose(
        snn_ties["xi"], cnn_best["xi"],
        atol=TIE_TOL, rtol=0.0
    )
)

if not matched_mask.any():
    raise RuntimeError(
        "No parameter-matched SNN optimum exists for the "
        "Classical NN optimum."
    )

snn_selected = snn_ties.loc[matched_mask].iloc[0][
    ["n", "t", "xi"]
].to_dict()

for key in ("n", "t", "xi"):
    if not np.isclose(
        float(snn_selected[key]),
        float(cnn_best[key]),
        atol=TIE_TOL,
        rtol=0.0
    ):
        raise RuntimeError(
            "The SNN and Classical NN configurations are not matched."
        )

OPT_N = int(snn_selected["n"])
OPT_T = float(snn_selected["t"])
OPT_XI = float(snn_selected["xi"])

print("=== Frozen Parameter-Matched Configuration ===")
print(f"SNN          : n={OPT_N}, t={OPT_T}, xi={OPT_XI}")
print(f"Classical NN : n={OPT_N}, t={OPT_T}, xi={OPT_XI}")
print(
    "Selected once using seed 42: months 1--72 fit, "
    "73--96 validation, 97--120 untouched.\n"
)

pd.concat(
    [snn_grid, cnn_grid],
    ignore_index=True
).to_csv(
    OUTPUT_DIR / "Table5_GridSearch_Validation_Results.csv",
    index=False
)


# ============================================================
# 4. FIVE-SEED BLIND-TEST LOOP WITH FROZEN HYPERPARAMETERS
# ============================================================
results = []

for idx, seed in enumerate(SEEDS, start=1):
    print(f"[{idx}/{len(SEEDS)}] Running seed {seed}")

    T_DATA = generate_series(seed)
    TRAIN_DATA = T_DATA[:TRAIN_SIZE]
    TEST_DATA = T_DATA[TRAIN_SIZE:]

    # Post-prediction reporting range only; not used for fitting or tuning.
    y_min = np.min(T_DATA)
    y_max = np.max(T_DATA)

    # Final refit on all 96 pre-test months.
    popt_final, _ = curve_fit(
        seasonal_trend_func,
        X_TRAIN,
        TRAIN_DATA,
        p0=p0,
        maxfev=20000
    )

    snn_preds = generalized_operator(
        X_TEST, OPT_N, F_density, OPT_T, OPT_XI, popt_final
    )

    cnn_preds = generalized_operator(
        X_TEST, OPT_N, aleph_density, OPT_T, OPT_XI, popt_final
    )

    # --------------------------------------------------------
    # MLP baseline
    # --------------------------------------------------------
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
        random_state=seed,
        solver="lbfgs"
    )

    mlp.fit(X_train_mlp_scaled, TRAIN_DATA)
    mlp_preds = mlp.predict(X_test_mlp_scaled)

    # --------------------------------------------------------
    # Seasonal ARIMA baseline
    # No silent zero-forecast fallback.
    # --------------------------------------------------------
    arima = ARIMA(
        TRAIN_DATA,
        order=(1, 0, 1),
        seasonal_order=(0, 1, 1, 12)
    ).fit()

    arima_preds = arima.forecast(steps=TEST_SIZE)

    model_predictions = {
        "SNN": snn_preds,
        "Classical NN": cnn_preds,
        "Seasonal ARIMA": arima_preds,
        "MLP": mlp_preds,
    }

    for model_name, preds in model_predictions.items():
        metrics = get_metrics(preds, TEST_DATA, y_min, y_max)
        results.append({
            "Model": model_name,
            "Seed": seed,
            **metrics
        })


# ============================================================
# 5. TABLE 5: FIVE-SEED SUMMARY
# ============================================================
df_results = pd.DataFrame(results)

df_results.to_csv(
    OUTPUT_DIR / "Table5_FiveSeed_PerSeed_Metrics.csv",
    index=False
)

agg_df = df_results.groupby("Model").agg({
    "NMAE": ["mean", "std"],
    "NRMSE": ["mean", "std"],
    "R2": ["mean", "std"],
})

MODEL_ORDER = [
    "SNN",
    "Classical NN",
    "Seasonal ARIMA",
    "MLP"
]

agg_df = agg_df.loc[MODEL_ORDER]

final_table = pd.DataFrame(index=MODEL_ORDER)

final_table["NMAE (Mean ± Std)"] = (
    agg_df[("NMAE", "mean")].map("{:.4f}".format)
    + " ± "
    + agg_df[("NMAE", "std")].map("{:.4f}".format)
)

final_table["NRMSE (Mean ± Std)"] = (
    agg_df[("NRMSE", "mean")].map("{:.4f}".format)
    + " ± "
    + agg_df[("NRMSE", "std")].map("{:.4f}".format)
)

final_table["R2 (Mean ± Std)"] = (
    agg_df[("R2", "mean")].map("{:.4f}".format)
    + " ± "
    + agg_df[("R2", "std")].map("{:.4f}".format)
)

final_table.to_csv(
    OUTPUT_DIR / "Table5_FiveSeed_Performance.csv",
    encoding="utf-8-sig"
)

print("\n=== TABLE 5: Five-Seed Blind-Test Performance ===")
print(final_table.to_string())


# ============================================================
# 6. FIGURE 8: FIVE-SEED ROBUSTNESS VISUALIZATION
# ============================================================
r2_means = agg_df[("R2", "mean")]
r2_stds = agg_df[("R2", "std")]

nmae_means = agg_df[("NMAE", "mean")]
nmae_stds = agg_df[("NMAE", "std")]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left panel: R^2
for i, model in enumerate(MODEL_ORDER):
    axes[0].bar(
        i,
        r2_means.loc[model],
        yerr=r2_stds.loc[model],
        capsize=8,
        alpha=0.82,
        edgecolor="black",
        linewidth=1.2
    )

axes[0].set_xticks(range(len(MODEL_ORDER)))
axes[0].set_xticklabels(MODEL_ORDER)
axes[0].set_title(r"$R^2$ Scores (5-Seed Mean $\pm$ Std)", fontsize=14)
axes[0].set_ylabel(r"$R^2$ Score", fontsize=12)
axes[0].set_ylim(0.0, 1.10)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].set_axisbelow(True)

for i, value in enumerate(r2_means):
    axes[0].text(
        i,
        0.10,
        f"{value:.4f}",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11
    )

# Right panel: NMAE
for i, model in enumerate(MODEL_ORDER):
    axes[1].bar(
        i,
        nmae_means.loc[model],
        yerr=nmae_stds.loc[model],
        capsize=8,
        alpha=0.82,
        edgecolor="black",
        linewidth=1.2
    )

axes[1].set_xticks(range(len(MODEL_ORDER)))
axes[1].set_xticklabels(MODEL_ORDER)
axes[1].set_title("NMAE Scores (5-Seed Mean ± Std)", fontsize=14)
axes[1].set_ylabel("Normalized Mean Absolute Error", fontsize=12)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].set_axisbelow(True)

# Dynamic label height keeps values readable if the means change.
label_y = max(0.006, float(nmae_means.max()) * 0.08)
for i, value in enumerate(nmae_means):
    axes[1].text(
        i,
        label_y,
        f"{value:.4f}",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure8_FiveSeed_Robustness_600dpi.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nSaved outputs:")
print("- Table5_GridSearch_Validation_Results.csv")
print("- Table5_FiveSeed_PerSeed_Metrics.csv")
print("- Table5_FiveSeed_Performance.csv")
print("- Figure8_FiveSeed_Robustness_600dpi.png")
