# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:58:21 2026

@author: Metin Zontul
"""

"""
=============================================================================
File: grid_search_ablation.py
Description: Conducts an independent grid search optimization for the SNN and 
             Classical NN operators over the validation set. Ensures perfectly 
             fair ablation study conditions by eliminating parameter bias.
=============================================================================
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. COMMON FUNCTIONS & CONSTANTS
# ============================================================
months_total = 120
train_size = 96
test_size = months_total - train_size
months_arr = np.arange(1, months_total + 1)
x_nodes = -1 + 2 * (months_arr - 1) / (months_total - 1)
x_train, x_test = x_nodes[:train_size], x_nodes[train_size:]

def x_to_month(x):
    return ((x + 1) / 2.0) * (months_total - 1) + 1

def seasonal_trend_func(x, A, w, phi, B, gamma):
    m = x_to_month(x)
    return A * np.sin(w * m + phi) + B + gamma * m

def k_activation(x, t, xi):
    val_clipped = np.clip(-xi * x, -np.inf, 200) 
    return 1.0 / (1.0 + t * np.exp(val_clipped))

def k_sym(x, t, xi):
    return 0.5 * (k_activation(x, t, xi) + k_activation(x, 1.0/t, xi))

def aleph_density(x, t, xi):
    return 0.5 * (k_activation(x + 1, t, xi) - k_activation(x - 1, t, xi))

def F_density(x, t, xi):
    return 0.5 * (k_sym(x + 1, t, xi) - k_sym(x - 1, t, xi))

def generalized_operator(x_eval, a_n, b_n, density_func, t_param, xi_param, popt_vals):
    m_vals = np.arange(int(np.ceil(-b_n)), int(np.floor(b_n)) + 1)
    f_nodes = seasonal_trend_func(m_vals / b_n, *popt_vals)
    z = a_n * x_eval[:, None] - m_vals[None, :]
    weights = density_func(z, t=t_param, xi=xi_param)
    num = weights @ f_nodes
    den = np.sum(weights, axis=1)
    return num / den, weights

# ============================================================
# 2. FAIR HYPERPARAMETER OPTIMIZATION (INDEPENDENT GRID SEARCH)
# ============================================================
print("Initiating Independent Grid Search Optimization for both models...\n")

# Fix seed to prevent data leakage during optimization
np.random.seed(42) 
T_data_opt = 15 + 10 * np.sin(2 * np.pi * months_arr / 12 - np.pi/2) + 0.05 * months_arr + np.random.normal(0, 1.2, months_total)
train_data_opt = T_data_opt[:train_size]

p0 = [10, 2 * np.pi / 12, -np.pi / 2, 15, 0.05]
popt_opt, _ = curve_fit(seasonal_trend_func, x_train, train_data_opt, p0=p0)

# Validation Set: The last 24 months of the training data
val_x = x_train[-24:]
val_y = train_data_opt[-24:]

# Search Space (t=1.0 is intentionally omitted to avoid exact formula overlap)
n_grid = [16, 32, 64]
t_grid = [0.5, 2.0, 3.0]
xi_grid = [0.5, 1.0, 2.0]

best_snn_r2, best_cnn_r2 = -np.inf, -np.inf
best_snn_params, best_cnn_params = {}, {}

for n_val in n_grid:
    for t_val in t_grid:
        for xi_val in xi_grid:
            a_n, b_n = n_val, np.sqrt(n_val**2 + 2)
            
            # Evaluate SNN
            val_preds_snn, _ = generalized_operator(val_x, a_n, b_n, F_density, t_val, xi_val, popt_opt)
            current_r2_snn = r2_score(val_y, val_preds_snn)
            if current_r2_snn > best_snn_r2:
                best_snn_r2 = current_r2_snn
                best_snn_params = {'n': n_val, 't': t_val, 'xi': xi_val}
            
            # Evaluate Classical NN
            val_preds_cnn, _ = generalized_operator(val_x, a_n, b_n, aleph_density, t_val, xi_val, popt_opt)
            current_r2_cnn = r2_score(val_y, val_preds_cnn)
            if current_r2_cnn > best_cnn_r2:
                best_cnn_r2 = current_r2_cnn
                best_cnn_params = {'n': n_val, 't': t_val, 'xi': xi_val}

print(f"Best SNN Parameters Found: n={best_snn_params['n']}, t={best_snn_params['t']}, xi={best_snn_params['xi']}")
print(f"Best Classical NN Parameters Found: n={best_cnn_params['n']}, t={best_cnn_params['t']}, xi={best_cnn_params['xi']}\n")
print("Grid Search completed. Please enforce these parameters in the subsequent test scripts to ensure a fair ablation study.")