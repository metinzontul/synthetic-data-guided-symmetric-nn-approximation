# Script-to-manuscript map

The current manuscript contains Figures 1–19 and Tables 1–10. The files below are the primary scripts supporting those outputs.

| Script | Manuscript output(s) | Purpose |
|---|---|---|
| `scripts/current/01_structural_diagnostics.py` | Figures 1–6; Tables 1–2 | Activation symmetry, density symmetry, partition of unity, integral normalization, and tail localization diagnostics. |
| `scripts/current/02_representative_blind_test.py` | Figures 7 and 9; Tables 3–4; validation-search supplement | Representative seed-42 blind prediction, matched operator comparison, ARIMA/MLP baselines, node-level SNN weights. |
| `scripts/current/03_five_seed_robustness.py` | Figure 8; Table 5 | Five-seed robustness analysis with frozen operator hyperparameters. |
| `scripts/current/04_fractional_example1.py` | Figures 10–13; Tables 6–7 | Half-order fractional approximation setting, uniform error, MAE/RMSE, and relative improvement. |
| `scripts/current/05_fractional_example2.py` | Figures 14–18; Tables 8–9 | Fractional Taylor remainder experiment for \(\alpha=3/2\). |
| `scripts/current/06_vector_valued_table10.py` | Table 10 | Vector-valued uniform error calculations in \(Y=\mathbb{R}^2\). |
| `scripts/current/07_vector_valued_figure19.py` | Figure 19 | Geometric images of the vector-valued target and its classical/SNN approximants. |

## Supporting scripts

- `scripts/supporting/grid_search_ablation.py`: independent SNN/classical NN grid-search check.
- `scripts/supporting/blind_prediction_operator_comparison.py`: earlier operator-focused blind-prediction implementation retained for traceability.
- `scripts/supporting/representative_baselines_figure.py`: representative baseline plotting implementation.

## Legacy scripts

The `scripts/legacy/` folder contains older plotting variants that were present in the supplied code archive but are not the primary source for the current manuscript numbering. They are retained only for provenance.

## Figure filename remapping for Figures 1–6

The original diagnostic script saved figures using an earlier internal order. The curated files under `results/figures/` are renamed to match the current manuscript order:

1. partition of unity;
2. activation-level reciprocal-deformation symmetry;
3. tail decay;
4. activation-level central-symmetry numerical check;
5. even density kernel;
6. density-symmetry error.
