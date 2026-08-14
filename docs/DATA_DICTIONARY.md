# Data dictionary

## `data/synthetic_monthly_temperature_seed42.csv`

Representative synthetic realization used for the seed-42 blind-prediction experiment.

| Column | Description |
|---|---|
| `seed` | Random seed used to generate the Gaussian noise realization. |
| `month` | Month index, 1–120. |
| `temperature_like_value` | Synthetic response \(T_j\). |
| `split` | `fit`, `validation`, or `blind_test`. |

## `data/synthetic_monthly_temperature_five_seeds.csv`

Long-format synthetic dataset containing all five robustness realizations (`42`, `123`, `777`, `2026`, `9999`). Columns are identical to the seed-42 file.

## `results/tables/`

- `Table01_structural_diagnostics.csv`: activation/density/partition/integral diagnostic values.
- `Table02_tail_decay.csv`: numerical tail mass, theoretical bound, and ratio.
- `Table03_model_configurations.csv`: final operator and baseline configurations.
- `Table04_representative_blind_test_metrics.csv`: seed-42 blind-test NMAE, NRMSE, and \(R^2\).
- `Table05_five_seed_performance_manuscript.csv`: five-seed summary values reported in the manuscript.
- `Table06_uniform_error_example1.csv`: uniform-error comparison for fractional Example 1.
- `Table07_extended_metrics_example1.csv`: MAE, RMSE, and \(R^2\) for fractional Example 1.
- `Table08_fractional_taylor_uniform_remainders.csv`: uniform fractional Taylor remainders for Example 2.
- `Table09_fractional_taylor_extended_metrics.csv`: extended remainder metrics for Example 2.
- `Table10_vector_valued_uniform_errors.csv`: vector-valued uniform errors and relative reductions.

Files prefixed with `Supplement_` are supporting numerical outputs not presented as numbered manuscript tables.
