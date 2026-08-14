# Reproducibility guide

## 1. Environment

Use Python 3.13 and install the tested dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

For non-interactive execution on Linux/macOS, it is convenient to set:

```bash
export MPLBACKEND=Agg
```

## 2. Structural diagnostics: Figures 1–6, Tables 1–2

```bash
python scripts/current/01_structural_diagnostics.py
```

The original script writes its diagnostic CSVs and figures into `activation_symmetry_results_final/` relative to the working directory. The curated copies are already provided in `results/` with filenames matching the current manuscript numbering.

## 3. Representative blind test: Figures 7 and 9, Table 4

```bash
python scripts/current/02_representative_blind_test.py
```

Protocol:

- months 1–72: fit during hyperparameter search;
- months 73–96: validation;
- months 1–96: final refit;
- months 97–120: blind test.

The seasonal trend is fitted with `scipy.optimize.curve_fit`. The initialization used in the current script is

```python
p0 = [10.0, 2.0*np.pi/12.0, -np.pi/2.0, 15.0, 0.05]
```

## 4. Five-seed robustness: Figure 8, Table 5

```bash
python scripts/current/03_five_seed_robustness.py
```

Seeds: `42, 123, 777, 2026, 9999`.

The operator hyperparameters are selected once using seed 42 and are then frozen for all five realizations.

The exact Table 5 values reported in the submitted manuscript are archived as:

```text
results/tables/Table05_five_seed_performance_manuscript.csv
```

## 5. Fractional Example 1: Figures 10–13, Tables 6–7

```bash
python scripts/current/04_fractional_example1.py
```

This script displays the four figures interactively and prints the numerical tables. Publication-ready copies are provided under `results/figures/` and `results/tables/`.

## 6. Fractional Example 2: Figures 14–18, Tables 8–9

```bash
python scripts/current/05_fractional_example2.py
```

The supplied script has `SAVE_FIGURES = False` by default. Change it to `True` if you want the figure files written directly by the script. The curated repository already contains Figures 14–18 as PNG files.

## 7. Vector-valued experiment: Figure 19, Table 10

```bash
python scripts/current/06_vector_valued_table10.py
python scripts/current/07_vector_valued_figure19.py
```

The experiment uses \(Y=\mathbb{R}^2\) with the Euclidean Banach-space norm.

## 8. Numerical precision

The diagnostics use NumPy `float64` arithmetic. The stable logistic evaluation in the structural-diagnostics script uses `scipy.special.expit` when available.
