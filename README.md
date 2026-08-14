# Synthetic Data-Guided Symmetric Neural Network Approximation in Banach Spaces

Code, synthetic data, numerical tables, and figures supporting the manuscript:

**George A. Anastassiou, Seda Karateke, and Metin Zontul**  
*“Synthetic Data-Guided Symmetric Neural Network Approximation in Banach Spaces”*  
Submitted to **Axioms** (2026).

## Overview

This repository accompanies a constructive approximation framework based on a reciprocal-deformation symmetric neural network (SNN) kernel. It contains the numerical diagnostics and computational experiments used to examine:

- activation-level reciprocal-deformation symmetry;
- density evenness, normalization, partition of unity, and tail localization;
- controlled blind prediction on a synthetic monthly temperature-like series;
- parameter-matched SNN versus classical NN comparisons;
- seasonal ARIMA and MLP empirical baselines;
- five-seed robustness analysis;
- half-order fractional approximation and fractional Taylor remainder experiments; and
- a vector-valued approximation example in \(Y=\mathbb{R}^2\).

No external or real-world dataset is used. All time-series observations are generated synthetically by the scripts in this repository.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── data/
│   ├── synthetic_monthly_temperature_seed42.csv
│   └── synthetic_monthly_temperature_five_seeds.csv
├── scripts/
│   ├── current/        # scripts corresponding to the current manuscript
│   ├── supporting/     # auxiliary / validation scripts
│   └── legacy/         # retained older plotting variants for provenance
├── results/
│   ├── tables/         # Tables 1–10 and supporting CSV outputs
│   └── figures/        # Figures 1–19
└── docs/
    ├── REPRODUCIBILITY.md
    ├── SCRIPT_MAP.md
    └── DATA_DICTIONARY.md
```

## Python environment

The numerical workflow is written for **Python 3.13**. A tested package set is provided in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Synthetic time series

For month \(j=1,\ldots,120\), the controlled temperature-like series is generated as

\[
T_j=15+10\sin\left(\frac{2\pi j}{12}-\frac{\pi}{2}\right)+0.05j+\varepsilon_j,
\qquad \varepsilon_j\sim\mathcal N(0,1.2^2).
\]

The model-selection protocol is:

- months 1–72: fitting stage for hyperparameter optimization;
- months 73–96: validation stage;
- months 1–96: final pre-test refit after hyperparameters are fixed;
- months 97–120: untouched blind-test interval.

The five robustness seeds are `42, 123, 777, 2026, 9999`.

The continuous seasonal-trend parameters are estimated in the current empirical scripts with `scipy.optimize.curve_fit`, using nonlinear least-squares fitting and the initialization

```text
[10, 2*pi/12, -pi/2, 15, 0.05]
```

for amplitude, angular frequency, phase, baseline, and linear trend, respectively.

## Main empirical configuration

The parameter-matched operator comparison uses

```text
n = 64
t = 0.5
xi = 2.0
S = e
```

The external baselines are:

- Seasonal ARIMA: `(1,0,1)(0,1,1)_12`
- MLP: hidden layers `(100, 50)`, standardized month/seasonal inputs, L-BFGS solver

For the representative seed-42 blind test, the manuscript reports the SNN result

```text
R²    = 0.9681
NMAE  = 0.0288
NRMSE = 0.0421
```

Across five independent synthetic realizations, the manuscript reports

```text
Mean R²    = 0.9500 ± 0.0239
Mean NMAE  = 0.0452 ± 0.0168
Mean NRMSE = 0.0570 ± 0.0175
```

## Reproducing the manuscript outputs

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for script-by-script commands and [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md) for the exact mapping between scripts and manuscript figures/tables.

The repository also includes the publication-ready tables and figures under `results/` so that the numerical outputs can be inspected without rerunning the full workflow.

## Data availability

All data in `data/` are synthetic and can be regenerated from the scripts. No patient, personal, proprietary, external, or real-world dataset is included.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the associated manuscript if you use this code or the numerical results.

## License

No open-source license has been assigned in this repository package. The authors should select a license before public release if reuse permissions are intended.
