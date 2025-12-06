# PySintetic

A small Python toolkit to generate synthetic tabular data from a sample dataset, inspired by the provided Jupyter notebook. It wraps SDV models (CTGAN, GaussianCopula, CopulaGAN) into a simple library API and a command-line interface.

Data location: place your datasets under the `data/` directory. This repo includes `data/IAFREE_Chile.xlsx` as an example input.

## Features
- Load CSV/Excel datasets
- Optional preprocessing (drop NA, coerce selected columns to numeric)
- Train SDV models: CTGAN, GaussianCopula, CopulaGAN
- Generate synthetic samples
- Evaluate distribution similarity using Hellinger distance per column
- Compute correlation matrices (Spearman and Kendall) for output data
- Optional plots: Hellinger bar chart, histogram comparisons, and correlation heatmaps
- CLI script for end-to-end generation (supports multiple trials and selects the best)
- Run single- or multi-factor Confirmatory Factor Analysis (CFA) in Python (semopy) with key fit indices (χ2, df, p, χ2/df, CFI, TLI, SRMR, RMSEA) and reliability (Cronbach’s alpha, McDonald’s omega); multi-factor via --model-spec. If semopy does not report p-values or SRMR directly, the toolkit now computes p-values from χ2 and df, and SRMR from observed vs. implied correlations.

## Installation
Install dependencies (Python 3.9+ recommended):

Option A (recommended):
```
pip install -r requirements.txt
```

Option B (manual):
```
pip install pandas numpy matplotlib seaborn sdv openpyxl semopy
```

Notes:
- `sdv` provides the tabular models (CTGAN, GaussianCopula, CopulaGAN) used by this project.
- Depending on your OS, `sdv` may require additional system packages or wheels (and may install optional extras like PyTorch). If you encounter build issues, please consult the SDV installation docs.

## Quick Start (Python)
```
from sinteticos import (
    load_dataframe, save_dataframe,
    preprocess_dataframe, train_model, generate_synthetic,
    evaluate_hellinger_by_column, plot_hellinger_bar, compare_histograms,
)

# 1) Load your dataset
# Use the bundled example dataset
df = load_dataframe("data/IAFREE_Chile.xlsx")
# or your own file under data/
# df = load_dataframe("data/your_dataset.xlsx")

# 2) Optional: subset columns by prefix (e.g., only EEE*)
df = df[[c for c in df.columns if c.startswith("EEE")]]

# 3) Optional preprocessing similar to the notebook
numeric_cols = ["EEE5", "EEE6", "EEE7", "EEE8", "EEE9", "EEE10", "EEE11", "Raça"]
df_prep = preprocess_dataframe(df, dropna=False, numeric_columns=numeric_cols, fill_value=-1)

# 4) Train a model and generate synthetic data
model = train_model(df_prep, model="CTGAN", epochs=600)
synth = generate_synthetic(model, num_rows=1000)

# 5) Evaluate similarity (Hellinger)
hell = evaluate_hellinger_by_column(df_prep, synth)
print("Hellinger mean:", float(hell["Hellinger"].mean()))
print(hell.head())

# 6) (Optional) Plot
plot_hellinger_bar(hell)
compare_histograms(df_prep, synth, columns=df_prep.columns[:6])

# 7) Save synthetic data
save_dataframe(synth, "data/synthetic_output.csv")
```

## Command Line Usage
A convenience script is provided at `scripts/generate_synthetic.py`.

Examples:

- Generate 1000 CTGAN samples from an Excel file and save as CSV:
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_ctgan.csv --model CTGAN --epochs 600 --samples 1000 --eval
```

- Use only columns that start with `EEE`, coerce specific columns to numeric, and drop NA rows:
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_eee.csv --subset-prefix EEE --numeric-columns EEE5 EEE6 EEE7 EEE8 EEE9 EEE10 EEE11 Raça --dropna --model CTGAN --epochs 600 --samples 1000 --eval
```

- Run multiple trials and keep only the best (lowest mean Hellinger) synthetic output:
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_best.csv --model CTGAN --epochs 600 --samples 1000 --trials 5 --eval
```

- Use GaussianCopula without epochs parameter:
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_gc.csv --model GaussianCopula --samples 500 --eval
```

### Selecting specific columns (dimensions)
If you want to run the pipeline on a specific set of items (e.g., the dmin1 dimension using items qa5, qa6, qa10), pass the --columns flag to either script:

- Generate synthetic data using only qa5, qa6, qa10:
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_dmin1.csv --columns qa5,qa6,qa10 --model GaussianCopula --samples 1000 --eval
```

- Build a Hellinger report using only qa5, qa6, qa10:
```
python scripts/generate_report.py data/IAFREE_Chile.xlsx data/hellinger_dmin1.csv --columns qa5,qa6,qa10 --samples 1000 --epochs 600 --eval
```

Notes:
- --columns accepts a comma-separated list. Columns must exist in your dataset (exact names).

## Correlation Matrices (Spearman and Kendall)
You can compute and visualize correlation matrices for the generated synthetic data from the CLI, and also save the heatmaps to files.

Display only (two windows will open, one per method):
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_corr.csv \
  --model CTGAN --epochs 600 --samples 1000 \
  --corr --corr-annot --corr-mask-upper
```

Display and save PNGs to a directory (created if missing):
```
python scripts/generate_synthetic.py data/IAFREE_Chile.xlsx data/synthetic_corr.csv \
  --model CTGAN --epochs 600 --samples 1000 \
  --corr --corr-annot --corr-mask-upper \
  --corr-save data/corr_plots
```
This writes the following files:
- data/corr_plots/spearman_correlation.png
- data/corr_plots/kendall_correlation.png

Flags explained:
- --corr: enables correlation computation and plotting for the best synthetic output from the run.
- --corr-annot: annotates each heatmap cell with the correlation value.
- --corr-mask-upper: hides the upper triangle to reduce visual clutter.
- --corr-save DIR: saves the heatmaps under DIR. If omitted, plots are only displayed.

Tips:
- Correlations are computed on numeric columns. If your variables are coded as strings, use preprocessing to coerce them to numeric (see --numeric-columns in the examples above).
- On headless servers (no GUI), figures may not display, but they will be saved when you pass --corr-save.

Python API example:
```
from sinteticos import spearman_correlation, kendall_correlation, plot_correlation_matrix

# df is your DataFrame (real or synthetic)
spear = spearman_correlation(df)
kend = kendall_correlation(df)

# Heatmaps
plot_correlation_matrix(df, method="spearman", annot=True, mask_upper=True, save_path="data/spearman_corr.png")
plot_correlation_matrix(df, method="kendall", save_path="data/kendall_corr.png")
```

### Hellinger Report: GaussianCopula vs CTGAN
A separate script generates a per-column Hellinger report comparing GaussianCopula and CTGAN:
```
python scripts/generate_report.py data/IAFREE_Chile.xlsx data/hellinger_report.csv --samples 1000 --epochs 600 --eval
```
The output CSV (by default saved at `data/hellinger_report.csv`) contains columns:
- H_GaussianCopula: Hellinger distance between real data and GaussianCopula synthetic
- H_CTGAN: Hellinger distance between real data and CTGAN synthetic
- H_BetweenSynths: Hellinger distance between the two synthetic outputs
A summary row `__MEAN__` gives the mean across columns.

## Confirmatory Factor Analysis (CFA) in Python (semopy)
You can evaluate the psychometric adequacy of a dimension (e.g., dmin1) directly in Python using semopy. This runs a single-factor CFA and reports the standard fit indices and reliability metrics.

CLI example (using items qa5, qa6, qa10 for factor dmin1):
```
python scripts/run_cfa.py data/IAFREE_Chile.xlsx \
  --columns qa5,qa6,qa10 \
  --factor-name dmin1 \
  --estimator DWLS \
  --output data/cfa_dmin1.csv
```
Notes:
- Estimator options: ML, WLS, DWLS. For ordinal items (Likert), DWLS is often recommended when available.
- Results are printed to the console and saved to the CSV specified by --output with columns:
  chisq, df, pvalue, chisq_df, cfi, tli, srmr, rmsea, alpha, omega

Python API example:
```
import pandas as pd
from sinteticos import load_dataframe
from sinteticos.cfa import run_cfa_python

df = load_dataframe("data/IAFREE_Chile.xlsx")
res = run_cfa_python(df, items=["qa5","qa6","qa10"], factor_name="dmin1", estimator="DWLS", save_path="data/cfa_dmin1.csv")
print(res)
```
Interpretation guidelines (per Brown, 2015; Hayes & Coutts, 2020):
- Good fit: p>0.05 (chi-square), χ2/df ≤ 5, CFI/TLI ≥ 0.95, SRMR/RMSEA ≤ 0.08
- Reliability acceptable from α, ω ≥ 0.60

### Multi-factor CFA (multiple dimensions)
You can fit multiple latent dimensions in one model. Provide a model specification string that defines each factor and its items using the syntax:

- One definition per factor in the form: name = item1 + item2 + item3
- Separate factor definitions with semicolons (;) or newlines
- Each factor must have at least 2 items

Examples of model specs:
- "dim2=qa1+qa2+qa24; dim3=qe4+qe6+qe9"
- "F1 = x1 + x2 + x3; F2 = y1 + y2 + y3 + y4"

CLI example:
```
python scripts/run_cfa.py data/IAFREE_Chile.xlsx \
  --sheet 0 \
  --model-spec "dim2=qa1+qa2+qa24; dim3=qe4+qe6+qe9" \
  --estimator DWLS \
  --output data/cfa_multi.csv
```

Python API examples:
- Using the top-level API:
```
from sinteticos import load_dataframe, parse_model_spec, run_cfa_python_multi

df = load_dataframe("data/IAFREE_Chile.xlsx")
factors = parse_model_spec("dim2=qa1+qa2+qa24; dim3=qe4+qe6+qe9")
res = run_cfa_python_multi(df, factors=factors, estimator="DWLS", save_path="data/cfa_multi.csv")
print(res)
```
- Or importing from the cfa module:
```
from sinteticos.cfa import parse_model_spec, run_cfa_python_multi
```

Outputs:
- Global fit indices: chisq, df, pvalue, chisq_df, cfi, tli, srmr, rmsea
- Per-factor reliability fields: alpha_<factor>, omega_<factor>
  - Note: omega_<factor> is currently a placeholder (NaN) in the multi-factor function; alpha_<factor> is Cronbach’s alpha computed from the factor’s items.

Notes:
- Items referenced in the model spec must exist as columns in your dataset (exact names).
- Estimator options are the same as for single-factor: ML, WLS, DWLS.

## Project Structure
- data/
  - IAFREE_Chile.xlsx
- scripts/
  - generate_synthetic.py
- sinteticos/
  - __init__.py
  - io_utils.py
  - generator.py
  - evaluation.py
  - plotting.py
- sintectic.ipynb
- README.md

## Function Paths (API)
- sinteticos.io_utils.load_dataframe(path, sheet_name=None)
- sinteticos.io_utils.save_dataframe(df, path, index=False)
- sinteticos.generator.preprocess_dataframe(df, dropna=False, numeric_columns=None, fill_value=-1)
- sinteticos.generator.train_model(df, model="CTGAN", **model_kwargs)
- sinteticos.generator.generate_synthetic(model, num_rows=1000)
- sinteticos.evaluation.hellinger_distance(p, q)
- sinteticos.evaluation.evaluate_hellinger_by_column(df_real, df_synth)
- sinteticos.evaluation.compute_correlation(df, method)
- sinteticos.evaluation.spearman_correlation(df)
- sinteticos.evaluation.kendall_correlation(df)
- sinteticos.plotting.plot_hellinger_bar(df_hellinger, title="...")
- sinteticos.plotting.compare_histograms(df_real, df_synth, columns=None, n_cols=3, bins=None)
- sinteticos.plotting.plot_correlation_matrix(df, method="spearman", annot=False, mask_upper=False, save_path=None)

These are also imported into the top-level namespace:
- from sinteticos import load_dataframe, save_dataframe, preprocess_dataframe, train_model, generate_synthetic, evaluate_hellinger_by_column, compute_correlation, spearman_correlation, kendall_correlation, plot_hellinger_bar, compare_histograms, plot_correlation_matrix

## Notes
- Depending on your dataset, you may need to adjust preprocessing (numeric columns and fill values) to get optimal model convergence.
- For Excel support, ensure `openpyxl` is installed.

## License
MIT (or adapt as needed).
