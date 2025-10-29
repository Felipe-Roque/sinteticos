# Sinteticos

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

## Installation
Install dependencies (Python 3.9+ recommended):

Option A (recommended):
```
pip install -r requirements.txt
```

Option B (manual):
```
pip install pandas numpy matplotlib seaborn sdv openpyxl
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
- This library is derived from the notebook `sintectic.ipynb`. It encapsulates the same workflow into reusable functions.
- Depending on your dataset, you may need to adjust preprocessing (numeric columns and fill values) to get optimal model convergence.
- For Excel support, ensure `openpyxl` is installed.

## License
MIT (or adapt as needed).
