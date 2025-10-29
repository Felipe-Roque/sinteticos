#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

# Ensure project root is on sys.path when running the script directly via path (python scripts/...) 
# so that `import sinteticos` works without installing the package.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sinteticos.io_utils import load_dataframe, save_dataframe
from sinteticos.generator import preprocess_dataframe, train_model, generate_synthetic
from sinteticos.evaluation import evaluate_hellinger_by_column
from sinteticos.plotting import plot_correlation_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic tabular data using SDV models")
    p.add_argument("input", help="Path to input dataset (CSV or Excel)")
    p.add_argument("output", help="Path to save synthetic dataset (CSV or Excel)")
    p.add_argument("--model", default="CTGAN", choices=["CTGAN", "GaussianCopula", "CopulaGAN"], help="Model type")
    p.add_argument("--epochs", type=int, default=300, help="Training epochs for GAN models")
    p.add_argument("--samples", type=int, default=1000, help="Number of synthetic rows to generate")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index (default: first sheet)")
    p.add_argument("--dropna", action="store_true", help="Drop rows with any NA before training")
    p.add_argument("--numeric-columns", nargs='*', default=None, help="Columns to coerce to numeric (fill NaN with -1 and cast to int)")
    p.add_argument("--subset-prefix", default=None, help="Only use columns starting with this prefix (e.g., EEE or ETI)")
    p.add_argument("--eval", action="store_true", help="Compute and print Hellinger distance per column")
    p.add_argument("--trials", type=int, default=1, help="Number of trials to run; the best (lowest mean Hellinger) synthetic will be saved")

    # Correlation options for output synthetic data
    p.add_argument("--corr", action="store_true", help="Compute and plot correlation matrices (Spearman and Kendall) for the synthetic output")
    p.add_argument("--corr-annot", action="store_true", help="Annotate correlation heatmaps with values")
    p.add_argument("--corr-mask-upper", action="store_true", help="Mask upper triangle in correlation heatmaps")
    p.add_argument("--corr-save", default=None, help="If provided, save correlation heatmaps to this directory (files named <method>_correlation.png)")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_dataframe(args.input, sheet_name=args.sheet)

    if args.subset_prefix:
        cols = [c for c in df.columns if c.startswith(args.subset_prefix)]
        if not cols:
            raise SystemExit(f"No columns starting with prefix '{args.subset_prefix}' found.")
        df = df[cols]

    df_prep = preprocess_dataframe(
        df,
        dropna=args.dropna,
        numeric_columns=args.numeric_columns,
        fill_value=-1,
    )

    model_kwargs = {}
    if args.model.lower() in {"ctgan", "copulagan"}:
        model_kwargs["epochs"] = int(args.epochs)

    best_synth = None
    best_score = None
    best_trial = None

    trials = max(1, int(args.trials))
    for t in range(1, trials + 1):
        model = train_model(df_prep, model=args.model, **model_kwargs)
        synth = generate_synthetic(model, num_rows=args.samples)
        # Evaluate mean Hellinger per column
        hell = evaluate_hellinger_by_column(df_prep, synth)
        score = float(hell["Hellinger"].mean())
        print(f"Trial {t}/{trials} - mean Hellinger: {score:.6f}")
        if best_score is None or score < best_score:
            best_score = score
            best_synth = synth
            best_trial = t

    # Save only the best synthetic dataset
    save_dataframe(best_synth, args.output, index=False)
    print(f"Saved best synthetic from trial {best_trial} with mean Hellinger={best_score:.6f} -> {args.output}")

    if args.eval:
        hell = evaluate_hellinger_by_column(df_prep, best_synth)
        print("Hellinger mean:", float(hell["Hellinger"].mean()))
        print(hell)

    if args.corr:
        # Prepare save paths if directory given
        save_dir = args.corr_save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for method in ("spearman", "kendall"):
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{method}_correlation.png")
            print(f"Plotting {method.capitalize()} correlation matrix for synthetic output...")
            plot_correlation_matrix(
                best_synth,
                method=method,
                annot=args.corr_annot,
                mask_upper=args.corr_mask_upper,
                save_path=save_path,
            )


if __name__ == "__main__":
    main()
