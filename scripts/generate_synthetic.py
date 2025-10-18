#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List, Optional

import pandas as pd

from sinteticos.io_utils import load_dataframe, save_dataframe
from sinteticos.generator import preprocess_dataframe, train_model, generate_synthetic
from sinteticos.evaluation import evaluate_hellinger_by_column


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic tabular data using SDV models")
    p.add_argument("input", help="Path to input dataset (CSV or Excel)")
    p.add_argument("output", help="Path to save synthetic dataset (CSV or Excel)")
    p.add_argument("--model", default="CTGAN", choices=["CTGAN", "GaussianCopula", "CopulaGAN"], help="Model type")
    p.add_argument("--epochs", type=int, default=300, help="Training epochs for GAN models")
    p.add_argument("--samples", type=int, default=1000, help="Number of synthetic rows to generate")
    p.add_argument("--sheet", default=None, help="Excel sheet name (if input is Excel)")
    p.add_argument("--dropna", action="store_true", help="Drop rows with any NA before training")
    p.add_argument("--numeric-columns", nargs='*', default=None, help="Columns to coerce to numeric (fill NaN with -1 and cast to int)")
    p.add_argument("--subset-prefix", default=None, help="Only use columns starting with this prefix (e.g., EEE or ETI)")
    p.add_argument("--eval", action="store_true", help="Compute and print Hellinger distance per column")
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

    model = train_model(df_prep, model=args.model, **model_kwargs)
    synth = generate_synthetic(model, num_rows=args.samples)

    save_dataframe(synth, args.output, index=False)

    if args.eval:
        hell = evaluate_hellinger_by_column(df_prep, synth)
        print("Hellinger mean:", float(hell["Hellinger"].mean()))
        print(hell)


if __name__ == "__main__":
    main()
