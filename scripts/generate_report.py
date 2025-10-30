#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List

import pandas as pd

# Ensure project root on sys.path for local execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sinteticos.io_utils import load_dataframe, save_dataframe
from sinteticos.generator import preprocess_dataframe, train_model, generate_synthetic
from sinteticos.evaluation import evaluate_hellinger_by_column


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a Hellinger distance report comparing GaussianCopula and CTGAN."
    )
    p.add_argument("input", help="Path to input dataset (CSV or Excel)")
    p.add_argument("output_csv", nargs="?", default=os.path.join(PROJECT_ROOT, "data", "hellinger_report.csv"),
                   help="Path to save the Hellinger report CSV (default: data/hellinger_report.csv)")
    p.add_argument("--samples", type=int, default=1000, help="Number of synthetic rows to generate per model")
    p.add_argument("--epochs", type=int, default=300, help="Training epochs for CTGAN")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index (default: first sheet)")
    p.add_argument("--dropna", action="store_true", help="Drop rows with any NA before training")
    p.add_argument("--numeric-columns", nargs='*', default=None, help="Columns to coerce to numeric (fill NaN with -1 and cast to int)")
    p.add_argument("--subset-prefix", default=None, help="Only use columns starting with this prefix (e.g., EEE or ETI)")
    p.add_argument("--columns", default=None, help="Comma-separated list of columns to keep (e.g., qa5,qa6,qa10)")
    p.add_argument("--eval", action="store_true", help="Print the full report to stdout (compatibility flag)")
    return p.parse_args()


def _train_and_generate(df_prep: pd.DataFrame, model_name: str, samples: int, epochs: Optional[int] = None) -> pd.DataFrame:
    kwargs = {}
    if epochs is not None and model_name.lower() in {"ctgan", "copulagan"}:
        kwargs["epochs"] = int(epochs)
    model = train_model(df_prep, model=model_name, **kwargs)
    return generate_synthetic(model, num_rows=samples)


def main():
    args = parse_args()

    df = load_dataframe(args.input, sheet_name=args.sheet)

    if args.subset_prefix:
        cols = [c for c in df.columns if c.startswith(args.subset_prefix)]
        if not cols:
            raise SystemExit(f"No columns starting with prefix '{args.subset_prefix}' found.")
        df = df[cols]

    # If explicit columns are provided, keep only those
    if args.columns:
        keep = [c.strip() for c in args.columns.split(",") if c.strip()]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise SystemExit(f"Columns not found in dataset: {missing}")
        df = df[keep]

    df_prep = preprocess_dataframe(
        df,
        dropna=args.dropna,
        numeric_columns=args.numeric_columns,
        fill_value=-1,
    )

    # Train and generate with GaussianCopula and CTGAN
    synth_gc = _train_and_generate(df_prep, "GaussianCopula", samples=args.samples)
    synth_ct = _train_and_generate(df_prep, "CTGAN", samples=args.samples, epochs=args.epochs)

    # Compute Hellinger distances vs real
    hell_gc = evaluate_hellinger_by_column(df_prep, synth_gc).rename(columns={"Hellinger": "H_GaussianCopula"})
    hell_ct = evaluate_hellinger_by_column(df_prep, synth_ct).rename(columns={"Hellinger": "H_CTGAN"})

    # Compute Hellinger distances between the two synthetic outputs as an additional comparison
    hell_synths = evaluate_hellinger_by_column(synth_gc[df_prep.columns], synth_ct[df_prep.columns]).rename(columns={"Hellinger": "H_BetweenSynths"})

    # Merge
    report = hell_gc.join(hell_ct, how="outer").join(hell_synths, how="outer")

    # Append a row with column-wise mean
    mean_row = pd.DataFrame({
        "H_GaussianCopula": [report["H_GaussianCopula"].mean()],
        "H_CTGAN": [report["H_CTGAN"].mean()],
        "H_BetweenSynths": [report["H_BetweenSynths"].mean()],
    }, index=["__MEAN__"])
    report_out = pd.concat([report, mean_row], axis=0)

    # Add distribution quality categories for each Hellinger column
    def _classify(h: float) -> str:
        if pd.isna(h):
            return "NA"
        if 0 <= h < 0.10:
            return "OD"  # Optimal distribution
        if 0.10 <= h < 0.25:
            return "GD"  # Good distribution
        if 0.25 <= h < 0.40:
            return "AD"  # Acceptable with restrictions
        return "UD"      # Unacceptable distribution

    report_out["C_GaussianCopula"] = report_out["H_GaussianCopula"].apply(_classify)
    report_out["C_CTGAN"] = report_out["H_CTGAN"].apply(_classify)
    report_out["C_BetweenSynths"] = report_out["H_BetweenSynths"].apply(_classify)

    # Save CSV
    out_path = args.output_csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report_out.to_csv(out_path, index=True)

    # Print summary
    print("Hellinger report saved to:", out_path)
    print("Means -> GaussianCopula:", float(mean_row["H_GaussianCopula"][0]),
          "CTGAN:", float(mean_row["H_CTGAN"][0]),
          "BetweenSynths:", float(mean_row["H_BetweenSynths"][0]))
    print("Preview:\n", report_out.head())

    if args.eval:
        print("Full report:\n", report_out)


if __name__ == "__main__":
    main()
