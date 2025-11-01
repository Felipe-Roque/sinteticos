#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

# Ensure project root on sys.path for local execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sinteticos.io_utils import load_dataframe, save_dataframe
from sinteticos.generator import (
    preprocess_dataframe,
    train_model,
    generate_synthetic,
)
from sinteticos.evaluation import evaluate_hellinger_by_column
from sinteticos.cfa import run_cfa_python_multi, parse_model_spec, CFAError


DEF_INPUT = os.path.join(PROJECT_ROOT, "data", "IAFREE_Chile.xlsx")
DEF_OUT_DIR = os.path.join(PROJECT_ROOT, "data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end evaluation for IAFREE: generate synthetic data, compute Hellinger report, "
            "and run multi-factor CFA on original and synthetic datasets."
        )
    )
    p.add_argument("--input", default=DEF_INPUT, help=f"Path to IAFREE dataset (default: {DEF_INPUT})")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index (default: first sheet)")
    p.add_argument("--out-dir", default=DEF_OUT_DIR, help=f"Directory to save outputs (default: {DEF_OUT_DIR})")
    p.add_argument("--model", default="GaussianCopula", choices=["GaussianCopula", "CTGAN", "CopulaGAN"],
                   help="Synthetic data model (default: GaussianCopula)")
    p.add_argument("--samples", type=int, default=1000, help="Number of synthetic rows to generate (default: 1000)")
    p.add_argument("--epochs", type=int, default=600, help="Training epochs for GAN models (CTGAN/CopulaGAN)")
    p.add_argument("--dropna", action="store_true", help="Drop rows with any NA before training")
    p.add_argument(
        "--numeric-columns",
        nargs='*',
        default=None,
        help=(
            "Columns to coerce to numeric (fill NaN with -1 and cast to int); "
            "if omitted, attempt to coerce all 'qa' prefixed columns automatically."
        ),
    )
    p.add_argument(
        "--factors",
        default="E_EST1=qa5+qa6+qa10; E_EST2=qa1+qa2+qa24; E_EST3=qa4+qa6+qa9",
        help=(
            "Multi-factor spec like 'E_EST1=qa5+qa6+qa10; E_EST2=qa1+qa2+qa24; E_EST3=qa4+qa6+qa9'"
        ),
    )
    p.add_argument("--prefix", default=None, help="Optional column prefix to subset before processing (e.g., qa)")
    return p.parse_args()


def maybe_infer_numeric_cols(df: pd.DataFrame, provided: Optional[List[str]]) -> Optional[List[str]]:
    if provided:
        return provided
    # Heuristic: choose columns starting with 'qa' that are not already numeric
    qa_cols = [c for c in df.columns if c.lower().startswith("qa")]
    if not qa_cols:
        return None
    return qa_cols


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()

    print("Loading dataset:", args.input)
    df_raw = load_dataframe(args.input, sheet_name=args.sheet)

    # Optionally subset by prefix
    if args.prefix:
        cols = [c for c in df_raw.columns if c.startswith(args.prefix)]
        if not cols:
            raise SystemExit(f"No columns starting with prefix '{args.prefix}' found in input dataset.")
        df_raw = df_raw[cols]

    numeric_cols = maybe_infer_numeric_cols(df_raw, args.numeric_columns)

    print("Preprocessing data...")
    df_prep = preprocess_dataframe(
        df_raw,
        dropna=args.dropna,
        numeric_columns=numeric_cols,
        fill_value=-1,
    )

    # Train synthetic model and generate data
    args.model = 'CTGAN'
    print(f"Training model: {args.model}")
    model_kwargs = {}
    if args.model.lower() in {"ctgan", "copulagan"}:
        model_kwargs["epochs"] = int(args.epochs)
    model = train_model(df_prep, model=args.model, **model_kwargs)

    print(f"Generating {args.samples} synthetic rows...")
    df_synth = generate_synthetic(model, num_rows=args.samples)

    # Save synthetic output
    ensure_out_dir(args.out_dir)
    synth_path = os.path.join(args.out_dir, "synthetic_iafree.csv")
    save_dataframe(df_synth, synth_path, index=False)
    print("Saved synthetic data to:", synth_path)

    # Hellinger report
    print("Computing Hellinger distance report (real vs synthetic)...")
    hell_report = evaluate_hellinger_by_column(df_prep, df_synth)
    # Classification columns (similar to generate_report.py)
    def _classify(h: float) -> str:
        if pd.isna(h):
            return "NA"
        if 0 <= h < 0.10:
            return "OD"
        if 0.10 <= h < 0.25:
            return "GD"
        if 0.25 <= h < 0.40:
            return "AD"
        return "UD"

    hell_report["Category"] = hell_report["Hellinger"].apply(_classify)
    hell_mean = float(hell_report["Hellinger"].mean())

    hell_path = os.path.join(args.out_dir, "hellinger_report.csv")
    hell_report.to_csv(hell_path, index=True)
    print(f"Hellinger report saved to: {hell_path} (mean={hell_mean:.6f})")

    # CFA on original and synthetic
    print("Parsing factor specification...")
    try:
        factors: Dict[str, List[str]] = parse_model_spec(args.factors)
    except CFAError as e:
        raise SystemExit(f"Invalid factor specification: {e}")

    print("Running CFA on original (preprocessed) data...")
    try:
        cfa_orig = run_cfa_python_multi(df_prep, factors=factors, estimator="ML")
    except CFAError as e:
        raise SystemExit(f"CFA on original data failed: {e}")

    cfa_orig_df = pd.DataFrame.from_dict(cfa_orig, orient="index", columns=["value"])  # type: ignore
    cfa_orig_path = os.path.join(args.out_dir, "cfa_multi_original.csv")
    cfa_orig_df.to_csv(cfa_orig_path, header=True)
    print("CFA (original) results saved to:", cfa_orig_path)

    print("Running CFA on synthetic data...")
    try:
        cfa_synth = run_cfa_python_multi(df_synth[df_prep.columns], factors=factors, estimator="ML")
    except CFAError as e:
        raise SystemExit(f"CFA on synthetic data failed: {e}")

    cfa_synth_df = pd.DataFrame.from_dict(cfa_synth, orient="index", columns=["value"])  # type: ignore
    cfa_synth_path = os.path.join(args.out_dir, "cfa_multi_synthetic.csv")
    cfa_synth_df.to_csv(cfa_synth_path, header=True)
    print("CFA (synthetic) results saved to:", cfa_synth_path)

    print("Done.")
    print("Summary:")
    print(f"  Synthetic saved: {synth_path}")
    print(f"  Hellinger report: {hell_path} (mean={hell_mean:.6f})")
    print(f"  CFA original: {cfa_orig_path}")
    print(f"  CFA synthetic: {cfa_synth_path}")


if __name__ == "__main__":
    main()
