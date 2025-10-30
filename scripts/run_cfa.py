#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

# Ensure project root on sys.path for local execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sinteticos.io_utils import load_dataframe
from sinteticos.cfa import run_cfa_python, run_cfa_python_multi, parse_model_spec, CFAError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Confirmatory Factor Analysis (CFA) in Python (semopy) for a set of items "
            "and report fit indices (χ2, df, p, χ2/df, CFI, TLI, SRMR, RMSEA) and reliability (α, ω)."
        )
    )
    p.add_argument("input", help="Path to input dataset (CSV or Excel)")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index (default: first sheet)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--columns", help="Comma-separated list of item columns for a single factor (e.g., qa5,qa6,qa10)")
    group.add_argument("--model-spec", help="Multi-factor spec like 'dim2=qa1+qa2+qa24; dim3=qe4+qe6+qe9'")
    p.add_argument("--factor-name", default="dmin1", help="Name of the latent factor (default: dmin1; only for --columns)")
    p.add_argument("--estimator", default="ML", choices=["ML", "WLS", "DWLS"],
                   help="Estimator for semopy (default: ML). DWLS recommended for ordinal items if available.")
    p.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "cfa_dmin1.csv"),
        help="Path to save CFA results CSV (default: data/cfa_dmin1.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = load_dataframe(args.input, sheet_name=args.sheet)

    try:
        if args.model_spec:
            factors = parse_model_spec(args.model_spec)
            results = run_cfa_python_multi(
                df,
                factors=factors,
                estimator=args.estimator,
                save_path=args.output,
            )
        else:
            items: List[str] = [c.strip() for c in args.columns.split(",") if c.strip()]
            results = run_cfa_python(
                df,
                items=items,
                factor_name=args.factor_name,
                estimator=args.estimator,
                save_path=args.output,
            )
    except CFAError as e:
        raise SystemExit(f"CFA failed: {e}")

    print("CFA results saved to:", args.output)
    print("Summary:")
    # Pretty print all keys; floats with 6 decimals
    for k in sorted(results.keys()):
        v = results.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            try:
                print(f"  {k}: {v:.6f}")
            except Exception:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
