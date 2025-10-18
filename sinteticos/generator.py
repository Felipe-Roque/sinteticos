from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union, Dict, Any

import numpy as np
import pandas as pd

# SDV models
try:
    from sdv.tabular import CTGAN, GaussianCopula, CopulaGAN
except Exception as e:  # pragma: no cover
    CTGAN = None
    GaussianCopula = None
    CopulaGAN = None


ModelType = Any


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    dropna: bool = False,
    numeric_columns: Optional[Union[Sequence[str], Dict[str, dict]]] = None,
    fill_value: Union[int, float] = -1,
) -> pd.DataFrame:
    """
    Basic preprocessing used in the notebook:
    - Optionally drop rows with any NA (dropna=True)
    - Optionally coerce selected columns to numeric and fill NaN with a sentinel, then cast to int

    numeric_columns:
      - If Sequence[str]: those columns will be coerced via to_numeric(errors='coerce').fillna(fill_value).astype(int)
      - If Dict[str, dict]: mapping col -> {errors, dtype, fill_value} to control behavior per column
    """
    out = df.copy()
    if dropna:
        out = out.dropna()

    if numeric_columns is None:
        return out

    if isinstance(numeric_columns, dict):
        for col, cfg in numeric_columns.items():
            if col not in out.columns:
                continue
            errs = cfg.get("errors", "coerce")
            fv = cfg.get("fill_value", fill_value)
            dtype = cfg.get("dtype", int)
            out[col] = pd.to_numeric(out[col], errors=errs).fillna(fv).astype(dtype)
        return out

    # sequence
    for col in numeric_columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill_value)
        # cast to int if looks like integer categories, else keep float
        try:
            out[col] = out[col].astype(int)
        except Exception:
            pass
    return out


def _select_model(name: str, **kwargs) -> ModelType:
    key = (name or "").strip().lower()
    if key == "ctgan":
        if CTGAN is None:
            raise ImportError("sdv.tabular.CTGAN not available. Please `pip install sdv`.")
        return CTGAN(**kwargs)
    if key == "gaussiancopula":
        if GaussianCopula is None:
            raise ImportError("sdv.tabular.GaussianCopula not available. Please `pip install sdv`.")
        return GaussianCopula(**kwargs)
    if key == "copulagan":
        if CopulaGAN is None:
            raise ImportError("sdv.tabular.CopulaGAN not available. Please `pip install sdv`.")
        return CopulaGAN(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Use 'CTGAN', 'GaussianCopula', or 'CopulaGAN'.")


def train_model(df: pd.DataFrame, model: str = "CTGAN", **model_kwargs) -> ModelType:
    """
    Create and fit an SDV tabular model on the provided DataFrame.
    Example: train_model(df, model="CTGAN", epochs=600)
    """
    mdl = _select_model(model, **model_kwargs)
    mdl.fit(df)
    return mdl


def generate_synthetic(model: ModelType, num_rows: int = 1000) -> pd.DataFrame:
    """Sample synthetic rows using a fitted model."""
    if not hasattr(model, "sample"):
        raise TypeError("Provided model does not support .sample().")
    return model.sample(int(num_rows))
