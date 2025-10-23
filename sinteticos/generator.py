from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union, Dict, Any

import numpy as np
import pandas as pd

# === SDV models (support legacy <1.0 and modern ≥1.0 import paths) ===
try:
    # Legacy SDV (<1.0)
    from sdv.tabular import CTGAN, GaussianCopula, CopulaGAN
except ImportError:
    try:
        # SDV >= 1.0 new-style synthesizers
        from sdv.single_table import (
            CTGANSynthesizer,
            GaussianCopulaSynthesizer,
            CopulaGANSynthesizer,
        )
        from sdv.metadata import SingleTableMetadata

        # Adapter wrappers that delay synthesizer construction until fit(df)
        class _BaseAdapter:
            def __init__(self, SynthClass, **kwargs):
                self._SynthClass = SynthClass
                self._kwargs = kwargs
                self._synth = None

            def fit(self, df: pd.DataFrame):
                if self._synth is None:
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=df)
                    self._synth = self._SynthClass(metadata, **self._kwargs)
                return self._synth.fit(df)

            def sample(self, num_rows: int) -> pd.DataFrame:
                if self._synth is None:
                    raise RuntimeError("Model has not been fit yet.")
                return self._synth.sample(num_rows)

        class CTGAN(_BaseAdapter):
            def __init__(self, **kwargs):
                super().__init__(CTGANSynthesizer, **kwargs)

        class GaussianCopula(_BaseAdapter):
            def __init__(self, **kwargs):
                super().__init__(GaussianCopulaSynthesizer, **kwargs)

        class CopulaGAN(_BaseAdapter):
            def __init__(self, **kwargs):
                super().__init__(CopulaGANSynthesizer, **kwargs)

    except ImportError:
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
        try:
            out[col] = out[col].astype(int)
        except Exception:
            pass
    return out


def _select_model(name: str, **kwargs) -> ModelType:
    key = (name or "").strip().lower()

    # --- CTGAN ---
    if key in {"ctgan"}:
        if CTGAN is None:
            raise ImportError(
                "SDV CTGAN model not available. We tried `sdv.tabular` and `sdv.single_table`. "
                "Please install or upgrade SDV: `pip install -U sdv`."
            )
        return CTGAN(**kwargs)

    # --- Gaussian Copula ---
    if key == "gaussiancopula":
        if GaussianCopula is None:
            try:
                from copulas.multivariate import GaussianMultivariate
            except Exception:
                raise ImportError(
                    "GaussianCopula model not available from SDV, and `copulas` fallback not found. "
                    "Please install or upgrade SDV (`pip install -U sdv`) or install copulas (`pip install copulas`)."
                )

            class _CopulasGaussian:
                def __init__(self, **_kwargs):
                    self._model = GaussianMultivariate()
                    self._columns = None
                    self._dtypes = None

                def fit(self, df: pd.DataFrame):
                    if not isinstance(df, pd.DataFrame):
                        if isinstance(df, dict):
                            try:
                                if all(np.isscalar(v) or getattr(v, "ndim", 0) == 0 for v in df.values()):
                                    df = pd.DataFrame([df])
                                else:
                                    df = pd.DataFrame(df)
                            except Exception:
                                df = pd.DataFrame([df])
                        else:
                            df = pd.DataFrame(df)
                    self._columns = list(df.columns)
                    self._dtypes = df.dtypes
                    try:
                        data = df[self._columns].astype(float)
                    except Exception:
                        data = df[self._columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
                    self._model.fit(data)

                def sample(self, num_rows: int) -> pd.DataFrame:
                    n = int(num_rows)
                    samples = self._model.sample(n)
                    if isinstance(samples, pd.DataFrame):
                        out = samples[self._columns]
                    else:
                        out = pd.DataFrame(samples, columns=self._columns)
                    try:
                        for col, dt in self._dtypes.items():
                            if pd.api.types.is_integer_dtype(dt):
                                out[col] = np.round(out[col]).astype(dt)
                            elif pd.api.types.is_bool_dtype(dt):
                                out[col] = out[col] > out[col].median()
                            else:
                                out[col] = out[col].astype(float)
                    except Exception:
                        pass
                    return out

            return _CopulasGaussian(**kwargs)
        return GaussianCopula(**kwargs)

    # --- CopulaGAN ---
    if key == "copulagan":
        if CopulaGAN is None:
            raise ImportError(
                "SDV CopulaGAN model not available. We tried `sdv.tabular` and `sdv.single_table`. "
                "Please install or upgrade SDV: `pip install -U sdv`."
            )
        return CopulaGAN(**kwargs)

    # --- Unknown ---
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
