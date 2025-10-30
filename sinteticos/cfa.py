from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CFAError(RuntimeError):
    pass


def parse_model_spec(spec: str) -> Dict[str, List[str]]:
    """
    Parse a simple multi-factor CFA specification string into a mapping.

    The expected format is a semicolon- or newline-separated list of factor definitions,
    where each definition has the form:
        factor = item1 + item2 + item3

    Example:
        "dim2=qa1 + qa2 + qa24; dim3=qe4 + qe6 + qe9"

    Returns a dictionary like {"dim2": ["qa1","qa2","qa24"], "dim3": ["qe4","qe6","qe9"]}
    """
    factors: Dict[str, List[str]] = {}
    if not spec:
        return factors
    # Split by semicolons or newlines
    parts = []
    for chunk in spec.replace("\n", ";").split(";"):
        s = chunk.strip()
        if s:
            parts.append(s)
    for part in parts:
        if "=" not in part:
            raise CFAError(f"Invalid factor definition (missing '='): {part}")
        name, rhs = part.split("=", 1)
        name = name.strip()
        if not name:
            raise CFAError(f"Empty factor name in part: {part}")
        # Allow + or , as separators
        rhs = rhs.replace(",", "+")
        items = [tok.strip() for tok in rhs.split("+") if tok.strip()]
        if len(items) < 2:
            # Keep consistent with alpha requiring >=2 items
            raise CFAError(f"Factor '{name}' must have at least 2 items: got {items}")
        factors[name] = items
    return factors


def _cronbach_alpha(x: np.ndarray) -> float:
    # x: n_samples x n_items
    if x.ndim != 2 or x.shape[1] < 2:
        return float("nan")
    k = x.shape[1]
    variances = np.nanvar(x, axis=0, ddof=1)
    total_var = np.nanvar(np.nansum(x, axis=1), ddof=1)
    if total_var <= 0:
        return float("nan")
    alpha = (k / (k - 1.0)) * (1.0 - np.nansum(variances) / total_var)
    return float(alpha)


def _omega_total_std(loadings: np.ndarray, theta: np.ndarray) -> float:
    """Compute McDonald's omega total from standardized solution.
    loadings: (p,) standardized loadings (lambda) for single factor
    theta: (p,) residual variances (unique variances) for standardized items
    Assumes factor variance = 1.
    """
    print(loadings, theta)
    if loadings.ndim != 1:
        loadings = loadings.ravel()
    if theta.ndim != 1:
        theta = theta.ravel()
    num = np.sum(loadings) ** 2
    den = num + np.sum(theta)
    if den <= 0:
        return float("nan")
    return float(num / den)


def run_cfa_python(
    df: pd.DataFrame,
    items: List[str],
    factor_name: str = "dmin1",
    estimator: str = "ML",
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run a single-factor Confirmatory Factor Analysis (CFA) in Python using semopy.

    Parameters:
    - df: DataFrame with the items.
    - items: list of item column names.
    - factor_name: name for the latent factor.
    - estimator: one of {'ML', 'WLS', 'DWLS'} depending on semopy support (default ML).
    - save_path: optional CSV path to save the results.

    Returns a dict with keys: chisq, df, pvalue, chisq_df, cfi, tli, srmr, rmsea, alpha, omega
    """
    try:
        from semopy import Model
        from semopy import calc_stats
    except Exception as e:
        raise CFAError(
            "semopy is required for Python CFA. Please install it (pip install semopy)."
        ) from e

    missing = [c for c in items if c not in df.columns]
    if missing:
        raise CFAError(f"Items not found in DataFrame: {missing}")

    data = df[items].copy()
    # Coerce to numeric and drop rows with NA
    for c in items:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(axis=0, how="any").reset_index(drop=True)
    if data.empty:
        raise CFAError("No data left after cleaning NA values in CFA items.")

    # Build model string
    model_rhs = " + ".join(items)
    model_desc = f"{factor_name} =~ {model_rhs}"

    # semopy model
    model = Model(model_desc)

    # semopy uses ML by default; DWLS/WLS available in newer versions via argument 'objective'/'estimator'.
    try:
        model.fit(data, estimator=estimator)
    except TypeError:
        # Older semopy: no estimator kw; fall back to default fit
        model.fit(data)

    # Collect fit statistics from semopy; different versions have different signatures.
    try:
        stats = calc_stats(model, data)
    except TypeError:
        # Older semopy versions expect only the model
        stats = calc_stats(model)

    print(stats)
    # stats is a DataFrame with index containing 'chi2', 'DoF', 'p-value', 'CFI', 'TLI', 'SRMR', 'RMSEA'
    def _get_stat(name: str) -> Optional[float]:
        # Try several variants to be compatible with different semopy versions
        for idx_name in (name, name.lower()):
            try:
                row = stats.loc[idx_name]
            except Exception:
                continue
            # Try common column names
            for col in ('Value', 'value', 'Stat', 'stat'):
                try:
                    val = row[col]
                    return float(val)
                except Exception:
                    continue
            # As a last resort, try to take the first numeric value in the row
            try:
                for v in row.tolist() if hasattr(row, 'tolist') else list(row):
                    try:
                        return float(v)
                    except Exception:
                        continue
            except Exception:
                pass
        return None

    chisq = _get_stat('chi2')
    dfree = _get_stat('DoF')
    pval = _get_stat('p-value')
    cfi = _get_stat('CFI')
    tli = _get_stat('TLI')
    srmr = _get_stat('SRMR')
    rmsea = _get_stat('RMSEA')
    chisq_df = float(chisq / dfree) if (chisq is not None and dfree not in (None, 0)) else None

    # Reliability metrics
    x = data.to_numpy(dtype=float)
    alpha = _cronbach_alpha(x)
    print(model.parameters)

    # Extract standardized solution to compute omega
    try:
        # semopy's inspect
        params = model.parameters
        # Compute standardized loadings: semopy provides 'lambdas' via inspect
        # We'll approximate using factor score regression approach via implied covariance.
        # Get implied covariance matrix Sigma and residuals Theta.
        Sigma = model.inspect('implied.cov')
        # Standardize so diagonal is 1
        d = np.sqrt(np.diag(Sigma))
        S_std = Sigma / np.outer(d, d)
        # For single factor, loadings can be approximated as first eigenvector of S_std scaled to communalities
        w, V = np.linalg.eigh(S_std)
        idx = np.argmax(w)
        load_std = np.sqrt(max(w[idx] - 1e-12, 0)) * V[:, idx]
        # Residual variances as 1 - lambda^2
        theta = 1.0 - np.clip(load_std**2, 0.0, 1.0)
        omega = _omega_total_std(load_std, theta)
    except Exception:
        omega = float('nan')

    res = {
        'chisq': chisq,
        'df': dfree,
        'pvalue': pval,
        'chisq_df': chisq_df,
        'cfi': cfi,
        'tli': tli,
        'srmr': srmr,
        'rmsea': rmsea,
        'alpha': alpha,
        'omega': omega,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame([res]).to_csv(save_path, index=False)

    return res


def run_cfa_python_multi(
    df: pd.DataFrame,
    factors: Dict[str, List[str]],
    estimator: str = "ML",
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run a multi-factor CFA using semopy.

    Parameters:
    - df: DataFrame with the items.
    - factors: dict mapping factor name -> list of item column names.
    - estimator: one of {'ML', 'WLS', 'DWLS'} depending on semopy support (default ML).
    - save_path: optional CSV path to save the results.

    Returns a dict with global fit indices and per-factor reliabilities:
      chisq, df, pvalue, chisq_df, cfi, tli, srmr, rmsea, alpha_<factor>, omega_<factor>
    """
    if not factors:
        raise CFAError("No factors specified for multi-factor CFA.")

    try:
        from semopy import Model
        from semopy import calc_stats
    except Exception as e:
        raise CFAError(
            "semopy is required for Python CFA. Please install it (pip install semopy)."
        ) from e

    # Validate and collect unique items
    all_items: List[str] = []
    for f, items in factors.items():
        if not items or len(items) < 2:
            raise CFAError(f"Factor '{f}' must have at least 2 items.")
        for c in items:
            if c not in df.columns:
                raise CFAError(f"Item '{c}' for factor '{f}' not found in DataFrame.")
        all_items.extend(items)
    # Deduplicate while preserving order
    seen = set()
    unique_items = []
    for c in all_items:
        if c not in seen:
            unique_items.append(c)
            seen.add(c)

    data = df[unique_items].copy()
    # Coerce to numeric and drop rows with any NA across all involved items
    for c in unique_items:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(axis=0, how="any").reset_index(drop=True)
    if data.empty:
        raise CFAError("No data left after cleaning NA values in CFA items.")

    # Build model string with one line per factor
    lines = []
    for f, items in factors.items():
        rhs = " + ".join(items)
        lines.append(f"{f} =~ {rhs}")
    model_desc = "\n".join(lines)

    model = Model(model_desc)

    try:
        model.fit(data, estimator=estimator)
    except TypeError:
        model.fit(data)

    try:
        stats = calc_stats(model, data)
    except TypeError:
        stats = calc_stats(model)

    def _get_stat(name: str) -> Optional[float]:
        for idx_name in (name, name.lower()):
            try:
                row = stats.loc[idx_name]
            except Exception:
                continue
            for col in ("Value", "value", "Stat", "stat"):
                try:
                    val = row[col]
                    return float(val)
                except Exception:
                    continue
            try:
                for v in row.tolist() if hasattr(row, "tolist") else list(row):
                    try:
                        return float(v)
                    except Exception:
                        continue
            except Exception:
                pass
        return None

    chisq = _get_stat("chi2")
    dfree = _get_stat("DoF")
    pval = _get_stat("p-value")
    cfi = _get_stat("CFI")
    tli = _get_stat("TLI")
    srmr = _get_stat("SRMR")
    rmsea = _get_stat("RMSEA")
    chisq_df = float(chisq / dfree) if (chisq is not None and dfree not in (None, 0)) else None

    # Per-factor reliability (Cronbach's alpha; omega left as NaN placeholder)
    res: Dict[str, float] = {
        "chisq": chisq,
        "df": dfree,
        "pvalue": pval,
        "chisq_df": chisq_df,
        "cfi": cfi,
        "tli": tli,
        "srmr": srmr,
        "rmsea": rmsea,
    }

    for f, items in factors.items():
        x = data[items].to_numpy(dtype=float)
        res[f"alpha_{f}"] = _cronbach_alpha(x)
        # Compute McDonald's omega (total) from item correlation matrix using a
        # one-factor approximation via the first eigenpair. Falls back to NaN on errors.
        try:
            # Compute correlation matrix of items
            R = np.corrcoef(x, rowvar=False)
            # Guard against numerical issues
            if not np.all(np.isfinite(R)) or R.shape[0] < 2:
                raise ValueError("invalid correlation matrix")
            # Ensure symmetry
            R = (R + R.T) / 2.0
            # Eigen-decomposition
            w, V = np.linalg.eigh(R)
            idx = int(np.argmax(w))
            lam1 = float(max(w[idx], 0.0))
            v1 = V[:, idx]
            # Standardized loadings approximation
            load_std = np.sqrt(max(lam1 - 1e-12, 0.0)) * v1
            # Unique variances (clipped to [0,1])
            theta = 1.0 - np.clip(load_std ** 2, 0.0, 1.0)
            omega_f = _omega_total_std(load_std, theta)
        except Exception:
            omega_f = float("nan")
        res[f"omega_{f}"] = omega_f

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame([res]).to_csv(save_path, index=False)

    return res
