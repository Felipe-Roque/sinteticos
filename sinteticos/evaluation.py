from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def hellinger_distance(p, q) -> float:
    """Compute Hellinger distance between two discrete distributions.
    Inputs can be sequences of counts; they will be normalized.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ps = p / (p.sum() if p.sum() != 0 else 1.0)
    qs = q / (q.sum() if q.sum() != 0 else 1.0)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(ps) - np.sqrt(qs)) ** 2)))


def evaluate_hellinger_by_column(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Hellinger distance per column between two dataframes with the same columns.
    Returns a DataFrame with index=column names and column 'Hellinger'.
    """
    distances: Dict[str, float] = {}
    for col in df_real.columns:
        real_counts = df_real[col].value_counts().sort_index()
        synth_counts = df_synth[col].value_counts().sort_index()
        all_idx = sorted(set(real_counts.index).union(set(synth_counts.index)))
        real_aligned = [real_counts.get(i, 0) for i in all_idx]
        synth_aligned = [synth_counts.get(i, 0) for i in all_idx]
        distances[col] = hellinger_distance(real_aligned, synth_aligned)
    out = pd.DataFrame.from_dict(distances, orient="index", columns=["Hellinger"]).sort_values(
        by="Hellinger", ascending=False
    )
    return out
