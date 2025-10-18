from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_hellinger_bar(df_hellinger: pd.DataFrame, title: str = "Distância de Hellinger por variável") -> None:
    """Bar plot for Hellinger distances per column."""
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    plt.figure(figsize=(10, 8))
    # When df has index with names and column 'Hellinger'
    sns.barplot(x='Hellinger', y=df_hellinger.index, data=df_hellinger, palette='viridis')
    plt.title(title)
    plt.xlabel("Distância de Hellinger")
    plt.ylabel("Variável")
    plt.tight_layout()
    plt.show()


def compare_histograms(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_cols: int = 3,
    bins: Optional[Sequence[float]] = None,
) -> None:
    """
    Compare histograms of selected columns from real vs synthetic DataFrames.
    Defaults replicate the notebook behavior for 1..5 categorical-like bins.
    """
    cols = list(columns) if columns is not None else list(df_real.columns)
    n = len(cols)
    n_rows = (n + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    if bins is None:
        bins = [1, 2, 3, 4, 5, 6]

    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df_real[col], bins=bins, stat='density', kde=False, label='Original', color='blue', alpha=0.6)
        sns.histplot(df_synth[col], bins=bins, stat='density', kde=False, label='Sintético', color='orange', alpha=0.6)
        plt.title(str(col))
        try:
            plt.xticks([1, 2, 3, 4, 5])
        except Exception:
            pass
        plt.legend()

    plt.tight_layout()
    plt.show()
