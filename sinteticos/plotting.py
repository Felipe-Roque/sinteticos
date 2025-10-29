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


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    annot: bool = False,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    mask_upper: bool = False,
    numeric_only: bool = True,
    figsize: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
) -> pd.DataFrame:
    """Compute and plot a correlation matrix heatmap.

    Parameters:
    - df: input DataFrame
    - method: 'spearman', 'kendall', or 'pearson'
    - annot: whether to show correlation values
    - cmap: matplotlib colormap name
    - title: optional plot title
    - mask_upper: if True, mask the upper triangle for a cleaner look
    - numeric_only: consider only numeric columns
    - figsize: optional figure size (w, h)
    - save_path: if provided, save the plot image to this path
    - dpi: save resolution when save_path is given

    Returns the correlation DataFrame used for plotting.
    """
    # Lazy import to avoid circular dependency and keep plotting module light
    from .evaluation import compute_correlation

    corr = compute_correlation(df, method=method, numeric_only=numeric_only)
    if corr.empty:
        # Nothing to plot
        return corr

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    if figsize is None:
        n = len(corr.columns)
        figsize = (max(6, 0.5 * n + 4), max(5, 0.5 * n + 3))

    plt.figure(figsize=figsize)

    mask = None
    if mask_upper:
        mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                mask.iloc[i, j] = True

    ax = sns.heatmap(corr, cmap=cmap, annot=annot, fmt='.2f' if annot else '', square=True, mask=mask, cbar=True, vmin=-1, vmax=1)
    ax.set_title(title or f"Matriz de Correlação ({method.capitalize()})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

    return corr
