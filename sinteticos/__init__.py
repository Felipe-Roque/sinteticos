"""
Sinteticos: Lightweight toolkit to generate and evaluate synthetic tabular data

Public API
- io_utils.load_dataframe
- io_utils.save_dataframe
- generator.preprocess_dataframe
- generator.train_model
- generator.generate_synthetic
- evaluation.hellinger_distance
- evaluation.evaluate_hellinger_by_column
- evaluation.compute_correlation
- evaluation.spearman_correlation
- evaluation.kendall_correlation
- plotting.plot_hellinger_bar
- plotting.compare_histograms
- plotting.plot_correlation_matrix
"""

from .io_utils import load_dataframe, save_dataframe
from .generator import preprocess_dataframe, train_model, generate_synthetic
from .evaluation import (
    hellinger_distance,
    evaluate_hellinger_by_column,
    compute_correlation,
    spearman_correlation,
    kendall_correlation,
)
from .plotting import plot_hellinger_bar, compare_histograms, plot_correlation_matrix

__all__ = [
    "load_dataframe",
    "save_dataframe",
    "preprocess_dataframe",
    "train_model",
    "generate_synthetic",
    "hellinger_distance",
    "evaluate_hellinger_by_column",
    "compute_correlation",
    "spearman_correlation",
    "kendall_correlation",
    "plot_hellinger_bar",
    "compare_histograms",
    "plot_correlation_matrix",
]
