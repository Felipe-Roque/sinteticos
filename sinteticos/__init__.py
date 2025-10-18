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
- plotting.plot_hellinger_bar
- plotting.compare_histograms
"""

from .io_utils import load_dataframe, save_dataframe
from .generator import preprocess_dataframe, train_model, generate_synthetic
from .evaluation import hellinger_distance, evaluate_hellinger_by_column
from .plotting import plot_hellinger_bar, compare_histograms

__all__ = [
    "load_dataframe",
    "save_dataframe",
    "preprocess_dataframe",
    "train_model",
    "generate_synthetic",
    "hellinger_distance",
    "evaluate_hellinger_by_column",
    "plot_hellinger_bar",
    "compare_histograms",
]
