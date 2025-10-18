from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def load_dataframe(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from a file path. Supports CSV and Excel by extension.
    - .csv/.txt: uses pandas.read_csv
    - .xls/.xlsx: uses pandas.read_excel (requires openpyxl or xlrd depending on version)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file extension: {ext}. Use CSV or Excel.")


essential_index_names = {"index", "Unnamed: 0"}


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Save DataFrame to CSV or Excel depending on extension."""
    ext = os.path.splitext(path)[1].lower()
    # avoid writing default index columns that often appear in notebooks
    df_to_save = df.copy()
    for col in list(df_to_save.columns):
        if col in essential_index_names:
            try:
                df_to_save = df_to_save.drop(columns=[col])
            except KeyError:
                pass
    if ext in {".csv", ".txt"}:
        df_to_save.to_csv(path, index=index)
        return
    if ext in {".xls", ".xlsx"}:
        df_to_save.to_excel(path, index=index)
        return
    raise ValueError(f"Unsupported file extension for saving: {ext}.")
