from __future__ import annotations

from typing import Literal

import pandas as pd

ColumnKind = Literal["NUMERIC", "CATEGORICAL"]

NUMERIC_UNIQUE_THRESHOLD = 12


def detect_column_type(
    series: pd.Series,
    unique_threshold: int = NUMERIC_UNIQUE_THRESHOLD,
) -> ColumnKind:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "CATEGORICAL"
    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return "CATEGORICAL"
    if pd.api.types.is_numeric_dtype(dtype):
        n_unique = series.nunique(dropna=True)
        if n_unique <= unique_threshold:
            return "CATEGORICAL"
        return "NUMERIC"
    return "CATEGORICAL"


def detect_column_types(
    df: pd.DataFrame,
    target: str,
    unique_threshold: int = NUMERIC_UNIQUE_THRESHOLD,
) -> dict[str, ColumnKind]:
    return {
        col: detect_column_type(df[col], unique_threshold)
        for col in df.columns
        if col != target
    }
