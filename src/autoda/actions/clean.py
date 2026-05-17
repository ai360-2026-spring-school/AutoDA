from __future__ import annotations

import functools
from typing import Any
import pandas as pd

from autoda.transformer import Transformer


# ---------------------------------------------------------------------------
# apply helpers (top-level, picklable via functools.partial)
# ---------------------------------------------------------------------------

def _apply_impute_missing(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "knn_imputer" in state:
        from sklearn.impute import KNNImputer  # noqa: F401
        imputer: KNNImputer = state["knn_imputer"]
        knn_cols: list[str] = [c for c in state["knn_cols"] if c in df.columns]
        if knn_cols:
            df[knn_cols] = imputer.transform(df[knn_cols])
    else:
        for col, fill_val in state["fills"].items():
            if col not in df.columns:
                continue
            df[col] = df[col].fillna(fill_val)
    return df


def _apply_drop_columns(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=state["columns"], errors="ignore")


def _apply_drop_duplicates(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    # no-op on test
    return df.copy()


def _apply_clip_outliers(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in state["bounds"].items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _apply_collapse_rare_categories(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col: str = state["column"]
    if col not in df.columns:
        return df
    rare_values = state["rare_values"]
    other_label = state["other_label"]
    df[col] = df[col].where(~df[col].isin(rare_values), other=other_label)
    return df


def _apply_cast_dtype(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col: str = state["column"]
    if col not in df.columns:
        return df
    dtype_str: str = state["dtype"]
    if dtype_str == "datetime64[ns]":
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        df[col] = df[col].astype(dtype_str)
    return df


# ---------------------------------------------------------------------------
# public action functions — return (df, Transformer, observation)
# ---------------------------------------------------------------------------

def impute_missing(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    strategy: str = "mean",
    fill_value: Any = None,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Fill NaNs in numeric / categorical columns with a learned fill value.

    Use when: profile_summary shows columns with non-zero missing_rate AND those columns look useful (don't bother imputing if you're about to drop).
    Effect: each column's NaNs become the train-fit fill (mean/median/mode/constant/knn). Same fill is re-applied to the test set.

    Args:
        columns: list[str]. Columns to impute. The target is silently skipped.
        strategy: str. One of "mean", "median", "mode", "constant", "knn". (default: "mean")
        fill_value: Any. Required only for strategy="constant".
    """
    valid = {"mean", "median", "mode", "constant", "knn"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {valid}, got {strategy!r}")

    df_out = df.copy()
    changed: list[str] = []

    if strategy == "knn":
        from sklearn.impute import KNNImputer
        knn_cols = [c for c in columns if c in df.columns and c != target and df[c].isna().any()]
        state: dict[str, Any]
        if knn_cols:
            imputer = KNNImputer()
            df_out[knn_cols] = imputer.fit_transform(df_out[knn_cols])
            changed = knn_cols
            state = {"knn_imputer": imputer, "knn_cols": knn_cols}
        else:
            state = {"knn_imputer": None, "knn_cols": []}
    else:
        fills: dict[str, Any] = {}
        for col in columns:
            if col not in df.columns or col == target:
                continue
            if df[col].isna().sum() == 0:
                continue
            if strategy == "mean":
                val = df[col].mean()
            elif strategy == "median":
                val = df[col].median()
            elif strategy == "mode":
                val = df[col].mode().iloc[0]
            else:  # constant
                val = fill_value
            fills[col] = val
            df_out[col] = df_out[col].fillna(val)
            changed.append(col)
        state = {"fills": fills}

    transformer = Transformer(
        operation="impute_missing",
        args={"columns": columns, "strategy": strategy, "fill_value": fill_value},
        state=state,
        apply=functools.partial(_apply_impute_missing, state),
    )
    return df_out, transformer, {
        "changed_columns": changed,
        "summary": f"imputed {len(changed)} column(s) with strategy={strategy!r}",
    }


def drop_columns(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Drop one or more feature columns.

    Use when: profile_summary lists them as constant, all-NaN, near-unique (likely an id), or suspected leakage; or feature importance is ~0.
    Effect: those columns disappear from train and test. Target column is silently protected.

    Args:
        columns: list[str]. Column names to drop. Missing names are silently skipped.
    """
    safe_columns = [c for c in columns if c != target and c in df.columns]
    df_out = df.drop(columns=safe_columns)
    state: dict[str, Any] = {"columns": safe_columns}
    transformer = Transformer(
        operation="drop_columns",
        args={"columns": columns},
        state=state,
        apply=functools.partial(_apply_drop_columns, state),
    )
    return df_out, transformer, {
        "changed_columns": safe_columns,
        "summary": f"dropped {len(safe_columns)} column(s)",
    }


def drop_duplicates(
    df: pd.DataFrame,
    target: str,
    subset: list[str] | None = None,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Remove duplicate rows from the training set only (identity on test).

    Use when: profile suggests many exact-duplicate rows that would bias CV.
    Effect: train shrinks; test/submission rows are NEVER dropped.

    Args:
        subset: list[str] | None. Subset of columns to consider for the duplicate check. (default: None — all columns)
    """
    before = len(df)
    df_out = df.drop_duplicates(subset=subset)
    removed = before - len(df_out)
    state: dict[str, Any] = {}
    transformer = Transformer(
        operation="drop_duplicates",
        args={"subset": subset},
        state=state,
        apply=functools.partial(_apply_drop_duplicates, state),
    )
    return df_out, transformer, {
        "changed_columns": [],
        "summary": f"removed {removed} duplicate row(s); apply is identity on test",
    }


def clip_outliers(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    method: str = "iqr",
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Cap extreme values in numeric columns.

    Use when: a numeric column has a few huge / tiny values that destabilise CV (see high_skew_columns in profile_summary).
    Effect: values are clamped to learned [lo, hi] bounds; same bounds re-applied to test.

    Args:
        columns: list[str]. Numeric columns to clip. Non-numeric / target are silently skipped.
        method: str. "iqr" uses Q1−1.5·IQR / Q3+1.5·IQR; "quantile" uses the lower/upper kwargs as quantile fractions. (default: "iqr")
        lower: float | None. Lower quantile for method="quantile". (default: 0.01)
        upper: float | None. Upper quantile for method="quantile". (default: 0.99)
    """
    if method not in ("iqr", "quantile"):
        raise ValueError(f"method must be 'iqr' or 'quantile', got {method!r}")

    df_out = df.copy()
    bounds: dict[str, list[float]] = {}

    for col in columns:
        if col not in df.columns or col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if method == "iqr":
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
        else:
            lo = float(df[col].quantile(lower if lower is not None else 0.01))
            hi = float(df[col].quantile(upper if upper is not None else 0.99))
        bounds[col] = [lo, hi]
        df_out[col] = df_out[col].clip(lo, hi)

    state: dict[str, Any] = {"bounds": bounds}
    transformer = Transformer(
        operation="clip_outliers",
        args={"columns": columns, "method": method, "lower": lower, "upper": upper},
        state=state,
        apply=functools.partial(_apply_clip_outliers, state),
    )
    return df_out, transformer, {
        "changed_columns": list(bounds.keys()),
        "summary": f"clipped outliers in {len(bounds)} column(s) using {method!r}",
    }


def collapse_rare_categories(
    df: pd.DataFrame,
    target: str,
    column: str,
    min_freq: int | float = 50,
    other_label: str = "OTHER",
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Replace rare category values in one column with a single bucket label.

    Use when: a categorical column has many low-frequency values (long tail) that produce noisy splits.
    Effect: values seen fewer than min_freq times become other_label; the same rare set is applied on test.

    Args:
        column: str. The categorical column to collapse.
        min_freq: int | float<1. Absolute count threshold (int) OR fraction of rows (float < 1). (default: 50)
        other_label: str. Label to use for the collapsed bucket. (default: "OTHER")
    """
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot collapse the target column")

    df_out = df.copy()
    vc = df[column].value_counts(dropna=False)

    if isinstance(min_freq, float) and min_freq < 1:
        threshold = int(min_freq * len(df))
    else:
        threshold = int(min_freq)

    rare = list(vc[vc < threshold].index)
    df_out[column] = df_out[column].where(~df_out[column].isin(rare), other=other_label)

    state: dict[str, Any] = {"column": column, "rare_values": rare, "other_label": other_label}
    transformer = Transformer(
        operation="collapse_rare_categories",
        args={"column": column, "min_freq": min_freq, "other_label": other_label},
        state=state,
        apply=functools.partial(_apply_collapse_rare_categories, state),
    )
    return df_out, transformer, {
        "changed_columns": [column],
        "summary": f"collapsed {len(rare)} rare categories in {column!r} (threshold={threshold})",
    }


def cast_dtype(
    df: pd.DataFrame,
    target: str,
    column: str,
    dtype: str,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Cast one column to a different dtype.

    Use when: a column is stored as the wrong type (e.g. numeric stored as object, or a date stored as string).
    Effect: column dtype changes on train and test.

    Args:
        column: str. Column to cast.
        dtype: str. One of "int", "float", "category", "datetime".
    """
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot cast the target column")

    dtype_map = {
        "int": "int64",
        "float": "float64",
        "category": "category",
        "datetime": "datetime64[ns]",
    }
    if dtype not in dtype_map:
        raise ValueError(f"dtype must be one of {list(dtype_map)}, got {dtype!r}")

    internal_dtype = dtype_map[dtype]
    df_out = df.copy()
    if dtype == "datetime":
        df_out[column] = pd.to_datetime(df_out[column], errors="coerce")
    else:
        df_out[column] = df_out[column].astype(internal_dtype)

    state: dict[str, Any] = {"column": column, "dtype": internal_dtype}
    transformer = Transformer(
        operation="cast_dtype",
        args={"column": column, "dtype": dtype},
        state=state,
        apply=functools.partial(_apply_cast_dtype, state),
    )
    return df_out, transformer, {
        "changed_columns": [column],
        "summary": f"cast {column!r} to {dtype!r}",
    }
