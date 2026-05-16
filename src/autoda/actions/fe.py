from typing import Any
import numpy as np
import pandas as pd


def expand_datetime(
    df: pd.DataFrame,
    target: str,
    column: str,
    parts: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")

    valid_parts = {"year", "month", "dow", "hour", "is_weekend"}
    parts = parts or list(valid_parts)
    invalid = set(parts) - valid_parts
    if invalid:
        raise ValueError(f"invalid parts: {invalid}")

    df = df.copy()
    dt = pd.to_datetime(df[column], errors="coerce")
    added: list[str] = []

    if "year" in parts:
        df[f"{column}__year"] = dt.dt.year
        added.append(f"{column}__year")
    if "month" in parts:
        df[f"{column}__month"] = dt.dt.month
        added.append(f"{column}__month")
    if "dow" in parts:
        df[f"{column}__dow"] = dt.dt.dayofweek
        added.append(f"{column}__dow")
    if "hour" in parts:
        df[f"{column}__hour"] = dt.dt.hour
        added.append(f"{column}__hour")
    if "is_weekend" in parts:
        df[f"{column}__is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        added.append(f"{column}__is_weekend")

    return df, {"changed_columns": added, "summary": f"expanded {column!r} into {added}"}


def binarize_missing(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    added: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        new_col = f"{col}__isna"
        df[new_col] = df[col].isna().astype(int)
        added.append(new_col)

    return df, {"changed_columns": added, "summary": f"added {len(added)} missing-indicator column(s)"}


def log_transform(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    plus_one: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot transform the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")

        vals = df[col] + 1 if plus_one else df[col]
        if (vals <= 0).any():
            raise ValueError(f"column {col!r} has non-positive values after shift; cannot log-transform")

        df[col] = np.log(vals)
        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"log-transformed {len(changed)} column(s)"}


def bin_numeric(
    df: pd.DataFrame,
    target: str,
    column: str,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot bin the target column")
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"column {column!r} is not numeric")

    df = df.copy()
    new_col = f"{column}__bin"

    if strategy == "quantile":
        df[new_col] = pd.qcut(df[column], q=n_bins, labels=False, duplicates="drop")
    else:
        df[new_col] = pd.cut(df[column], bins=n_bins, labels=False)

    return df, {"changed_columns": [new_col], "summary": f"binned {column!r} into {n_bins} bins ({strategy})"}


def interaction(
    df: pd.DataFrame,
    target: str,
    cols: list[str],
    op: str = "mul",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(cols) != 2:
        raise ValueError("cols must have exactly 2 elements")
    a, b = cols
    for c in (a, b):
        if c not in df.columns:
            raise ValueError(f"column not found: {c!r}")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c!r} is not numeric")

    if op not in ("mul", "div", "add", "sub"):
        raise ValueError(f"op must be one of mul/div/add/sub, got {op!r}")

    df = df.copy()
    new_col = f"{a}__{op}__{b}"

    if op == "mul":
        df[new_col] = df[a] * df[b]
    elif op == "div":
        df[new_col] = df[a] / df[b].replace(0, float("nan"))
    elif op == "add":
        df[new_col] = df[a] + df[b]
    else:
        df[new_col] = df[a] - df[b]

    return df, {"changed_columns": [new_col], "summary": f"created interaction {new_col!r}"}


def group_aggregate(
    df: pd.DataFrame,
    target: str,
    by: str,
    value: str,
    agg: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for col in (by, value):
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
    if agg not in ("mean", "median", "std", "count"):
        raise ValueError(f"agg must be one of mean/median/std/count, got {agg!r}")

    df = df.copy()
    new_col = f"{value}__{agg}_by_{by}"
    mapping = df.groupby(by, dropna=False)[value].agg(agg)
    df[new_col] = df[by].map(mapping)

    return df, {"changed_columns": [new_col], "summary": f"group-aggregated {value!r} by {by!r} ({agg}) -> {new_col!r}"}
