from __future__ import annotations

import functools
from typing import Any
import numpy as np
import pandas as pd

from autoda.transformer import Transformer


# ---------------------------------------------------------------------------
# apply helpers (top-level, picklable)
# ---------------------------------------------------------------------------

def _apply_expand_datetime(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col: str = state["column"]
    parts: list[str] = state["parts"]
    if col not in df.columns:
        return df
    dt = pd.to_datetime(df[col], errors="coerce")
    if "year" in parts:
        df[f"{col}__year"] = dt.dt.year
    if "month" in parts:
        df[f"{col}__month"] = dt.dt.month
    if "dow" in parts:
        df[f"{col}__dow"] = dt.dt.dayofweek
    if "hour" in parts:
        df[f"{col}__hour"] = dt.dt.hour
    if "is_weekend" in parts:
        df[f"{col}__is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return df


def _apply_binarize_missing(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in state["columns"]:
        if col in df.columns:
            df[f"{col}__isna"] = df[col].isna().astype(int)
    return df


def _apply_log_transform(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    plus_one: bool = state["plus_one"]
    for col in state["columns"]:
        if col not in df.columns:
            continue
        vals = df[col] + 1 if plus_one else df[col]
        # clip non-positive values to a small positive number instead of raising
        vals = vals.clip(lower=1e-9)
        df[col] = np.log(vals)
    return df


def _apply_bin_numeric(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col: str = state["column"]
    bin_edges: list[float] = state["bin_edges"]
    new_col: str = state["new_col"]
    if col not in df.columns:
        return df
    df[new_col] = pd.cut(df[col], bins=bin_edges, labels=False, include_lowest=True)
    return df


def _apply_interaction(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    a, b = state["cols"]
    op: str = state["op"]
    new_col: str = state["new_col"]
    if a not in df.columns or b not in df.columns:
        return df
    if op == "mul":
        df[new_col] = df[a] * df[b]
    elif op == "div":
        df[new_col] = df[a] / df[b].replace(0, float("nan"))
    elif op == "add":
        df[new_col] = df[a] + df[b]
    else:
        df[new_col] = df[a] - df[b]
    return df


def _apply_group_aggregate(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by: str = state["by"]
    new_col: str = state["new_col"]
    mapping: dict = state["mapping"]
    default: float = state["default"]
    # value column may be absent on test (it's the aggregation source)
    # we just map from the by column using precomputed mapping
    if by not in df.columns:
        return df
    df[new_col] = df[by].map(mapping).fillna(default)
    return df


# ---------------------------------------------------------------------------
# public action functions — return (df, Transformer, observation)
# ---------------------------------------------------------------------------

def expand_datetime(
    df: pd.DataFrame,
    target: str,
    column: str,
    parts: list[str] | None = None,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Extract components from a datetime column into separate numeric features.

    Use when: profile_summary lists the column under datetime_columns and the timestamp likely carries seasonal signal.
    Effect: adds columns like <col>__year, <col>__month, <col>__dow, etc.; original column is kept.

    Args:
        column: str. Datetime-typed (or coercible) column.
        parts: list[str] | None. Subset of {"year", "month", "dow", "hour", "is_weekend"}. (default: all of them)
    """
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")

    valid_parts = {"year", "month", "dow", "hour", "is_weekend"}
    parts = parts or list(valid_parts)
    invalid = set(parts) - valid_parts
    if invalid:
        raise ValueError(f"invalid parts: {invalid}")

    state: dict[str, Any] = {"column": column, "parts": parts}
    df_out = _apply_expand_datetime(state, df)
    added = [c for c in df_out.columns if c not in df.columns]

    transformer = Transformer(
        operation="expand_datetime",
        args={"column": column, "parts": parts},
        state=state,
        apply=functools.partial(_apply_expand_datetime, state),
    )
    return df_out, transformer, {
        "changed_columns": added,
        "summary": f"expanded {column!r} into {added}",
    }


def binarize_missing(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Add a 0/1 indicator column for each input column's missingness pattern.

    Use when: missingness itself is informative (e.g. "no Cabin" predicts Survival).
    Effect: adds <col>__isna columns of 0/1; original column is untouched.

    Args:
        columns: list[str]. Columns to add missing-indicators for.
    """
    state: dict[str, Any] = {"columns": columns}
    df_out = _apply_binarize_missing(state, df)
    added = [f"{c}__isna" for c in columns if c in df.columns]

    transformer = Transformer(
        operation="binarize_missing",
        args={"columns": columns},
        state=state,
        apply=functools.partial(_apply_binarize_missing, state),
    )
    return df_out, transformer, {
        "changed_columns": added,
        "summary": f"added {len(added)} missing-indicator column(s)",
    }


def log_transform(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    plus_one: bool = True,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Apply natural log to numeric columns to reduce heavy right-skew.

    Use when: a numeric column appears in high_skew_columns (|skew|>1) and all values are non-negative.
    Effect: each column is replaced with log(x) or log(x+1).

    Args:
        columns: list[str]. Numeric columns to log-transform.
        plus_one: bool. Use log(x+1) so zeros are safe. Disable only when you know x>0 strictly. (default: True)
    """
    valid_cols = []
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot transform the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")
        valid_cols.append(col)

    state: dict[str, Any] = {"columns": valid_cols, "plus_one": plus_one}
    df_out = _apply_log_transform(state, df)

    transformer = Transformer(
        operation="log_transform",
        args={"columns": columns, "plus_one": plus_one},
        state=state,
        apply=functools.partial(_apply_log_transform, state),
    )
    return df_out, transformer, {
        "changed_columns": valid_cols,
        "summary": f"log-transformed {len(valid_cols)} column(s)",
    }


def bin_numeric(
    df: pd.DataFrame,
    target: str,
    column: str,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Discretise a numeric column into bin indices.

    Use when: the column has a non-linear effect that tree splits aren't capturing, or it should be treated as categorical.
    Effect: adds <col>__bin column with integer bin indices; original column kept.

    Args:
        column: str. Numeric column to bin.
        n_bins: int. Number of bins. (default: 5)
        strategy: str. "quantile" (equal frequency) or "uniform" (equal width). (default: "quantile")
    """
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot bin the target column")
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"column {column!r} is not numeric")

    new_col = f"{column}__bin"
    if strategy == "quantile":
        _, bin_edges = pd.qcut(df[column], q=n_bins, retbins=True, duplicates="drop")
    else:
        _, bin_edges = pd.cut(df[column], bins=n_bins, retbins=True)

    bin_edges_list = [float(e) for e in bin_edges]
    state: dict[str, Any] = {"column": column, "bin_edges": bin_edges_list, "new_col": new_col}
    df_out = _apply_bin_numeric(state, df)

    transformer = Transformer(
        operation="bin_numeric",
        args={"column": column, "n_bins": n_bins, "strategy": strategy},
        state=state,
        apply=functools.partial(_apply_bin_numeric, state),
    )
    return df_out, transformer, {
        "changed_columns": [new_col],
        "summary": f"binned {column!r} into bins ({strategy}); new_col={new_col!r}",
    }


def interaction(
    df: pd.DataFrame,
    target: str,
    cols: list[str],
    op: str = "mul",
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Create a single pairwise interaction feature between two numeric columns.

    Use when: domain reason or insight ledger suggests a product/ratio carries signal CatBoost won't trivially derive.
    Effect: adds <a>__<op>__<b> column.

    Args:
        cols: list[str]. Exactly two numeric columns [a, b].
        op: str. One of "mul", "div", "add", "sub". (default: "mul")
    """
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

    new_col = f"{a}__{op}__{b}"
    state: dict[str, Any] = {"cols": [a, b], "op": op, "new_col": new_col}
    df_out = _apply_interaction(state, df)

    transformer = Transformer(
        operation="interaction",
        args={"cols": cols, "op": op},
        state=state,
        apply=functools.partial(_apply_interaction, state),
    )
    return df_out, transformer, {
        "changed_columns": [new_col],
        "summary": f"created interaction {new_col!r}",
    }


def group_aggregate(
    df: pd.DataFrame,
    target: str,
    by: str,
    value: str,
    agg: str = "mean",
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Add a group-level statistic of `value` joined back per row by `by`.

    Use when: a categorical column has meaningful group statistics (e.g. mean fare per port).
    Effect: adds <value>__<agg>_by_<by>; mapping is fit on train and re-used on test (unseen keys → train-wide aggregate).

    Args:
        by: str. Grouping column.
        value: str. Numeric column to aggregate.
        agg: str. One of "mean", "median", "std", "count". (default: "mean")
    """
    if by not in df.columns:
        raise ValueError(f"column not found: {by!r}")
    if value not in df.columns:
        raise ValueError(f"column not found: {value!r}")
    if agg not in ("mean", "median", "std", "count"):
        raise ValueError(f"agg must be one of mean/median/std/count, got {agg!r}")

    new_col = f"{value}__{agg}_by_{by}"
    mapping_series = df.groupby(by, dropna=False)[value].agg(agg)
    mapping = {k: float(v) for k, v in mapping_series.items()}

    # compute overall default (used as fallback for unseen groups)
    if agg == "mean":
        default = float(df[value].mean())
    elif agg == "median":
        default = float(df[value].median())
    elif agg == "std":
        default = float(df[value].std())
    else:  # count
        default = float(df[value].count())

    state: dict[str, Any] = {
        "by": by,
        "value": value,
        "new_col": new_col,
        "mapping": mapping,
        "default": default,
    }
    df_out = _apply_group_aggregate(state, df)

    transformer = Transformer(
        operation="group_aggregate",
        args={"by": by, "value": value, "agg": agg},
        state=state,
        apply=functools.partial(_apply_group_aggregate, state),
    )
    return df_out, transformer, {
        "changed_columns": [new_col],
        "summary": f"group-aggregated {value!r} by {by!r} ({agg}) -> {new_col!r}",
    }
