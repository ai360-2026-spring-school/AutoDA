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
    if "day" in parts:
        df[f"{col}__day"] = dt.dt.day
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
    elif op == "sub":
        df[new_col] = df[a] - df[b]
    elif op == "abs":  # |A - B|
        df[new_col] = (df[a] - df[b]).abs()
    elif op == "sign":  # sign(A - B) → -1, 0, +1
        df[new_col] = np.sign(df[a] - df[b])
    elif op == "max":
        df[new_col] = df[[a, b]].max(axis=1)
    elif op == "min":
        df[new_col] = df[[a, b]].min(axis=1)
    else:
        df[new_col] = df[a] - df[b]
    return df


def _apply_power_transform(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col: str = state["column"]
    power: float = state["power"]
    new_col: str = state["new_col"]
    if col not in df.columns:
        return df
    vals = df[col].clip(lower=0)  # non-negative for fractional powers
    df[new_col] = np.power(vals, power)
    return df


def _apply_multi_interaction(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols: list[str] = state["cols"]
    op: str = state["op"]
    new_col: str = state["new_col"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return df
    result = df[cols[0]].copy()
    for c in cols[1:]:
        if op == "mul":
            result = result * df[c]
        elif op == "add":
            result = result + df[c]
        elif op == "sub":
            result = result - df[c]
    df[new_col] = result
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

    valid_parts = {"year", "month", "day", "dow", "hour", "is_weekend"}
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
    """Create a pairwise interaction feature between two numeric columns.

    Use when: domain reason suggests a product/ratio/difference carries signal CatBoost won't trivially derive.
    Effect: adds <a>__<op>__<b> column.

    Args:
        cols: list[str]. Exactly two numeric columns [a, b].
        op: str. One of "mul"(A×B), "div"(A/B), "add"(A+B), "sub"(A-B), "abs"(|A-B|), "sign"(sign(A-B)→-1/0/1), "max", "min". (default: "mul")
    """
    _VALID_OPS = ("mul", "div", "add", "sub", "abs", "sign", "max", "min")
    _OP_ALIASES = {
        "multiply": "mul", "product": "mul", "prod": "mul", "times": "mul",
        "divide": "div", "ratio": "div", "divided_by": "div",
        "plus": "add", "sum": "add",
        "minus": "sub", "subtract": "sub", "diff": "sub", "difference": "sub",
        "abs_diff": "abs", "absolute": "abs", "absdiff": "abs",
        "sgn": "sign", "signum": "sign",
        "maximum": "max", "minimum": "min",
        "sqrt": None,  # handled below
    }
    if len(cols) != 2:
        raise ValueError("cols must have exactly 2 elements")
    a, b = cols
    for c in (a, b):
        if c not in df.columns:
            raise ValueError(f"column not found: {c!r}")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c!r} is not numeric")
    op = _OP_ALIASES.get(op.lower(), op)
    if op not in _VALID_OPS:
        raise ValueError(f"op must be one of {'/'.join(_VALID_OPS)}, got {op!r}")

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


def power_transform(
    df: pd.DataFrame,
    target: str,
    column: str,
    power: float = 0.5,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Raise a numeric column to a power (square root, square, cube root, etc.).

    Use when: a physical/domain relationship is polynomial (e.g. strength ∝ cement^0.7, energy ∝ v²).
    Effect: adds <col>__pow_<power> column; original column kept; negative values clipped to 0 for fractional powers.

    Args:
        column: str. Numeric column to transform.
        power: float. Exponent (0.5=sqrt, 2=square, 0.333=cube root, -1=reciprocal). (default: 0.5)
    """
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot transform the target column")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"column {column!r} is not numeric")

    power_str = str(power).replace(".", "_")
    new_col = f"{column}__pow_{power_str}"
    state: dict[str, Any] = {"column": column, "power": power, "new_col": new_col}
    df_out = _apply_power_transform(state, df)

    transformer = Transformer(
        operation="power_transform",
        args={"column": column, "power": power},
        state=state,
        apply=functools.partial(_apply_power_transform, state),
    )
    return df_out, transformer, {
        "changed_columns": [new_col],
        "summary": f"power-transformed {column!r}^{power} → {new_col!r}",
    }


def multi_interaction(
    df: pd.DataFrame,
    target: str,
    cols: list[str],
    op: str = "mul",
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Chain an operation across 3 or more numeric columns (A op B op C …).

    Use when: a triple (or higher-order) combination carries signal — e.g. smoker×bmi×age for insurance costs, or left_weight×left_distance×right_weight for balance physics.
    Effect: adds a single result column named <a>__<op>__<b>__<op>__<c>.

    Args:
        cols: list[str]. 3+ numeric columns to combine.
        op: str. One of "mul" (product) or "add" (sum). (default: "mul")
    """
    if len(cols) < 3:
        raise ValueError("cols must have at least 3 elements; use interaction() for pairs")
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"column not found: {c!r}")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c!r} is not numeric")
    if op not in ("mul", "add", "sub"):
        raise ValueError(f"op must be one of mul/add/sub, got {op!r}")

    new_col = (f"__{op}__").join(c[:15] for c in cols)
    state: dict[str, Any] = {"cols": list(cols), "op": op, "new_col": new_col}
    df_out = _apply_multi_interaction(state, df)

    transformer = Transformer(
        operation="multi_interaction",
        args={"cols": list(cols), "op": op},
        state=state,
        apply=functools.partial(_apply_multi_interaction, state),
    )
    return df_out, transformer, {
        "changed_columns": [new_col],
        "summary": f"multi-interaction {op.upper()} of {cols} → {new_col!r}",
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
        agg: str. One of "mean", "median", "std", "count", "min", "max". (default: "mean")
    """
    if by not in df.columns:
        raise ValueError(f"column not found: {by!r}")
    if value not in df.columns:
        raise ValueError(f"column not found: {value!r}")
    # Normalize common aliases
    _agg_aliases = {"average": "mean", "avg": "mean", "stddev": "std", "variance": "std",
                    "cnt": "count", "n": "count", "minimum": "min", "maximum": "max"}
    agg = _agg_aliases.get(agg.lower(), agg)
    if agg not in ("mean", "median", "std", "count", "min", "max"):
        raise ValueError(f"agg must be one of mean/median/std/count/min/max, got {agg!r}")

    new_col = f"{value}__{agg}_by_{by}"
    mapping_series = df.groupby(by, dropna=False)[value].agg(agg)
    mapping = {k: float(v) for k, v in mapping_series.items()}

    # compute overall default (used as fallback for unseen groups)
    defaults = {
        "mean": df[value].mean, "median": df[value].median,
        "std": df[value].std, "count": df[value].count,
        "min": df[value].min, "max": df[value].max,
    }
    default = float(defaults[agg]())

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
