from __future__ import annotations

from typing import Any

import pandas as pd


def groupby_agg(
    df: pd.DataFrame,
    target: str,
    *,
    col: str,
    by: str,
    agg_fn: str = "mean",
) -> dict[str, Any]:
    """Info-tool: compute group-level aggregate of `col` grouped by `by`.

    Use when: you want to see how a numeric feature varies across groups before deciding on group_aggregate transform.

    Args:
        col: str. Numeric column to aggregate.
        by: str. Grouping column.
        agg_fn: str. One of mean/median/std/count/min/max. (default: mean)
    """
    valid_fns = {"mean", "median", "std", "count", "min", "max"}
    if agg_fn not in valid_fns:
        return {"error": f"agg_fn must be one of {valid_fns}"}
    if col not in df.columns:
        return {"error": f"column not found: {col!r}"}
    if by not in df.columns:
        return {"error": f"column not found: {by!r}"}
    try:
        result = df.groupby(by, dropna=False)[col].agg(agg_fn)
        return {
            "summary": f"{agg_fn}({col}) grouped by {by}",
            "result": {str(k): round(float(v), 6) for k, v in result.items() if pd.notna(v)},
        }
    except Exception as e:
        return {"error": repr(e)}


def value_counts(
    df: pd.DataFrame,
    target: str,
    *,
    col: str,
    top_k: int = 20,
) -> dict[str, Any]:
    """Info-tool: show value frequency distribution for a column.

    Args:
        col: str. Column to inspect.
        top_k: int. How many top values to return. (default: 20)
    """
    if col not in df.columns:
        return {"error": f"column not found: {col!r}"}
    try:
        vc = df[col].value_counts(dropna=False).head(top_k)
        return {
            "summary": f"value_counts({col}), top {top_k}",
            "n_unique": int(df[col].nunique(dropna=True)),
            "counts": {str(k): int(v) for k, v in vc.items()},
        }
    except Exception as e:
        return {"error": repr(e)}


def correlation_matrix(
    df: pd.DataFrame,
    target: str,
    *,
    col1: str,
    col2: str,
) -> dict[str, Any]:
    """Info-tool: compute Pearson correlation between two columns.

    Args:
        col1: str. First numeric column.
        col2: str. Second numeric column (can be the target).
    """
    for c in (col1, col2):
        if c not in df.columns:
            return {"error": f"column not found: {c!r}"}
        if not pd.api.types.is_numeric_dtype(df[c]):
            return {"error": f"column {c!r} is not numeric"}
    try:
        r = float(df[col1].corr(df[col2]))
        return {
            "summary": f"pearson({col1}, {col2})",
            "col1": col1,
            "col2": col2,
            "pearson": round(r, 4),
        }
    except Exception as e:
        return {"error": repr(e)}


def describe_column(
    df: pd.DataFrame,
    target: str,
    *,
    col: str,
) -> dict[str, Any]:
    """Info-tool: descriptive statistics for a single column.

    Args:
        col: str. Column to describe.
    """
    if col not in df.columns:
        return {"error": f"column not found: {col!r}"}
    try:
        s = df[col]
        result: dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe().round(4).to_dict()
            result["stats"] = {k: float(v) for k, v in desc.items() if pd.notna(v)}
            result["skew"] = round(float(s.skew()), 4)
        else:
            vc = s.value_counts(dropna=False).head(10)
            result["top_values"] = {str(k): int(v) for k, v in vc.items()}
        return {"summary": f"describe({col})", "result": result}
    except Exception as e:
        return {"error": repr(e)}


def view_precomputed_stats(
    df: pd.DataFrame,
    target: str,
    *,
    target_correlation_stats: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Info-tool: return the precomputed correlation/class-mean stats computed after preprocessing.

    Args:
        (no additional args; stats are loaded from state automatically)
    """
    if not target_correlation_stats:
        return {"summary": "precomputed stats", "result": {}, "note": "no stats available yet"}
    return {"summary": "precomputed target correlation stats", "result": target_correlation_stats}


def view_long_summary(
    df: pd.DataFrame,
    target: str,
    *,
    long_description_summary: str | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Info-tool: view the full description summary generated at start.

    Args:
        (no additional args; summary is loaded from state automatically)
    """
    if not long_description_summary:
        return {"summary": "long description summary", "result": "(not available — no description was provided)"}
    return {"summary": "long description summary", "result": long_description_summary}
