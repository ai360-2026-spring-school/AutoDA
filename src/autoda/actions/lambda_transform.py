from __future__ import annotations

import re
import functools
from typing import Any

import numpy as np
import pandas as pd

from ..transformer import Transformer

_BANNED = re.compile(
    r'\b(import|exec|eval|open|__\w*|os|sys|subprocess|globals|locals|getattr|setattr|delattr|vars|dir|type|object|compile|__class__|__bases__|__mro__)\b'
)

_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "len": len, "int": int, "float": float, "str": str, "bool": bool,
    "True": True, "False": False, "None": None,
    "sum": sum, "zip": zip, "enumerate": enumerate, "range": range,
}

_IDENT_RE = re.compile(r'[^a-zA-Z0-9_]')


def _sanitize_varname(name: str) -> str:
    """Convert column name to a valid Python identifier."""
    s = _IDENT_RE.sub('_', name).strip('_')
    # Avoid leading digit
    if s and s[0].isdigit():
        s = '_' + s
    return s or '_col'


def _validate_expression(expression: str) -> None:
    if _BANNED.search(expression):
        raise ValueError(
            "Expression contains forbidden token. "
            "Only numpy/pandas vector operations are allowed."
        )


def _build_local_ns(input_columns: list[str], df: pd.DataFrame) -> dict[str, Any]:
    """Build eval namespace: sanitized var names + col() helper for special-char columns."""
    ns: dict[str, Any] = {"np": np, "pd": pd}
    for col_name in input_columns:
        safe = _sanitize_varname(col_name)
        ns[safe] = df[col_name]
        # Also inject under the original name when it happens to be a valid identifier
        if col_name != safe and col_name.isidentifier():
            ns[col_name] = df[col_name]

    # col('column name') helper — safe, just indexes the df
    _df_ref = df
    ns["col"] = lambda name: _df_ref[name]
    return ns


def _apply_lambda(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    expression = state["expression"]
    input_columns = state["input_columns"]
    result_column = state["result_column"]
    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    local_ns = _build_local_ns(input_columns, df)
    result = eval(expression, {"__builtins__": _ALLOWED_BUILTINS}, local_ns)  # noqa: S307
    if isinstance(result, pd.Series) and len(result) != len(df):
        raise ValueError("Expression result length does not match dataframe length.")
    df = df.copy()
    df[result_column] = result
    return df


def multi_col_lambda(
    df: pd.DataFrame,
    target: str,
    *,
    expression: str,
    input_columns: list[str],
    result_column: str,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Apply a vectorised Python expression over one or more columns.

    Column names are available as sanitized Python identifiers (spaces and special
    characters replaced with underscores). Columns with special characters can also
    be accessed via col('original name'). result_column can be new or existing.

    Args:
        expression: str. Vectorised Python expression; use sanitized col names or col('name').
        input_columns: list[str]. Columns whose values are available in the expression.
        result_column: str. Name of the output column (new or overwrite).
    """
    if result_column == target:
        raise ValueError(f"result_column cannot be the target column '{target}'.")
    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        raise ValueError(f"input_columns not found in dataframe: {missing}")
    _validate_expression(expression)

    # Build var name mapping for the observation so LLM knows what to use
    var_map = {col: _sanitize_varname(col) for col in input_columns}

    state = {
        "expression": expression,
        "input_columns": list(input_columns),
        "result_column": result_column,
    }
    df_out = _apply_lambda(state, df)
    transformer = Transformer(
        operation="multi_col_lambda",
        args={"expression": expression, "input_columns": input_columns, "result_column": result_column},
        state=state,
        apply=functools.partial(_apply_lambda, state),
    )
    action = "overwritten" if result_column in df.columns else "created"
    observation = {
        "result_column": result_column,
        "action": action,
        "dtype": str(df_out[result_column].dtype),
        "sample": df_out[result_column].head(3).tolist(),
        "var_names": var_map,  # sanitized names used in expression
    }
    return df_out, transformer, observation
