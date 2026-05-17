from __future__ import annotations

import functools
from typing import Any, Literal

import pandas as pd

from .column_typer import ColumnKind, detect_column_types
from .evaluator import RANDOM_SEED
from .transformer import Transformer


# ---------------------------------------------------------------------------
# apply helpers (top-level, picklable)
# ---------------------------------------------------------------------------

def _apply_expand_datetime_preprocess(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col: str = state["column"]
    if col not in df.columns:
        return df
    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}__day"] = dt.dt.day
    df[f"{col}__month"] = dt.dt.month
    df[f"{col}__year"] = dt.dt.year
    df[f"{col}__dow"] = dt.dt.dayofweek
    df = df.drop(columns=[col])
    return df


def _apply_impute_numeric(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mean_val in state["means"].items():
        if col not in df.columns:
            continue
        missed_col = f"is_{col}_missed"
        if missed_col not in df.columns and col in state["has_missing"]:
            df[missed_col] = df[col].isna().astype("int8")
        df[col] = df[col].fillna(mean_val)
    return df


def _apply_impute_categorical(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in state["columns"]:
        if col in df.columns:
            df[col] = df[col].fillna("MISSED").astype(str)
    return df


def _apply_one_hot_preprocess(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns: list[str] = state["columns"]
    categories: dict[str, list] = state["categories"]
    dummy_cols: list[str] = state["dummy_cols"]

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.Categorical(df[col], categories=categories[col])

    df = pd.get_dummies(df, columns=[c for c in columns if c in df.columns], dummy_na=False)
    for c in dummy_cols:
        if c not in df.columns:
            df[c] = 0
    return df


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run_preprocess(
    df: pd.DataFrame,
    target: str,
    task: Literal["binary", "multiclass", "regression"],
    *,
    ohe_max_cardinality: int = 12,
    numeric_unique_threshold: int = 12,
    oversample: bool = True,
) -> tuple[pd.DataFrame, list[Transformer], dict[str, ColumnKind], dict[str, Any]]:
    """Deterministic preprocessing pipeline run once before the iterative loop.

    Returns:
        (preprocessed_df, transformers, column_type_map, preprocess_report)
    """
    transformers: list[Transformer] = []
    report: dict[str, Any] = {}

    # --- Step 1: expand datetime columns ---
    datetime_cols = [
        c for c in df.columns
        if c != target and pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    for col in datetime_cols:
        state = {"column": col}
        df = _apply_expand_datetime_preprocess(state, df)
        transformers.append(Transformer(
            operation="preprocess_expand_datetime",
            args={"column": col},
            state=state,
            apply=functools.partial(_apply_expand_datetime_preprocess, state),
        ))
    report["datetime_expanded"] = datetime_cols

    # --- Step 2: detect real column types ---
    col_type_map = detect_column_types(df, target, unique_threshold=numeric_unique_threshold)

    # --- Step 3: handle missing values ---
    numeric_cols = [c for c in col_type_map if col_type_map[c] == "NUMERIC"]
    cat_cols = [c for c in col_type_map if col_type_map[c] == "CATEGORICAL"]

    # Numeric: fill mean + add is_xxx_missed
    numeric_with_missing = [c for c in numeric_cols if df[c].isna().any()]
    means: dict[str, float] = {}
    for col in numeric_cols:
        m = df[col].mean()
        means[col] = float(m) if pd.notna(m) else 0.0

    if numeric_cols:
        state_num = {"means": means, "has_missing": set(numeric_with_missing)}
        df = _apply_impute_numeric(state_num, df)
        transformers.append(Transformer(
            operation="preprocess_impute_numeric",
            args={"columns": numeric_cols},
            state=state_num,
            apply=functools.partial(_apply_impute_numeric, state_num),
        ))

    # Categorical: fillna "MISSED"
    cat_with_missing = [c for c in cat_cols if c in df.columns and df[c].isna().any()]
    if cat_with_missing:
        state_cat = {"columns": cat_with_missing}
        df = _apply_impute_categorical(state_cat, df)
        transformers.append(Transformer(
            operation="preprocess_impute_categorical",
            args={"columns": cat_with_missing},
            state=state_cat,
            apply=functools.partial(_apply_impute_categorical, state_cat),
        ))

    report["numeric_imputed"] = len(numeric_with_missing)
    report["numeric_missed_cols_added"] = [f"is_{c}_missed" for c in numeric_with_missing]
    report["categorical_imputed"] = len(cat_with_missing)

    # --- Step 4: upsampling (train-only, no Transformer) ---
    if oversample and task in ("binary", "multiclass"):
        class_counts = df[target].value_counts()
        majority_count = int(class_counts.max())
        minority_classes = class_counts[class_counts < majority_count].index.tolist()
        extras = []
        for cls in minority_classes:
            subset = df[df[target] == cls]
            n_needed = majority_count - len(subset)
            sampled = subset.sample(n=n_needed, replace=True, random_state=RANDOM_SEED)
            extras.append(sampled)
        if extras:
            df = pd.concat([df] + extras, ignore_index=True)
            df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        report["upsampled"] = {str(cls): majority_count for cls in minority_classes}
    else:
        report["upsampled"] = {}

    # --- Step 5: OHE low-cardinality categoricals ---
    # Re-detect types after imputation (new is_xxx_missed cols are CATEGORICAL)
    col_type_map = detect_column_types(df, target, unique_threshold=numeric_unique_threshold)
    cat_cols_now = [c for c in col_type_map if col_type_map[c] == "CATEGORICAL" and c in df.columns]
    ohe_cols = []
    for _c in cat_cols_now:
        n_uniq = df[_c].nunique(dropna=True)
        if n_uniq >= ohe_max_cardinality:
            continue
        _dtype = df[_c].dtype
        # Skip bool and binary numeric (0/1) columns — already fine for CatBoost
        if pd.api.types.is_bool_dtype(_dtype):
            continue
        if pd.api.types.is_numeric_dtype(_dtype) and n_uniq <= 2:
            continue
        ohe_cols.append(_c)

    if ohe_cols:
        categories: dict[str, list] = {}
        for col in ohe_cols:
            categories[col] = sorted(df[col].dropna().unique().tolist())

        state_ohe = {"columns": ohe_cols, "categories": categories, "dummy_cols": []}
        df_ohe = _apply_one_hot_preprocess(state_ohe, df)
        dummy_cols = [c for c in df_ohe.columns if c not in df.columns]
        state_ohe["dummy_cols"] = dummy_cols
        df = df_ohe
        transformers.append(Transformer(
            operation="preprocess_ohe",
            args={"columns": ohe_cols, "max_cardinality": ohe_max_cardinality},
            state=state_ohe,
            apply=functools.partial(_apply_one_hot_preprocess, state_ohe),
        ))
        report["ohe_columns"] = ohe_cols

    # Final type map (after OHE)
    col_type_map = detect_column_types(df, target, unique_threshold=numeric_unique_threshold)

    # --- Step 6: precompute correlation stats ---
    target_correlation_stats = _compute_correlation_stats(df, target, task, col_type_map)
    report["target_correlation_stats_computed"] = True

    return df, transformers, col_type_map, target_correlation_stats


def _compute_correlation_stats(
    df: pd.DataFrame,
    target: str,
    task: Literal["binary", "multiclass", "regression"],
    col_type_map: dict[str, ColumnKind],
) -> dict[str, Any]:
    numeric_cols = [c for c in col_type_map if col_type_map[c] == "NUMERIC" and c in df.columns]
    if not numeric_cols:
        return {}

    if task in ("regression", "binary"):
        if target not in df.columns:
            return {}
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        corr = df[numeric_cols].corrwith(target_numeric).round(4)
        return {col: {"pearson": float(v)} for col, v in corr.items() if pd.notna(v)}
    else:
        # multiclass: mean of each col per class
        if target not in df.columns:
            return {}
        result: dict[str, dict[str, float]] = {}
        grouped = df.groupby(target)[numeric_cols].mean().round(4)
        for col in numeric_cols:
            if col in grouped.columns:
                result[col] = {str(cls): float(v) for cls, v in grouped[col].items()}
        return result
