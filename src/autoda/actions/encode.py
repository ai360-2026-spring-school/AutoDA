from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from autoda.evaluator import RANDOM_SEED, DEFAULT_N_SPLITS


def frequency_encode(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot encode the target column")
        freq = df[col].value_counts(normalize=True, dropna=False)
        df[col] = df[col].map(freq).astype(float)
        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"frequency-encoded {len(changed)} column(s)"}


def target_encode_oof(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("target must be numeric for target encoding")

    df = df.copy()
    y = df[target]
    global_mean = float(y.mean())

    n_unique = y.nunique()
    if n_unique <= 2 or (pd.api.types.is_integer_dtype(y) and n_unique <= 50):
        splitter = StratifiedKFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(df, y)
    else:
        splitter = KFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(df)

    changed: list[str] = []
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot encode the target column")

        encoded = pd.Series(index=df.index, dtype=float)

        for train_idx, val_idx in splitter.split(df, y) if hasattr(splitter, "split") else split_iter:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            stats = train_df.groupby(col)[target].agg(["mean", "count"])
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            encoded.iloc[val_idx] = val_df[col].map(stats["smoothed"]).fillna(global_mean).values

        df[col] = encoded.fillna(global_mean)
        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"OOF target-encoded {len(changed)} column(s) (smoothing={smoothing})"}


def one_hot(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    max_cardinality: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot one-hot encode the target column")
        if df[col].nunique(dropna=True) > max_cardinality:
            raise ValueError(f"column {col!r} has >{max_cardinality} unique values; reduce cardinality first")
        changed.append(col)

    df = pd.get_dummies(df, columns=columns, dummy_na=False)
    new_cols = [c for c in df.columns if any(c.startswith(f"{col}_") for col in changed)]

    return df, {"changed_columns": new_cols, "summary": f"one-hot encoded {len(changed)} column(s) -> {len(new_cols)} new columns"}


def standard_scale(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot scale the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")

        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std if std > 0 else 1)
        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"standard-scaled {len(changed)} column(s)"}


def min_max_scale(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot scale the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")

        lo = df[col].min()
        hi = df[col].max()
        df[col] = (df[col] - lo) / (hi - lo if hi > lo else 1)
        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"min-max scaled {len(changed)} column(s)"}
