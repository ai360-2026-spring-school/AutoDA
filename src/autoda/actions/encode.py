from __future__ import annotations

import functools
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from autoda.evaluator import RANDOM_SEED, DEFAULT_N_SPLITS
from autoda.transformer import Transformer


# ---------------------------------------------------------------------------
# apply helpers (top-level, picklable)
# ---------------------------------------------------------------------------

def _apply_frequency_encode(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, freq_map in state["freq_maps"].items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(freq_map).fillna(0).astype(float)
    return df


def _apply_target_encode_oof(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Apply uses full-train encoding maps (not OOF)."""
    df = df.copy()
    global_mean: float = state["global_mean"]
    for col, enc_map in state["encoding_maps"].items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(enc_map).fillna(global_mean)
    return df


def _apply_one_hot(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns: list[str] = state["columns"]
    categories: dict[str, list] = state["categories"]
    dummy_cols: list[str] = state["dummy_cols"]  # only the new cols created at fit time

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.Categorical(df[col], categories=categories[col])

    df = pd.get_dummies(df, columns=[c for c in columns if c in df.columns], dummy_na=False)

    # add any missing dummy columns (unseen categories) as 0-filled
    for c in dummy_cols:
        if c not in df.columns:
            df[c] = 0

    return df


def _apply_standard_scale(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, params in state["scale_params"].items():
        if col not in df.columns:
            continue
        mean = params["mean"]
        std = params["std"]
        df[col] = (df[col] - mean) / max(std, 1e-8)
    return df


def _apply_min_max_scale(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, params in state["scale_params"].items():
        if col not in df.columns:
            continue
        lo = params["min"]
        hi = params["max"]
        df[col] = (df[col] - lo) / max(hi - lo, 1e-8)
    return df


# ---------------------------------------------------------------------------
# public action functions — return (df, Transformer, observation)
# ---------------------------------------------------------------------------

def frequency_encode(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    freq_maps: dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot encode the target column")
        freq = df[col].value_counts(normalize=True, dropna=False)
        freq_maps[col] = {k: float(v) for k, v in freq.items()}

    state: dict[str, Any] = {"freq_maps": freq_maps}
    df_out = _apply_frequency_encode(state, df)

    transformer = Transformer(
        operation="frequency_encode",
        args={"columns": columns},
        state=state,
        apply=functools.partial(_apply_frequency_encode, state),
    )
    return df_out, transformer, {
        "changed_columns": list(freq_maps.keys()),
        "summary": f"frequency-encoded {len(freq_maps)} column(s)",
    }


def target_encode_oof(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("target must be numeric for target encoding")

    y = df[target]
    global_mean = float(y.mean())

    n_unique = y.nunique()
    if n_unique <= 2 or (pd.api.types.is_integer_dtype(y) and n_unique <= 50):
        splitter: StratifiedKFold | KFold = StratifiedKFold(
            n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED
        )
    else:
        splitter = KFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    df_out = df.copy()
    changed: list[str] = []

    # --- OOF encode for train df ---
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot encode the target column")

        encoded = pd.Series(index=df.index, dtype=float)
        for train_idx, val_idx in splitter.split(df, y):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            stats = train_df.groupby(col)[target].agg(["mean", "count"])
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            encoded.iloc[val_idx] = val_df[col].map(stats["smoothed"]).fillna(global_mean).values

        df_out[col] = encoded.fillna(global_mean)
        changed.append(col)

    # --- Full-train encoding maps for state (used in apply on test) ---
    encoding_maps: dict[str, dict] = {}
    for col in columns:
        stats = df.groupby(col)[target].agg(["mean", "count"])
        stats["smoothed"] = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        encoding_maps[col] = {k: float(v) for k, v in stats["smoothed"].items()}

    state: dict[str, Any] = {"encoding_maps": encoding_maps, "global_mean": global_mean}
    transformer = Transformer(
        operation="target_encode_oof",
        args={"columns": columns, "smoothing": smoothing},
        state=state,
        apply=functools.partial(_apply_target_encode_oof, state),
    )
    return df_out, transformer, {
        "changed_columns": changed,
        "summary": f"OOF target-encoded {len(changed)} column(s) (smoothing={smoothing})",
    }


def one_hot(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    max_cardinality: int = 20,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot one-hot encode the target column")
        if df[col].nunique(dropna=True) > max_cardinality:
            raise ValueError(
                f"column {col!r} has >{max_cardinality} unique values; reduce cardinality first"
            )

    # compute categories per column
    categories: dict[str, list] = {}
    for col in columns:
        cats = sorted(df[col].dropna().unique().tolist())
        categories[col] = cats

    # apply categorical + get_dummies on train
    df_tmp = df.copy()
    for col in columns:
        df_tmp[col] = pd.Categorical(df_tmp[col], categories=categories[col])
    df_out = pd.get_dummies(df_tmp, columns=columns, dummy_na=False)

    dummy_cols = [c for c in df_out.columns if any(c.startswith(f"{col}_") for col in columns)]
    state: dict[str, Any] = {
        "columns": columns,
        "categories": categories,
        "dummy_cols": dummy_cols,
    }
    transformer = Transformer(
        operation="one_hot",
        args={"columns": columns, "max_cardinality": max_cardinality},
        state=state,
        apply=functools.partial(_apply_one_hot, state),
    )
    new_cols = [c for c in df_out.columns if c not in df.columns]
    return df_out, transformer, {
        "changed_columns": new_cols,
        "summary": f"one-hot encoded {len(columns)} column(s) -> {len(new_cols)} new columns",
    }


def standard_scale(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    scale_params: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot scale the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")
        scale_params[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}

    state: dict[str, Any] = {"scale_params": scale_params}
    df_out = _apply_standard_scale(state, df)

    transformer = Transformer(
        operation="standard_scale",
        args={"columns": columns},
        state=state,
        apply=functools.partial(_apply_standard_scale, state),
    )
    return df_out, transformer, {
        "changed_columns": list(scale_params.keys()),
        "summary": f"standard-scaled {len(scale_params)} column(s)",
    }


def min_max_scale(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    scale_params: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot scale the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")
        scale_params[col] = {"min": float(df[col].min()), "max": float(df[col].max())}

    state: dict[str, Any] = {"scale_params": scale_params}
    df_out = _apply_min_max_scale(state, df)

    transformer = Transformer(
        operation="min_max_scale",
        args={"columns": columns},
        state=state,
        apply=functools.partial(_apply_min_max_scale, state),
    )
    return df_out, transformer, {
        "changed_columns": list(scale_params.keys()),
        "summary": f"min-max scaled {len(scale_params)} column(s)",
    }
