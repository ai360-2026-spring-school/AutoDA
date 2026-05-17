from __future__ import annotations

import functools
from typing import Any
import pandas as pd

from autoda.transformer import Transformer


# ---------------------------------------------------------------------------
# apply helpers (top-level, picklable)
# ---------------------------------------------------------------------------

def _apply_drop_cols_by_state(state: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=state["columns"], errors="ignore")


# ---------------------------------------------------------------------------
# public action functions — return (df, Transformer, observation)
# ---------------------------------------------------------------------------

def drop_constant(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Drop every column with a single unique value (incl. all-NaN).

    Use when: profile_summary lists constant_columns / all_nan_columns. Safe baseline cleanup.
    Effect: those columns disappear from train and test. Cheap and almost always a CV-neutral or slight win.

    Args:
        (no args)
    """
    constant_cols = [
        col for col in df.columns
        if col != target and df[col].nunique(dropna=False) <= 1
    ]
    state: dict[str, Any] = {"columns": constant_cols}
    df_out = df.drop(columns=constant_cols)
    transformer = Transformer(
        operation="drop_constant",
        args={},
        state=state,
        apply=functools.partial(_apply_drop_cols_by_state, state),
    )
    return df_out, transformer, {
        "changed_columns": constant_cols,
        "summary": f"dropped {len(constant_cols)} constant column(s)",
    }


def drop_high_corr(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.98,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Drop near-duplicate numeric features by pairwise correlation.

    Use when: profile_summary lists high_correlation_pairs and the model is wasting splits on redundant columns.
    Effect: from each correlated pair (|r| > threshold), one column is dropped. Same drop list applied to test.

    Args:
        threshold: float. Absolute correlation above which a column is considered redundant. (default: 0.98)
    """
    numeric = df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
    corr = numeric.corr().abs()

    to_drop: list[str] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[j] not in to_drop and corr.iloc[i, j] > threshold:
                to_drop.append(cols[j])

    state: dict[str, Any] = {"columns": to_drop}
    df_out = df.drop(columns=to_drop, errors="ignore")
    transformer = Transformer(
        operation="drop_high_corr",
        args={"threshold": threshold},
        state=state,
        apply=functools.partial(_apply_drop_cols_by_state, state),
    )
    return df_out, transformer, {
        "changed_columns": to_drop,
        "summary": f"dropped {len(to_drop)} highly-correlated column(s) (threshold={threshold})",
    }


def drop_low_importance(
    df: pd.DataFrame,
    target: str,
    top_k_keep: int | None = None,
    min_importance: float | None = None,
    feature_importances: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, Transformer, dict[str, Any]]:
    """Drop features with low CatBoost feature_importance from the last CV.

    Use when: a previous CV run produced importances and you suspect dead weight columns are hurting splits / generalisation.
    Effect: keeps only the top features by the last CV's importance scores. Same drop list applied to test.

    Args:
        top_k_keep: int | None. Keep only the top-K features by importance. Mutually exclusive with min_importance.
        min_importance: float | None. Keep only features whose importance >= this. Mutually exclusive with top_k_keep.
    """
    if feature_importances is None:
        raise ValueError("feature_importances must be provided by apply_node from the last CatBoost run")
    if top_k_keep is None and min_importance is None:
        raise ValueError("provide top_k_keep or min_importance")

    features = [c for c in df.columns if c != target and c in feature_importances]
    ranked = sorted(features, key=lambda c: feature_importances.get(c, 0), reverse=True)

    if top_k_keep is not None:
        keep = set(ranked[:top_k_keep])
    else:
        keep = {c for c in ranked if feature_importances.get(c, 0) >= min_importance}  # type: ignore[operator]

    to_drop = [c for c in features if c not in keep]
    state: dict[str, Any] = {"columns": to_drop}
    df_out = df.drop(columns=to_drop, errors="ignore")
    transformer = Transformer(
        operation="drop_low_importance",
        args={"top_k_keep": top_k_keep, "min_importance": min_importance},
        state=state,
        apply=functools.partial(_apply_drop_cols_by_state, state),
    )
    return df_out, transformer, {
        "changed_columns": to_drop,
        "summary": f"dropped {len(to_drop)} low-importance column(s); kept top {len(keep)}",
    }
