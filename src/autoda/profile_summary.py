"""Compact summary derived from ``dataset_profile`` (the ydata-profiling JSON).

The summary is the *small* thing we inject into every planner / reflect prompt.
The raw profile is large (often 6-10kB of JSON) and burns context budget;
the summary highlights only the actionable signal: what's missing, what looks
constant, what looks like an id, what classes are imbalanced, what's correlated.

Computed once in ``profile_node`` and re-used as-is each iteration. (The
profile itself is also refreshed on ``keep`` — see graph.reflect_node —
so the summary should be re-built there too.)
"""

from __future__ import annotations
from typing import Any
import pandas as pd


def build_profile_summary(
    df: pd.DataFrame,
    target: str,
    profile: dict[str, Any] | None = None,
    *,
    high_missing_threshold: float = 0.5,
    high_card_threshold: int = 20,
    near_unique_ratio: float = 0.95,
    high_skew_threshold: float = 1.0,
    high_corr_threshold: float = 0.95,
) -> dict[str, Any]:
    """Return a structured summary of the dataset.

    Most fields come straight from ``df``; ``profile`` is only used to lift
    high-correlation pairs and suspected-leakage columns the profiler already
    computed (so we don't recompute the correlation matrix on huge frames).

    Returns
    -------
    dict with the keys documented in ``__doc__`` above plus:
      shape, n_rows, n_cols, n_numeric, n_categorical, n_datetime,
      constant_columns, all_nan_columns, high_missing_columns,
      all_unique_columns, high_cardinality_categoricals,
      high_skew_columns, datetime_columns, class_balance,
      high_correlation_pairs, suspected_leakage_columns,
      target_dtype, target_nunique.
    """
    profile = profile or {}
    n_rows, n_cols = df.shape
    summary: dict[str, Any] = {
        "shape": [int(n_rows), int(n_cols)],
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_numeric": 0,
        "n_categorical": 0,
        "n_datetime": 0,
        "constant_columns": [],
        "all_nan_columns": [],
        "high_missing_columns": [],
        "all_unique_columns": [],
        "high_cardinality_categoricals": [],
        "high_skew_columns": [],
        "datetime_columns": [],
        "class_balance": None,
        "high_correlation_pairs": [],
        "suspected_leakage_columns": [],
        "target_dtype": str(df[target].dtype) if target in df.columns else None,
        "target_nunique": int(df[target].nunique(dropna=True)) if target in df.columns else None,
    }

    if n_rows == 0:
        return summary

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        dtype = s.dtype
        n_unique = int(s.nunique(dropna=True))
        missing_rate = float(s.isna().mean())

        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            summary["n_numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            summary["n_datetime"] += 1
            summary["datetime_columns"].append(col)
        else:
            summary["n_categorical"] += 1

        # all-NaN
        if missing_rate >= 1.0:
            summary["all_nan_columns"].append(col)
            continue

        # constant
        if n_unique <= 1:
            summary["constant_columns"].append(col)
            continue

        # near-unique (likely an id/timestamp not detected via dtype)
        if n_rows > 1 and (n_unique / n_rows) >= near_unique_ratio:
            summary["all_unique_columns"].append(col)

        # high missing
        if missing_rate >= high_missing_threshold:
            summary["high_missing_columns"].append(
                {"name": col, "missing_rate": round(missing_rate, 4)}
            )

        # high cardinality among categoricals
        if (not pd.api.types.is_numeric_dtype(dtype)) and n_unique > high_card_threshold:
            summary["high_cardinality_categoricals"].append(
                {"name": col, "n_unique": n_unique}
            )

        # skew
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            try:
                skew = float(s.dropna().skew())
                if pd.notna(skew) and abs(skew) > high_skew_threshold:
                    summary["high_skew_columns"].append(
                        {"name": col, "skew": round(skew, 4)}
                    )
            except (TypeError, ValueError):
                pass

    # Pull from profile (already computed by ydata wrapper)
    summary["high_correlation_pairs"] = list(profile.get("high_corr_pairs", []))[:20]
    summary["suspected_leakage_columns"] = list(profile.get("suspected_leakage", []))

    # Target class balance (for clf-style targets only)
    if target in df.columns:
        ty = df[target]
        nu = int(ty.nunique(dropna=True))
        if 2 <= nu <= 50:
            vc = ty.value_counts(normalize=True, dropna=False).head(20)
            summary["class_balance"] = {str(k): round(float(v), 4) for k, v in vc.items()}

    return summary


def format_profile_summary(summary: dict[str, Any], *, max_cols_per_list: int = 30) -> str:
    """Render the structured summary as a short human-readable block for prompts."""
    if not summary:
        return "(no profile summary available)"

    def _truncate(items, n=max_cols_per_list):
        items = list(items)
        if len(items) <= n:
            return items
        return items[:n] + [f"... ({len(items) - n} more)"]

    lines: list[str] = []
    shape = summary.get("shape", [0, 0])
    lines.append(
        f"Shape: {shape[0]} rows × {shape[1]} cols  "
        f"({summary.get('n_numeric',0)} numeric, "
        f"{summary.get('n_categorical',0)} categorical, "
        f"{summary.get('n_datetime',0)} datetime)"
    )

    tgt_dtype = summary.get("target_dtype")
    tgt_nu = summary.get("target_nunique")
    if tgt_dtype is not None:
        lines.append(f"Target: dtype={tgt_dtype}, n_unique={tgt_nu}")
    if summary.get("class_balance"):
        lines.append(f"Target class balance: {summary['class_balance']}")

    def _section(title, value):
        if value:
            lines.append(f"{title}: {_truncate(value)}")

    _section("ALL-NaN columns (DROP)", summary.get("all_nan_columns"))
    _section("Constant columns (DROP)", summary.get("constant_columns"))
    _section(
        "Near-unique columns (likely id/timestamp — usually DROP unless engineered)",
        summary.get("all_unique_columns"),
    )
    _section("High-missing columns (>50%)", summary.get("high_missing_columns"))
    _section("Datetime columns (expand_datetime candidates)", summary.get("datetime_columns"))
    _section("High-cardinality categoricals (frequency/target encode)", summary.get("high_cardinality_categoricals"))
    _section("Heavy-skew numerics (log_transform candidates)", summary.get("high_skew_columns"))
    _section("Suspected leakage with target (|r|>0.95)", summary.get("suspected_leakage_columns"))

    pairs = summary.get("high_correlation_pairs", [])
    if pairs:
        pair_strs = [f"{p.get('col_a')}↔{p.get('col_b')} (r={p.get('r')})" for p in pairs[:10]]
        if len(pairs) > 10:
            pair_strs.append(f"... ({len(pairs) - 10} more)")
        lines.append("High-correlation feature pairs (drop_high_corr candidates): " + ", ".join(pair_strs))

    return "\n".join(lines)
