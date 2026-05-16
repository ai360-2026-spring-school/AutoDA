from typing import Callable, Literal

from .clean import (impute_missing, drop_columns, drop_duplicates, clip_outliers,
                    collapse_rare_categories, cast_dtype)
from .fe import (expand_datetime, binarize_missing, log_transform, bin_numeric,
                 interaction, group_aggregate)
from .encode import (frequency_encode, target_encode_oof, one_hot,
                     standard_scale, min_max_scale)
from .select import drop_constant, drop_high_corr, drop_low_importance
from .info import sparse_linear_features, baseline_linear_model

TRANSFORMERS: dict[str, Callable] = {
    "impute_missing": impute_missing,
    "drop_columns": drop_columns,
    "drop_duplicates": drop_duplicates,
    "clip_outliers": clip_outliers,
    "collapse_rare_categories": collapse_rare_categories,
    "cast_dtype": cast_dtype,
    "expand_datetime": expand_datetime,
    "binarize_missing": binarize_missing,
    "log_transform": log_transform,
    "bin_numeric": bin_numeric,
    "interaction": interaction,
    "group_aggregate": group_aggregate,
    "frequency_encode": frequency_encode,
    "target_encode_oof": target_encode_oof,
    "one_hot": one_hot,
    "standard_scale": standard_scale,
    "min_max_scale": min_max_scale,
    "drop_constant": drop_constant,
    "drop_high_corr": drop_high_corr,
    "drop_low_importance": drop_low_importance,
}

INFO_TOOLS: dict[str, Callable] = {
    "sparse_linear_features": sparse_linear_features,
    "baseline_linear_model": baseline_linear_model,
}

REGISTRY: dict[str, Callable] = {**TRANSFORMERS, **INFO_TOOLS}


def kind_of(op: str) -> Literal["transformer", "info"]:
    if op in TRANSFORMERS:
        return "transformer"
    if op in INFO_TOOLS:
        return "info"
    raise KeyError(f"unknown operation: {op!r}")


SCHEMA: list[dict] = [
    # ---- clean (transformers) ----
    {"operation": "impute_missing", "kind": "transformer", "args": {"columns": ["col1"], "strategy": "mean|median|mode|constant|knn", "fill_value": "(optional)"}},
    {"operation": "drop_columns", "kind": "transformer", "args": {"columns": ["col1"]}},
    {"operation": "drop_duplicates", "kind": "transformer", "args": {"subset": "(optional list[str])"}},
    {"operation": "clip_outliers", "kind": "transformer", "args": {"columns": ["col1"], "method": "iqr|quantile", "lower": "(optional float)", "upper": "(optional float)"}},
    {"operation": "collapse_rare_categories", "kind": "transformer", "args": {"column": "col", "min_freq": "int or float<1", "other_label": "OTHER"}},
    {"operation": "cast_dtype", "kind": "transformer", "args": {"column": "col", "dtype": "int|float|category|datetime"}},
    # ---- fe (transformers) ----
    {"operation": "expand_datetime", "kind": "transformer", "args": {"column": "col", "parts": ["year", "month", "dow", "hour", "is_weekend"]}},
    {"operation": "binarize_missing", "kind": "transformer", "args": {"columns": ["col1"]}},
    {"operation": "log_transform", "kind": "transformer", "args": {"columns": ["col1"], "plus_one": True}},
    {"operation": "bin_numeric", "kind": "transformer", "args": {"column": "col", "n_bins": 5, "strategy": "quantile|uniform"}},
    {"operation": "interaction", "kind": "transformer", "args": {"cols": ["a", "b"], "op": "mul|div|add|sub"}},
    {"operation": "group_aggregate", "kind": "transformer", "args": {"by": "col", "value": "col", "agg": "mean|median|std|count"}},
    # ---- encode (transformers) ----
    {"operation": "frequency_encode", "kind": "transformer", "args": {"columns": ["col1"]}},
    {"operation": "target_encode_oof", "kind": "transformer", "args": {"columns": ["col1"], "smoothing": 10.0}},
    {"operation": "one_hot", "kind": "transformer", "args": {"columns": ["col1"], "max_cardinality": 20}},
    {"operation": "standard_scale", "kind": "transformer", "args": {"columns": ["col1"]}},
    {"operation": "min_max_scale", "kind": "transformer", "args": {"columns": ["col1"]}},
    # ---- select (transformers) ----
    {"operation": "drop_constant", "kind": "transformer", "args": {}},
    {"operation": "drop_high_corr", "kind": "transformer", "args": {"threshold": 0.98}},
    {"operation": "drop_low_importance", "kind": "transformer", "args": {"top_k_keep": "(optional int)", "min_importance": "(optional float)"}},
    # ---- info tools ----
    {"operation": "sparse_linear_features", "kind": "info", "args": {"n_keep": 15, "alpha": "(optional float, None=CV)"}},
    {"operation": "baseline_linear_model", "kind": "info", "args": {}},
]
