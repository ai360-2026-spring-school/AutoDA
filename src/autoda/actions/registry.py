from typing import Callable
import pandas as pd

from .clean import (
    impute_missing,
    drop_columns,
    drop_duplicates,
    clip_outliers,
    collapse_rare_categories,
    cast_dtype,
)
from .fe import (
    expand_datetime,
    binarize_missing,
    log_transform,
    bin_numeric,
    interaction,
    group_aggregate,
)
from .encode import (
    frequency_encode,
    target_encode_oof,
    one_hot,
    standard_scale,
    min_max_scale,
)
from .select import (
    drop_constant,
    drop_high_corr,
    drop_low_importance,
)
from .insight import record_insight

REGISTRY: dict[str, Callable] = {
    # clean
    "impute_missing": impute_missing,
    "drop_columns": drop_columns,
    "drop_duplicates": drop_duplicates,
    "clip_outliers": clip_outliers,
    "collapse_rare_categories": collapse_rare_categories,
    "cast_dtype": cast_dtype,
    # fe
    "expand_datetime": expand_datetime,
    "binarize_missing": binarize_missing,
    "log_transform": log_transform,
    "bin_numeric": bin_numeric,
    "interaction": interaction,
    "group_aggregate": group_aggregate,
    # encode
    "frequency_encode": frequency_encode,
    "target_encode_oof": target_encode_oof,
    "one_hot": one_hot,
    "standard_scale": standard_scale,
    "min_max_scale": min_max_scale,
    # select
    "drop_constant": drop_constant,
    "drop_high_corr": drop_high_corr,
    "drop_low_importance": drop_low_importance,
    # insight
    "record_insight": record_insight,
}

SCHEMA: list[dict] = [
    # ---- clean ----
    {"operation": "impute_missing", "args": {"columns": ["col1"], "strategy": "mean|median|mode|constant|knn", "fill_value": "(optional)"}},
    {"operation": "drop_columns", "args": {"columns": ["col1"]}},
    {"operation": "drop_duplicates", "args": {"subset": "(optional list[str])"}},
    {"operation": "clip_outliers", "args": {"columns": ["col1"], "method": "iqr|quantile", "lower": "(optional float)", "upper": "(optional float)"}},
    {"operation": "collapse_rare_categories", "args": {"column": "col", "min_freq": "int or float<1", "other_label": "OTHER"}},
    {"operation": "cast_dtype", "args": {"column": "col", "dtype": "int|float|category|datetime"}},
    # ---- fe ----
    {"operation": "expand_datetime", "args": {"column": "col", "parts": ["year", "month", "dow", "hour", "is_weekend"]}},
    {"operation": "binarize_missing", "args": {"columns": ["col1"]}},
    {"operation": "log_transform", "args": {"columns": ["col1"], "plus_one": True}},
    {"operation": "bin_numeric", "args": {"column": "col", "n_bins": 5, "strategy": "quantile|uniform"}},
    {"operation": "interaction", "args": {"cols": ["a", "b"], "op": "mul|div|add|sub"}},
    {"operation": "group_aggregate", "args": {"by": "col", "value": "col", "agg": "mean|median|std|count"}},
    # ---- encode ----
    {"operation": "frequency_encode", "args": {"columns": ["col1"]}},
    {"operation": "target_encode_oof", "args": {"columns": ["col1"], "smoothing": 10.0}},
    {"operation": "one_hot", "args": {"columns": ["col1"], "max_cardinality": 20}},
    {"operation": "standard_scale", "args": {"columns": ["col1"]}},
    {"operation": "min_max_scale", "args": {"columns": ["col1"]}},
    # ---- select ----
    {"operation": "drop_constant", "args": {}},
    {"operation": "drop_high_corr", "args": {"threshold": 0.98}},
    {"operation": "drop_low_importance", "args": {"top_k_keep": "(optional int)", "min_importance": "(optional float)"}},
    # ---- insight ----
    {"operation": "record_insight", "args": {"title": "str", "body": "str", "evidence": {}}},
]
