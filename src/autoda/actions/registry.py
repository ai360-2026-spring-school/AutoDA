from typing import Callable, Literal

from .catalog import build_catalog, format_catalog_compact

from .clean import (impute_missing, drop_columns, drop_duplicates, clip_outliers,
                    collapse_rare_categories, cast_dtype)
from .fe import (expand_datetime, binarize_missing, log_transform, bin_numeric,
                 interaction, group_aggregate, power_transform, multi_interaction)
from .encode import (frequency_encode, target_encode_oof, one_hot,
                     standard_scale, min_max_scale)
from .select import drop_constant, drop_high_corr, drop_low_importance
from .lambda_transform import multi_col_lambda
from .info import sparse_linear_features, baseline_linear_model
from .info_new import (
    groupby_agg, value_counts, correlation_matrix,
    describe_column, view_precomputed_stats, view_long_summary,
)

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
    "power_transform": power_transform,
    "multi_interaction": multi_interaction,
    "group_aggregate": group_aggregate,
    "frequency_encode": frequency_encode,
    "target_encode_oof": target_encode_oof,
    "one_hot": one_hot,
    "standard_scale": standard_scale,
    "min_max_scale": min_max_scale,
    "drop_constant": drop_constant,
    "drop_high_corr": drop_high_corr,
    "drop_low_importance": drop_low_importance,
    "multi_col_lambda": multi_col_lambda,
}

INFO_TOOLS: dict[str, Callable] = {
    "sparse_linear_features": sparse_linear_features,
    "baseline_linear_model": baseline_linear_model,
    "groupby_agg": groupby_agg,
    "value_counts": value_counts,
    "correlation_matrix": correlation_matrix,
    "describe_column": describe_column,
    "view_precomputed_stats": view_precomputed_stats,
    "view_long_summary": view_long_summary,
}

REGISTRY: dict[str, Callable] = {**TRANSFORMERS, **INFO_TOOLS}


def kind_of(op: str) -> Literal["transformer", "info"]:
    if op in TRANSFORMERS:
        return "transformer"
    if op in INFO_TOOLS:
        return "info"
    raise KeyError(f"unknown operation: {op!r}")


CATALOG: list[dict] = build_catalog(TRANSFORMERS, INFO_TOOLS)
SCHEMA: list[dict] = CATALOG


def catalog_text() -> str:
    return format_catalog_compact(CATALOG)
