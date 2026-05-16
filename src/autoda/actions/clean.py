from typing import Any
import pandas as pd


def impute_missing(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    strategy: str = "mean",
    fill_value: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = {"mean", "median", "mode", "constant", "knn"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {valid}, got {strategy!r}")

    df = df.copy()
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot impute the target column")
        if df[col].isna().sum() == 0:
            continue

        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif strategy == "constant":
            df[col] = df[col].fillna(fill_value)
        elif strategy == "knn":
            from sklearn.impute import KNNImputer
            imp = KNNImputer()
            df[[col]] = imp.fit_transform(df[[col]])

        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"imputed {len(changed)} column(s) with strategy={strategy!r}"}


def drop_columns(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for col in columns:
        if col == target:
            raise ValueError("cannot drop the target column")
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")

    df = df.drop(columns=columns)
    return df, {"changed_columns": columns, "summary": f"dropped {len(columns)} column(s)"}


def drop_duplicates(
    df: pd.DataFrame,
    target: str,
    subset: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    removed = before - len(df)
    return df, {"changed_columns": [], "summary": f"removed {removed} duplicate row(s)"}


def clip_outliers(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    method: str = "iqr",
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if method not in ("iqr", "quantile"):
        raise ValueError(f"method must be 'iqr' or 'quantile', got {method!r}")

    df = df.copy()
    changed: list[str] = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"column not found: {col!r}")
        if col == target:
            raise ValueError("cannot clip the target column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"column {col!r} is not numeric")

        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
        else:
            lo = df[col].quantile(lower if lower is not None else 0.01)
            hi = df[col].quantile(upper if upper is not None else 0.99)

        df[col] = df[col].clip(lo, hi)
        changed.append(col)

    return df, {"changed_columns": changed, "summary": f"clipped outliers in {len(changed)} column(s) using {method!r}"}


def collapse_rare_categories(
    df: pd.DataFrame,
    target: str,
    column: str,
    min_freq: int | float = 50,
    other_label: str = "OTHER",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot collapse the target column")

    df = df.copy()
    vc = df[column].value_counts(dropna=False)

    if isinstance(min_freq, float) and min_freq < 1:
        threshold = int(min_freq * len(df))
    else:
        threshold = int(min_freq)

    rare = vc[vc < threshold].index
    df[column] = df[column].where(~df[column].isin(rare), other=other_label)

    return df, {
        "changed_columns": [column],
        "summary": f"collapsed {len(rare)} rare categories in {column!r} (threshold={threshold})",
    }


def cast_dtype(
    df: pd.DataFrame,
    target: str,
    column: str,
    dtype: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if column not in df.columns:
        raise ValueError(f"column not found: {column!r}")
    if column == target:
        raise ValueError("cannot cast the target column")

    dtype_map = {"int": "int64", "float": "float64", "category": "category", "datetime": "datetime64[ns]"}
    if dtype not in dtype_map:
        raise ValueError(f"dtype must be one of {list(dtype_map)}, got {dtype!r}")

    df = df.copy()
    if dtype == "datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce")
    else:
        df[column] = df[column].astype(dtype_map[dtype])

    return df, {"changed_columns": [column], "summary": f"cast {column!r} to {dtype!r}"}
