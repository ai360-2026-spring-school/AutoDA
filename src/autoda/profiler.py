from pathlib import Path
from typing import Any
import pandas as pd


def profile_dataset(
    df: pd.DataFrame,
    target: str,
    *,
    html_path: Path | None = None,
    minimal: bool = True,
) -> dict[str, Any]:
    import ydata_profiling  # lazy import

    report = ydata_profiling.ProfileReport(df, minimal=minimal, progress_bar=False)

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_file(html_path)

    desc = report.get_description()
    variables = desc.variables

    # --- target summary ---
    target_info: dict[str, Any] = {"name": target, "dtype": str(df[target].dtype)}
    if target in variables:
        v = variables[target]
        if pd.api.types.is_numeric_dtype(df[target]):
            for k in ("mean", "std", "min", "max", "p25", "p75"):
                val = getattr(v, k, None)
                if val is not None:
                    try:
                        target_info[k] = float(val)
                    except (TypeError, ValueError):
                        pass
        vc = df[target].value_counts(normalize=True, dropna=False)
        target_info["class_balance"] = {str(k): round(float(v), 4) for k, v in vc.head(10).items()}

    # --- per-column compact summary ---
    columns_summary = []
    for col in df.columns:
        s = df[col]
        v = variables.get(col)
        item: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_rate": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            for attr in ("mean", "std", "skewness", "kurtosis", "iqr"):
                val = getattr(v, attr, None) if v else None
                if val is not None:
                    try:
                        item[attr] = round(float(val), 4)
                    except (TypeError, ValueError):
                        pass
            item["skew_flag"] = abs(item.get("skewness", 0)) > 1
        else:
            vc = s.value_counts(dropna=False)
            item["top_3"] = {str(k): int(cnt) for k, cnt in vc.head(3).items()}
        columns_summary.append(item)

    # --- high-correlation pairs ---
    numeric_df = df.select_dtypes(include="number")
    high_corr_pairs: list[dict[str, Any]] = []
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr(numeric_only=True)
        seen: set[frozenset] = set()
        for c1 in corr.columns:
            for c2 in corr.columns:
                if c1 == c2:
                    continue
                pair = frozenset({c1, c2})
                if pair in seen:
                    continue
                seen.add(pair)
                r = corr.loc[c1, c2]
                if pd.notna(r) and abs(r) > 0.95:
                    high_corr_pairs.append({"col_a": c1, "col_b": c2, "r": round(float(r), 4)})

    # --- suspected leakage (|corr with target| > 0.95, excluding target itself) ---
    suspected_leakage: list[str] = []
    if target in numeric_df.columns:
        target_corr = numeric_df.corr()[target].drop(labels=[target], errors="ignore").dropna()
        suspected_leakage = [col for col, r in target_corr.items() if abs(r) > 0.95]

    return {
        "shape": list(df.shape),
        "target": target_info,
        "columns": columns_summary,
        "high_corr_pairs": high_corr_pairs,
        "suspected_leakage": suspected_leakage,
    }
