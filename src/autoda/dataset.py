from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class DatasetContext:
    dataset_id: str
    df: pd.DataFrame
    target: str | None = None


def profile_dataframe(df: pd.DataFrame, max_categories: int = 20) -> dict[str, Any]:
    profile = {
        "shape": list(df.shape),
        "columns": [],
    }

    for col in df.columns:
        s = df[col]
        item = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)), # мб ещё unique_rate ?
        }

        if s.nunique(dropna=True) <= max_categories:
            item["top_values"] = (
                s.value_counts(dropna=False).head(10).astype(int).to_dict()
            )

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe().to_dict()
            item["stats"] = {k: float(v) for k, v in desc.items() if pd.notna(v)}

        profile["columns"].append(item)

    return profile
