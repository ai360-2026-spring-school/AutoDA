from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .transformer import Transformer


@dataclass
class DatasetContext:
    dataset_id: str
    df: pd.DataFrame
    target: str | None = None
    test_df: pd.DataFrame | None = None
    current_df: pd.DataFrame = field(init=False)
    working_df: pd.DataFrame | None = field(default=None, init=False)
    current_test_df: pd.DataFrame | None = field(init=False)
    working_test_df: pd.DataFrame | None = field(default=None, init=False)
    working_transformer: "Transformer | None" = field(default=None, init=False)
    fitted_transformers: "list[Transformer]" = field(default_factory=list, init=False)
    test_id_values: "pd.Series | None" = field(default=None, init=False)
    test_id_column_name: "str | None" = field(default=None, init=False)

    def __post_init__(self):
        self.current_df = self.df.copy()
        self.current_test_df = self.test_df.copy() if self.test_df is not None else None


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
            "n_unique": int(s.nunique(dropna=True)),
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
