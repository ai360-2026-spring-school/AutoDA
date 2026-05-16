from typing import Any
import pandas as pd


class PandasTools:
    def __init__(self, df: pd.DataFrame, target: str | None = None):
        self.df = df
        self.target = target

    def describe_columns(self, columns: list[str]) -> dict[str, Any]:
        out = {}

        for col in columns:
            if col not in self.df.columns:
                out[col] = {"error": "column not found"}
                continue

            s = self.df[col]
            out[col] = {
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(s):
                out[col]["describe"] = {
                    k: float(v)
                    for k, v in s.describe().to_dict().items()
                    if pd.notna(v)
                }
            else:
                out[col]["top_values"] = (
                    s.value_counts(dropna=False).head(10).astype(int).to_dict()
                )

        return out

    def groupby_agg(
        self,
        by: str,
        value: str,
        agg: str = "mean",
    ) -> dict[str, Any]:
        if by not in self.df.columns:
            return {"error": f"column not found: {by}"}

        if value not in self.df.columns:
            return {"error": f"column not found: {value}"}

        if agg not in {"mean", "median", "sum", "count", "min", "max"}:
            return {"error": f"unsupported agg: {agg}"}

        result = (
            self.df.groupby(by, dropna=False)[value]
            .agg(agg)
            .sort_values(ascending=False)
            .head(30)
        )

        return {
            "by": by,
            "value": value,
            "agg": agg,
            "result": result.to_dict(),
        }

    def correlation_with_target(self) -> dict[str, Any]:
        if self.target is None:
            return {"error": "target is not set"}

        if self.target not in self.df.columns:
            return {"error": f"target column not found: {self.target}"}

        numeric = self.df.select_dtypes(include="number")

        if self.target not in numeric.columns:
            return {"error": "target is not numeric"}

        corr = (
            numeric.corr(numeric_only=True)[self.target]
            .drop(labels=[self.target], errors="ignore")
            .dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(30)
        )

        return {
            "target": self.target,
            "correlations": corr.to_dict(),
        }
