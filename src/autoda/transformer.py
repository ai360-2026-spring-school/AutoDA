from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd


@dataclass
class Transformer:
    operation: str
    args: dict[str, Any]
    state: dict[str, Any]
    apply: Callable[["pd.DataFrame"], "pd.DataFrame"]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)
