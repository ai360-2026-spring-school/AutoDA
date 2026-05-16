from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd
from .transformer import Transformer


def load_pipeline(path: Path) -> list[Transformer]:
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_pipeline(df: pd.DataFrame, transformers: list[Transformer]) -> pd.DataFrame:
    for t in transformers:
        df = t.apply(df)
    return df


def replay(
    test_df: pd.DataFrame,
    pipeline_path: Path,
) -> pd.DataFrame:
    return apply_pipeline(test_df, load_pipeline(pipeline_path))
