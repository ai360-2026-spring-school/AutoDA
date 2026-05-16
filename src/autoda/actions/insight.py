from typing import Any
import pandas as pd


def record_insight(
    df: pd.DataFrame,
    target: str,
    title: str,
    body: str,
    evidence: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not title:
        raise ValueError("title cannot be empty")
    if not body:
        raise ValueError("body cannot be empty")

    observation = {
        "changed_columns": [],
        "summary": f"insight recorded: {title!r}",
        "insight": {
            "title": title,
            "body": body,
            "evidence": evidence or {},
        },
    }
    return df, observation
