from typing import Any, Literal, TypedDict
from typing_extensions import Annotated
import operator


class AgentState(TypedDict):
    # --- Task metadata ---
    goal: str
    target: str
    task: Literal["binary", "multiclass", "regression"]
    metric_name: str
    metric_direction: Literal["max", "min"]
    dataset_id: str

    # --- Baseline CV (set in preprocess_node) ---
    baseline_cv_mean: float | None
    baseline_cv_std: float | None

    has_test_df: bool

    # --- Loop control ---
    current_step: int
    decision: Literal["continue", "finish"]
    final_report: str | None
    submission_path: str | None

    # --- Accumulator fields (append-only via operator.add) ---
    experiment_log: Annotated[list[dict[str, Any]], operator.add]
    applied_pipeline: Annotated[list[dict[str, Any]], operator.add]

    # --- Preprocessing outputs (set once in preprocess_node) ---
    column_type_map: dict[str, Literal["NUMERIC", "CATEGORICAL"]]
    target_correlation_stats: dict[str, Any]
    feature_columns: list[str]

    # --- Description summaries (set once in summarise_node) ---
    dataset_description: str | None
    long_description_summary: str | None   # ~1000 chars, for critic
    short_description_summary: str | None  # ~300 chars, for planner

    # --- Planner memory (replace-semantics, not operator.add) ---
    planner_memory: list[str]

    # --- Critic output ---
    critic_message: str | None

    # --- Intra-iteration tracking ---
    current_iteration_transforms: list[dict[str, Any]]
    iteration_start_cv: float | None

    # --- Planner turn control ---
    planner_addendum: dict[str, Any] | None
    planner_turn_count: int
