from typing import Any, Literal, TypedDict
from typing_extensions import Annotated
import operator


class Iteration(TypedDict):
    step: int
    thought: str
    action: dict[str, Any]
    applied: bool
    observation: dict[str, Any]
    cv_before: float | None
    cv_after: float | None
    cv_delta: float | None
    cv_std_before: float | None
    cv_std_after: float | None
    decision: Literal["keep", "reject", "error"]
    conclusion: str
    insight: dict[str, Any] | None


class AgentState(TypedDict):
    goal: str
    target: str
    task: Literal["binary", "multiclass", "regression"]
    metric_name: str
    metric_direction: Literal["max", "min"]

    dataset_id: str
    dataset_profile: dict[str, Any]

    baseline_cv_mean: float | None
    baseline_cv_std: float | None

    has_test_df: bool

    iterations: Annotated[list[Iteration], operator.add]
    current_step: int

    proposed_action: dict[str, Any] | None
    last_observation: dict[str, Any] | None
    last_error: str | None

    insights: Annotated[list[dict[str, Any]], operator.add]
    applied_actions: Annotated[list[dict[str, Any]], operator.add]
    applied_pipeline: Annotated[list[dict[str, Any]], operator.add]
    info_tool_results: Annotated[list[dict[str, Any]], operator.add]

    decision: Literal["continue", "finish"]
    final_report: str | None
    dataset_description: str | None
    submission_path: str | None

    # v4: compact, structured signal derived from dataset_profile; injected
    # into planner and reflect prompts every turn (much cheaper than the raw
    # profile JSON). Re-built when current_df changes on a keep.
    profile_summary: dict[str, Any]

    # v4: deduped history of every action the planner has tried so far,
    # appended in reflect_node so the planner can see "don't repeat this".
    experiment_log: Annotated[list[dict[str, Any]], operator.add]
