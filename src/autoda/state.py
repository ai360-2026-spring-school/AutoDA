from typing import Any, Literal, TypedDict
from typing_extensions import Annotated
import operator


class Iteration(TypedDict):
    step: int
    thought: str
    action: dict[str, Any]
    observation: dict[str, Any]
    conclusion: str


class AgentState(TypedDict):
    goal: str
    target: str | None

    dataset_id: str
    dataset_profile: dict[str, Any]

    iterations: Annotated[list[Iteration], operator.add]
    current_step: int

    proposed_action: dict[str, Any] | None
    last_observation: dict[str, Any] | None
    last_error: str | None

    decision: Literal["continue", "finish"]
    final_report: str | None
