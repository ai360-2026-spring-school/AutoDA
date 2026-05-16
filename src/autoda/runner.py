from dataclasses import dataclass
from typing import Any, Literal
import pandas as pd
import uuid

from .dataset import DatasetContext
from .graph import build_graph
from .models import make_model

@dataclass
class AgentResult:
    report: str
    iterations: list[dict[str, Any]]
    raw_state: dict[str, Any]


class PDAgent:
    def __init__(
        self,
        provider: Literal["yandex", "timeweb"],
        model: str | None = None,
        max_iterations: int = 8,
        temperature: float = 0,
        max_tokens: int = 1000,
    ):
        self.provider = provider
        self.model_name = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(
        self,
        df: pd.DataFrame,
        goal: str,
        target: str | None = None,
    ) -> AgentResult:
        dataset_id = str(uuid.uuid4())

        context = DatasetContext(
            dataset_id=dataset_id,
            df=df,
            target=target,
        )

        llm = make_model(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        app = build_graph(
            model=llm,
            context=context,
            max_iterations=self.max_iterations,
        )

        initial_state = {
            "goal": goal,
            "target": target,
            "dataset_id": dataset_id,
            "dataset_profile": {},
            "iterations": [],
            "current_step": 0,
            "proposed_action": None,
            "last_observation": None,
            "last_error": None,
            "decision": "continue",
            "final_report": None,
        }

        final_state = None
        last_printed_step = 0

        for state in app.stream(initial_state, stream_mode="values"):
            final_state = state

            iterations = state.get("iterations", [])

            if iterations:
                last_iteration = iterations[-1]
                step = last_iteration.get("step", 0)

                if step > last_printed_step:
                    last_printed_step = step

                    print("\n" + "=" * 80)
                    print(f"ITERATION {step}")
                    print("THOUGHT:", last_iteration.get("thought"))
                    print("ACTION:", last_iteration.get("action"))
                    print("CONCLUSION:", last_iteration.get("conclusion"))
                    print("=" * 80)

        if final_state is None:
            raise RuntimeError("Graph finished without returning state")

        return AgentResult(
            report=final_state["final_report"],
            iterations=final_state["iterations"],
            raw_state=final_state,
        )
