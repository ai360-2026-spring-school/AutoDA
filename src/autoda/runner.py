from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .dataset import DatasetContext
from .evaluator import CatBoostEvaluator
from .graph import build_graph
from .models import make_model
from .state import Iteration


@dataclass
class AgentResult:
    report: str
    iterations: list[Iteration]
    applied_actions: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    baseline_cv: float
    final_cv: float
    final_df: pd.DataFrame
    raw_state: dict[str, Any]


class PDAgent:
    def __init__(
        self,
        provider: Literal["timeweb"] = "timeweb",
        model: str | None = None,
        max_iterations: int = 20,
        patience: int = 4,
        tolerance: float = 1e-4,
        n_splits: int = 5,
        task: str | None = None,
        metric: str | None = None,
        temperature: float = 0,
        max_tokens: int = 2000,
    ):
        self.provider = provider
        self.model_name = model
        self.max_iterations = max_iterations
        self.patience = patience
        self.tolerance = tolerance
        self.n_splits = n_splits
        self.task = task
        self.metric = metric
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        goal: str,
        reports_dir: Path = Path("reports"),
    ) -> AgentResult:
        dataset_id = str(uuid.uuid4())

        context = DatasetContext(
            dataset_id=dataset_id,
            df=df,
            target=target,
        )

        evaluator = CatBoostEvaluator.auto(
            df[target],
            override_task=self.task,
            n_splits=self.n_splits,
        )

        if self.metric:
            evaluator.metric_name = self.metric

        llm = make_model(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        app = build_graph(
            model=llm,
            context=context,
            evaluator=evaluator,
            max_iterations=self.max_iterations,
            patience=self.patience,
            tolerance=self.tolerance,
            reports_dir=reports_dir,
        )

        initial_state = {
            "goal": goal,
            "target": target,
            "task": evaluator.task,
            "metric_name": evaluator.metric_name,
            "metric_direction": evaluator.metric_direction,
            "dataset_id": dataset_id,
            "dataset_profile": {},
            "baseline_cv_mean": None,
            "baseline_cv_std": None,
            "no_improve_streak": 0,
            "iterations": [],
            "current_step": 0,
            "proposed_action": None,
            "last_observation": None,
            "last_error": None,
            "insights": [],
            "applied_actions": [],
            "decision": "continue",
            "final_report": None,
        }

        final_state: dict[str, Any] | None = None
        last_printed_step = 0

        for state in app.stream(initial_state, stream_mode="values"):
            final_state = state
            iterations = state.get("iterations", [])

            for it in iterations:
                step = it.get("step", 0)
                if step <= last_printed_step or step == 0:
                    continue
                last_printed_step = step

                op = (it.get("action") or {}).get("operation", "?")
                args = (it.get("action") or {}).get("args", {})
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                dec = it.get("decision", "?")
                cv_before = it.get("cv_before")
                cv_after = it.get("cv_after")
                delta = it.get("cv_delta")
                metric = state.get("metric_name", "metric")

                if op == "record_insight":
                    insight_title = (it.get("observation") or {}).get("insight", {}).get("title", "")
                    print(f"[step {step}] insight  {insight_title!r}")
                else:
                    arrow = f"{cv_before:.4f} -> {cv_after:.4f}  ({delta:+.4f})" if cv_before is not None and cv_after is not None and delta is not None else "n/a"
                    print(f"[step {step}] {dec:<7} {op}({args_str})")
                    print(f"         {metric}: {arrow}")

        if final_state is None:
            raise RuntimeError("Graph finished without returning state")

        baseline_cv = (final_state.get("iterations") or [{}])[0].get("cv_after") or 0.0
        final_cv = final_state.get("baseline_cv_mean") or baseline_cv

        return AgentResult(
            report=final_state.get("final_report") or "",
            iterations=final_state.get("iterations", []),
            applied_actions=final_state.get("applied_actions", []),
            insights=final_state.get("insights", []),
            baseline_cv=baseline_cv,
            final_cv=final_cv,
            final_df=context.current_df,
            raw_state=final_state,
        )
