from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .dataset import DatasetContext
from .evaluator import CatBoostEvaluator
from .graph import build_graph
from .models import make_model


def sanitize_target(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "target"


def resolve_model_label(provider: str, model_name: str | None) -> str:
    if model_name:
        return model_name
    if provider == "timeweb":
        return os.getenv("TIMEWEB_MODEL", "timeweb-agent")
    if provider == "gigachat":
        return os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")
    return f"{provider}-unknown"


def _coerce_description(description: Any) -> str | None:
    if description is None:
        return None
    if isinstance(description, Path):
        return description.read_text().strip() or None
    if isinstance(description, str):
        try:
            if Path(description).exists():
                return Path(description).read_text().strip() or None
        except (OSError, ValueError):
            pass
        return description.strip() or None
    if isinstance(description, (list, tuple)):
        parts: list[str] = []
        for item in description:
            if item is None:
                continue
            parts.append(str(item).strip())
        text = "\n\n".join(p for p in parts if p)
        return text or None
    return str(description).strip() or None


@dataclass
class AgentResult:
    report: str
    experiment_log: list[dict[str, Any]]
    applied_pipeline: list[dict[str, Any]]
    baseline_cv: float
    final_cv: float
    final_df: pd.DataFrame
    final_test_df: pd.DataFrame | None
    fitted_transformers: list
    pipeline_path: Path
    raw_state: dict[str, Any]
    submission_path: Path | None
    submission_df: pd.DataFrame | None
    column_type_map: dict[str, str]
    target_correlation_stats: dict[str, Any]


class PDAgent:
    def __init__(
        self,
        provider: Literal["timeweb", "gigachat"] = "timeweb",
        model: str | None = None,
        agent_id: str | None = None,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        n_splits: int = 5,
        task: str | None = None,
        metric: str | None = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        max_inner_turns: int = 15,
        debug: bool = False,
        critic_every: int = 3,
        critic_provider: Literal["timeweb", "gigachat"] | None = None,
        critic_model_name: str | None = None,
        critic_agent_id: str | None = None,
        n_ideators: int = 0,
        ideator_provider: Literal["timeweb", "gigachat"] | None = None,
        ideator_model_name: str | None = None,
        ideator_agent_id: str | None = None,
    ):
        self.provider = provider
        self.model_name = model
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_splits = n_splits
        self.task = task
        self.metric = metric
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_inner_turns = max_inner_turns
        self.debug = debug
        self.critic_every = critic_every
        self.critic_provider = critic_provider
        self.critic_model_name = critic_model_name
        self.critic_agent_id = critic_agent_id
        self.n_ideators = n_ideators
        self.ideator_provider = ideator_provider
        self.ideator_model_name = ideator_model_name
        self.ideator_agent_id = ideator_agent_id

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        goal: str,
        test_df: pd.DataFrame | None = None,
        reports_dir: Path = Path("reports"),
        *,
        description: str | Path | None = None,
        id_column: str | None = None,
        ohe_max_cardinality: int = 12,
        numeric_unique_threshold: int = 12,
        oversample: bool = True,
    ) -> AgentResult:
        if self.debug:
            print(f"[autoda] debug=True | provider={self.provider} | target={target!r}", flush=True)

        dataset_id = str(uuid.uuid4())

        model_label = resolve_model_label(self.provider, self.model_name)
        subfolder_name = f"{sanitize_target(target)}__{sanitize_target(model_label)}"
        reports_dir = Path(reports_dir) / subfolder_name
        reports_dir.mkdir(parents=True, exist_ok=True)

        llm = make_model(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            agent_id=self.agent_id,
        )

        critic_llm = None
        if self.critic_provider is not None:
            critic_llm = make_model(
                provider=self.critic_provider,
                model_name=self.critic_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                agent_id=self.critic_agent_id,
            )
            if self.debug:
                label = self.critic_model_name or self.critic_agent_id or self.critic_provider
                print(f"[autoda] critic={label}", flush=True)

        ideator_llm = None
        if self.n_ideators > 0:
            _ideator_prov = self.ideator_provider or self.provider
            ideator_llm = make_model(
                provider=_ideator_prov,
                model_name=self.ideator_model_name,
                temperature=0.7,  # slight warmth for diversity
                max_tokens=400,   # short answers only
                agent_id=self.ideator_agent_id,
            )
            if self.debug:
                from .prompts import IDEATOR_ROLES as _ROLES
                _roles_preview = ", ".join(r["name"] for r in _ROLES[:self.n_ideators])
                print(f"[autoda] ideators={self.n_ideators} ({_roles_preview})", flush=True)

        # Resolve description (raw text — summarisation happens in summarise_node)
        resolved_description: str | None = _coerce_description(description)
        if resolved_description:
            try:
                (reports_dir / "dataset_description.txt").write_text(resolved_description)
            except Exception:
                pass

        # Handle id column extraction from test_df
        test_id_values: pd.Series | None = None
        test_id_column_name: str | None = None
        if test_df is not None:
            test_df = test_df.copy()
            id_col = id_column or ("id" if "id" in test_df.columns else None)
            if id_col is not None and id_col in test_df.columns:
                test_id_column_name = id_col
                test_id_values = test_df[id_col].copy()
                test_df = test_df.drop(columns=[id_col])
            else:
                test_id_column_name = test_df.index.name or "id"
                test_id_values = pd.Series(test_df.index, name=test_id_column_name)

        context = DatasetContext(
            dataset_id=dataset_id,
            df=df,
            target=target,
            test_df=test_df,
        )
        context.test_id_values = test_id_values
        context.test_id_column_name = test_id_column_name

        evaluator = CatBoostEvaluator.auto(
            df[target],
            override_task=self.task,
            n_splits=self.n_splits,
        )
        if self.metric:
            evaluator.metric_name = self.metric
        evaluator.history_path = reports_dir / "cv_history.jsonl"

        app = build_graph(
            model=llm,
            context=context,
            evaluator=evaluator,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            reports_dir=reports_dir,
            ohe_max_cardinality=ohe_max_cardinality,
            numeric_unique_threshold=numeric_unique_threshold,
            oversample=oversample,
            max_inner_turns=self.max_inner_turns,
            debug=self.debug,
            critic_every=self.critic_every,
            critic_model=critic_llm,
            n_ideators=self.n_ideators,
            ideator_model=ideator_llm,
        )

        initial_state = {
            "goal": goal,
            "target": target,
            "task": evaluator.task,
            "metric_name": evaluator.metric_name,
            "metric_direction": evaluator.metric_direction,
            "dataset_id": dataset_id,
            "baseline_cv_mean": None,
            "baseline_cv_std": None,
            "has_test_df": test_df is not None,
            "current_step": 0,
            "decision": "continue",
            "final_report": None,
            "dataset_description": resolved_description,
            "submission_path": None,
            "experiment_log": [],
            "applied_pipeline": [],
            "column_type_map": {},
            "target_correlation_stats": {},
            "feature_columns": [],
            "long_description_summary": None,
            "short_description_summary": None,
            "planner_memory": [],
            "critic_message": None,
            "current_iteration_transforms": [],
            "iteration_start_cv": None,
            "planner_addendum": None,
            "planner_turn_count": 0,
            "ideator_suggestions": [],
        }

        final_state: dict[str, Any] | None = None
        last_printed_step = -1

        for state in app.stream(initial_state, stream_mode="values"):
            final_state = state

            # Print progress on experiment_log updates
            exp_log = state.get("experiment_log", [])
            for entry in exp_log:
                step = entry.get("step", 0)
                if step <= last_printed_step:
                    continue
                last_printed_step = step
                decision = entry.get("decision", "?")
                delta = entry.get("delta", 0.0)
                cv_after = entry.get("cv_after") or 0.0
                transforms = entry.get("transforms", [])
                metric_name = state.get("metric_name", "cv")
                ops_str = ", ".join(t.get("op", "?") for t in transforms) or "(none)"
                print(
                    f"[step {step}] {decision:<7} "
                    f"{metric_name}={cv_after:.4f} ({delta:+.4f}) | {ops_str}"
                )

        if final_state is None:
            raise RuntimeError("Graph finished without returning state")

        # Resolve baseline and final CV
        baseline_cv = final_state.get("baseline_cv_mean") or 0.0
        exp_log = final_state.get("experiment_log", [])
        kept = [e for e in exp_log if e.get("decision") == "keep"]
        final_cv = kept[-1]["cv_after"] if kept else baseline_cv

        pipeline_path = reports_dir / "fitted_pipeline.pkl"
        sub_path_str = final_state.get("submission_path")
        sub_path = Path(sub_path_str) if sub_path_str else None
        sub_df = pd.read_csv(sub_path) if sub_path and sub_path.exists() else None

        return AgentResult(
            report=final_state.get("final_report") or "",
            experiment_log=exp_log,
            applied_pipeline=final_state.get("applied_pipeline", []),
            baseline_cv=baseline_cv,
            final_cv=final_cv,
            final_df=context.current_df,
            final_test_df=context.current_test_df,
            fitted_transformers=context.fitted_transformers,
            pipeline_path=pipeline_path,
            raw_state=final_state,
            submission_path=sub_path,
            submission_df=sub_df,
            column_type_map=final_state.get("column_type_map", {}),
            target_correlation_stats=final_state.get("target_correlation_stats", {}),
        )
