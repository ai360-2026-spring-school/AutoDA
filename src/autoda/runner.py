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
from .state import Iteration
from .actions.registry import kind_of
from .prompts import (
    DESCRIPTION_TOKEN_BUDGET,
    summarize_description,
)


def sanitize_target(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "target"


def resolve_model_label(provider: str, model_name: str | None) -> str:
    """Pick the user-facing model label used in the reports subfolder.

    Resolution order:
      1. explicit ``model_name`` kwarg on PDAgent.
      2. provider-specific env var (TIMEWEB_MODEL / GIGACHAT_MODEL).
      3. provider-specific fallback ("timeweb-agent" / "GigaChat-2-Max").
    """
    if model_name:
        return model_name
    if provider == "timeweb":
        return os.getenv("TIMEWEB_MODEL", "timeweb-agent")
    if provider == "gigachat":
        return os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")
    return f"{provider}-unknown"


def _coerce_description(description: Any) -> str | None:
    """Accept str | Path | list[str] | None. Returns a single trimmed string or None.

    Lists are joined with blank lines so multi-paragraph inputs keep
    paragraph boundaries. A string that names an existing file is read.
    Anything else is converted with ``str()``.
    """
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
    iterations: list[Iteration]
    applied_actions: list[dict[str, Any]]
    applied_pipeline: list[dict[str, Any]]
    info_tool_results: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    baseline_cv: float
    final_cv: float
    final_df: pd.DataFrame
    final_test_df: pd.DataFrame | None
    fitted_transformers: list
    pipeline_path: Path
    raw_state: dict[str, Any]
    submission_path: Path | None
    submission_df: pd.DataFrame | None


class PDAgent:
    def __init__(
        self,
        provider: Literal["timeweb", "gigachat"] = "timeweb",
        model: str | None = None,
        max_iterations: int = 20,
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
        test_df: pd.DataFrame | None = None,
        reports_dir: Path = Path("reports"),
        *,
        description: str | Path | None = None,
        id_column: str | None = None,
    ) -> AgentResult:
        dataset_id = str(uuid.uuid4())

        # subfolder = <target>__<model_label> so runs across models live side-by-side
        model_label = resolve_model_label(self.provider, self.model_name)
        subfolder_name = f"{sanitize_target(target)}__{sanitize_target(model_label)}"
        reports_dir = Path(reports_dir) / subfolder_name
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Build LLM once — may be needed early for description summarisation
        llm = make_model(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # --- T16+T23: resolve dataset description (accept str | Path | list[str] | None) ---
        resolved_description: str | None = None
        raw_text = _coerce_description(description)
        if raw_text:
            (reports_dir / "dataset_description.txt").write_text(raw_text)
            if len(raw_text) > DESCRIPTION_TOKEN_BUDGET:
                raw_text = summarize_description(llm, raw_text)
                (reports_dir / "dataset_description.summary.txt").write_text(raw_text)
            resolved_description = raw_text

        # --- T19: handle id column before constructing DatasetContext ---
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

        # Set id metadata on context
        context.test_id_values = test_id_values
        context.test_id_column_name = test_id_column_name

        evaluator = CatBoostEvaluator.auto(
            df[target],
            override_task=self.task,
            n_splits=self.n_splits,
        )

        if self.metric:
            evaluator.metric_name = self.metric

        # --- T15: point evaluator history at per-target subfolder ---
        evaluator.history_path = reports_dir / "cv_history.jsonl"

        app = build_graph(
            model=llm,
            context=context,
            evaluator=evaluator,
            max_iterations=self.max_iterations,
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
            "has_test_df": test_df is not None,
            "iterations": [],
            "current_step": 0,
            "proposed_action": None,
            "last_observation": None,
            "last_error": None,
            "insights": [],
            "applied_actions": [],
            "applied_pipeline": [],
            "info_tool_results": [],
            "decision": "continue",
            "final_report": None,
            "dataset_description": resolved_description,
            "submission_path": None,
        }

        final_state: dict[str, Any] | None = None
        last_printed_step = 0
        analyze_printed = False

        for state in app.stream(initial_state, stream_mode="values"):
            final_state = state

            # print analyze summary once when insights arrive
            if not analyze_printed and state.get("insights"):
                det = sum(1 for i in state["insights"] if i.get("source") == "deterministic")
                llm_c = sum(1 for i in state["insights"] if i.get("source") == "llm")
                err_c = sum(1 for i in state["insights"] if i.get("source") == "analyze_llm_error")
                if det + llm_c + err_c > 0:
                    suffix = f" + {err_c} parse error(s)" if err_c else ""
                    print(f"[analyze] deterministic={det}  llm={llm_c}{suffix}")
                    analyze_printed = True

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
                metric_name = state.get("metric_name", "metric")

                # determine if this is an info op
                try:
                    is_info = kind_of(op) == "info"
                except KeyError:
                    is_info = False

                if is_info:
                    obs = it.get("observation") or {}
                    summary = obs.get("summary", str(obs)[:100])
                    print(f"[step {step}] info     {op}({args_str})")
                    print(f"         logged: {summary}")
                else:
                    arrow = (
                        f"{cv_before:.4f} -> {cv_after:.4f}  ({delta:+.4f})"
                        if cv_before is not None and cv_after is not None and delta is not None
                        else "n/a"
                    )
                    print(f"[step {step}] {dec:<7} {op}({args_str})")
                    print(f"         {metric_name}: {arrow}")

        if final_state is None:
            raise RuntimeError("Graph finished without returning state")

        baseline_cv = (final_state.get("iterations") or [{}])[0].get("cv_after") or 0.0
        final_cv = final_state.get("baseline_cv_mean") or baseline_cv

        pipeline_path = reports_dir / "fitted_pipeline.pkl"

        sub_path_str = final_state.get("submission_path")
        sub_path = Path(sub_path_str) if sub_path_str else None
        sub_df = pd.read_csv(sub_path) if sub_path and sub_path.exists() else None

        return AgentResult(
            report=final_state.get("final_report") or "",
            iterations=final_state.get("iterations", []),
            applied_actions=final_state.get("applied_actions", []),
            applied_pipeline=final_state.get("applied_pipeline", []),
            info_tool_results=final_state.get("info_tool_results", []),
            insights=final_state.get("insights", []),
            baseline_cv=baseline_cv,
            final_cv=final_cv,
            final_df=context.current_df,
            final_test_df=context.current_test_df,
            fitted_transformers=context.fitted_transformers,
            pipeline_path=pipeline_path,
            raw_state=final_state,
            submission_path=sub_path,
            submission_df=sub_df,
        )
