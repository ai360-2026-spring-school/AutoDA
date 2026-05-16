import json
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END

from .state import AgentState, Iteration
from .dataset import DatasetContext
from .evaluator import CatBoostEvaluator, is_keep, CVResult
from .profiler import profile_dataset
from .actions.registry import REGISTRY, SCHEMA
from .prompts import build_planner_prompt, build_reflect_prompt, build_final_report_prompt


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse JSON: {e}", "raw": text}


def _signed_delta(new: CVResult, base: CVResult) -> float:
    if new.metric_direction == "max":
        return new.mean - base.mean
    return base.mean - new.mean


def build_graph(
    model,
    context: DatasetContext,
    evaluator: CatBoostEvaluator,
    *,
    max_iterations: int = 20,
    patience: int = 4,
    tolerance: float = 1e-4,
    reports_dir: Path = Path("reports"),
):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # baseline result stored as mutable cell so reflect_node can update it
    baseline_cell: list[CVResult | None] = [None]

    def profile_node(state: AgentState) -> dict[str, Any]:
        target = state["target"]

        # profile
        html_path = reports_dir / "profile_initial.html"
        try:
            dataset_profile = profile_dataset(context.current_df, target, html_path=html_path)
        except Exception as e:
            # ydata-profiling may not be installed in test envs
            from .dataset import profile_dataframe
            dataset_profile = profile_dataframe(context.current_df)
            dataset_profile["profile_error"] = str(e)

        # baseline CV
        X = context.current_df.drop(columns=[target])
        y = context.current_df[target]
        baseline = evaluator.cv(X, y, step=0)
        baseline_cell[0] = baseline

        # synthetic iteration-0 record
        baseline_iter: Iteration = {
            "step": 0,
            "thought": "baseline",
            "action": {"operation": "baseline"},
            "applied": True,
            "observation": {"summary": "initial CatBoost CV"},
            "cv_before": None,
            "cv_after": baseline.mean,
            "cv_delta": None,
            "cv_std_before": None,
            "cv_std_after": baseline.std,
            "decision": "keep",
            "conclusion": f"Baseline {evaluator.metric_name}={baseline.mean:.4f} ± {baseline.std:.4f}",
            "insight": None,
        }

        return {
            "dataset_profile": dataset_profile,
            "task": evaluator.task,
            "metric_name": evaluator.metric_name,
            "metric_direction": evaluator.metric_direction,
            "baseline_cv_mean": baseline.mean,
            "baseline_cv_std": baseline.std,
            "no_improve_streak": 0,
            "current_step": 0,
            "iterations": [baseline_iter],
            "insights": [],
            "applied_actions": [],
            "decision": "continue",
            "last_error": None,
            "final_report": None,
            "proposed_action": None,
            "last_observation": None,
        }

    def planner_node(state: AgentState) -> dict[str, Any]:
        if state.get("proposed_action", {}) and state["proposed_action"].get("stop"):
            return {"decision": "finish", "proposed_action": None}

        prompt = build_planner_prompt(state, SCHEMA, tolerance=tolerance)
        response = model.invoke(prompt)
        content = getattr(response, "content", str(response))
        action = parse_json_response(content)

        if "error" in action:
            return {"last_error": action["error"], "proposed_action": None}

        if action.get("stop"):
            return {"decision": "finish", "proposed_action": action, "last_error": None}

        return {"proposed_action": action, "last_error": None}

    def apply_node(state: AgentState) -> dict[str, Any]:
        action = state.get("proposed_action")

        if not action:
            return {"last_observation": None, "last_error": "no proposed action"}

        if action.get("stop"):
            return {"last_observation": None, "last_error": None}

        op = action.get("operation")
        args = action.get("args", {})

        if op not in REGISTRY:
            return {
                "last_observation": None,
                "last_error": f"unknown operation: {op!r}",
            }

        # drop_low_importance needs feature importances injected
        if op == "drop_low_importance":
            args = dict(args, feature_importances=evaluator.last_feature_importances_)

        try:
            df_new, observation = REGISTRY[op](context.current_df, state["target"], **args)
            context.working_df = df_new
            return {"last_observation": observation, "last_error": None}
        except Exception as e:
            context.working_df = None
            return {"last_observation": None, "last_error": repr(e)}

    def evaluate_node(state: AgentState) -> dict[str, Any]:
        # pass-through on error or when apply didn't produce a df
        if state.get("last_error") or context.working_df is None:
            return {}

        op = (state.get("proposed_action") or {}).get("operation", "")

        # insight actions don't change df → skip CV
        if op == "record_insight":
            return {}

        target = state["target"]
        X = context.working_df.drop(columns=[target], errors="ignore")
        y = context.working_df[target]

        try:
            result = evaluator.cv(X, y, step=state.get("current_step", 0) + 1)
        except Exception as e:
            return {"last_error": repr(e)}

        # store temporarily in observation so reflect_node can read it
        obs = dict(state.get("last_observation") or {})
        obs["_cv_result"] = result.as_dict()
        return {"last_observation": obs}

    def reflect_node(state: AgentState) -> dict[str, Any]:
        base = baseline_cell[0]
        action = state.get("proposed_action") or {}
        op = action.get("operation", "")
        error = state.get("last_error")
        obs = state.get("last_observation") or {}
        step = state.get("current_step", 0) + 1

        cv_result_dict = obs.pop("_cv_result", None) if obs else None
        cv_after_result: CVResult | None = None
        if cv_result_dict:
            cv_after_result = CVResult(**cv_result_dict)

        cv_before = base.mean if base else None
        cv_std_before = base.std if base else None
        cv_after = cv_after_result.mean if cv_after_result else None
        cv_std_after = cv_after_result.std if cv_after_result else None

        if error:
            iter_decision = "error"
            cv_delta = None
        elif op == "record_insight":
            iter_decision = "keep"
            cv_delta = None
        elif cv_after_result and base:
            kept = is_keep(cv_after_result, base, tol=tolerance)
            iter_decision = "keep" if kept else "reject"
            cv_delta = _signed_delta(cv_after_result, base)
        else:
            iter_decision = "error"
            cv_delta = None

        # ask LLM to reflect
        reflect_prompt = build_reflect_prompt(
            state,
            cv_before=cv_before,
            cv_after=cv_after,
            cv_delta=cv_delta,
            decision=iter_decision,
        )
        try:
            response = model.invoke(reflect_prompt)
            content = getattr(response, "content", str(response))
            reflection = parse_json_response(content)
        except Exception:
            reflection = {}

        conclusion = reflection.get("conclusion", "")
        new_insight = reflection.get("insight")

        iteration: Iteration = {
            "step": step,
            "thought": action.get("thought", ""),
            "action": action,
            "applied": iter_decision == "keep",
            "observation": obs,
            "cv_before": cv_before,
            "cv_after": cv_after,
            "cv_delta": cv_delta,
            "cv_std_before": cv_std_before,
            "cv_std_after": cv_std_after,
            "decision": iter_decision,
            "conclusion": conclusion,
            "insight": new_insight,
        }

        updates: dict[str, Any] = {
            "iterations": [iteration],
            "current_step": step,
            "last_error": None,
            "last_observation": obs,
        }

        if iter_decision == "keep":
            if op != "record_insight" and cv_after_result:
                context.current_df = context.working_df
                baseline_cell[0] = cv_after_result
                updates["baseline_cv_mean"] = cv_after_result.mean
                updates["baseline_cv_std"] = cv_after_result.std

                # re-profile on keep
                try:
                    html_path = reports_dir / f"profile_step_{step}.html"
                    new_profile = profile_dataset(context.current_df, state["target"], html_path=html_path)
                    updates["dataset_profile"] = new_profile
                except Exception:
                    pass

            updates["no_improve_streak"] = 0
            updates["applied_actions"] = [action]
        else:
            context.working_df = None
            updates["no_improve_streak"] = state.get("no_improve_streak", 0) + 1

        if new_insight:
            updates["insights"] = [new_insight]

        # stop conditions
        no_improve = updates.get("no_improve_streak", state.get("no_improve_streak", 0))
        if step >= max_iterations or no_improve >= patience or action.get("stop"):
            updates["decision"] = "finish"
        else:
            updates["decision"] = "continue"

        return updates

    def final_report_node(state: AgentState) -> dict[str, Any]:
        prompt = build_final_report_prompt(state)
        response = model.invoke(prompt)
        report = getattr(response, "content", str(response))

        # write artifacts
        (reports_dir / "final_report.md").write_text(report)
        try:
            context.current_df.to_parquet(reports_dir / "final_dataset.parquet", index=False)
        except Exception:
            pass

        return {"final_report": report}

    def route_after_planner(state: AgentState) -> Literal["apply", "final_report"]:
        if state.get("decision") == "finish":
            return "final_report"
        return "apply"

    def route_after_reflect(state: AgentState) -> Literal["planner", "final_report"]:
        if state.get("decision") == "finish":
            return "final_report"
        return "planner"

    graph = StateGraph(AgentState)

    graph.add_node("profile", profile_node)
    graph.add_node("planner", planner_node)
    graph.add_node("apply", apply_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "planner")
    graph.add_conditional_edges("planner", route_after_planner, {"apply": "apply", "final_report": "final_report"})
    graph.add_edge("apply", "evaluate")
    graph.add_edge("evaluate", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {"planner": "planner", "final_report": "final_report"})
    graph.add_edge("final_report", END)

    return graph.compile()
