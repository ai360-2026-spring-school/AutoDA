from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END

from .state import AgentState, Iteration
from .dataset import DatasetContext
from .evaluator import CatBoostEvaluator, is_keep, CVResult
from .profiler import profile_dataset
from .profile_summary import build_profile_summary
from .experiment_log import make_experiment_entry as _make_experiment_entry
from .actions.registry import REGISTRY, SCHEMA, kind_of
from .experiment_log import args_signature as _args_sig
from .prompts import (
    build_planner_prompt,
    build_reflect_prompt,
    build_final_report_prompt,
    build_analyze_prompt
)


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


def _find_duplicate(action: dict, exp_log: list) -> dict | None:
    """Return the experiment_log entry if (operation, args) was already kept/rejected."""
    op = action.get("operation", "")
    sig = _args_sig(action.get("args", {}))
    for e in exp_log:
        if e.get("operation") == op and e.get("args_signature") == sig:
            if e.get("decision") in ("keep", "reject"):
                return e
    return None


def build_graph(
    model,
    context: DatasetContext,
    evaluator: CatBoostEvaluator,
    *,
    max_iterations: int = 20,
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

        # v4: compact profile summary used by every prompt
        profile_summary = build_profile_summary(context.current_df, target, dataset_profile)

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
            "profile_summary": profile_summary,
            "task": evaluator.task,
            "metric_name": evaluator.metric_name,
            "metric_direction": evaluator.metric_direction,
            "baseline_cv_mean": baseline.mean,
            "baseline_cv_std": baseline.std,
            "has_test_df": context.test_df is not None,
            "current_step": 0,
            "iterations": [baseline_iter],
            "insights": [],
            "applied_actions": [],
            "applied_pipeline": [],
            "info_tool_results": [],
            "experiment_log": [],
            "decision": "continue",
            "last_error": None,
            "final_report": None,
            "proposed_action": None,
            "last_observation": None,
        }

    def analyze_node(state: AgentState) -> dict[str, Any]:
        """Deterministic + LLM analysis of the initial profile."""
        profile = state.get("dataset_profile", {})
        columns = profile.get("columns", [])
        target = state["target"]
        task = state["task"]
        insights: list[dict[str, Any]] = []

        # --- deterministic insights ---
        # Columns with >50% missing
        high_missing = [
            c["name"] for c in columns
            if c["name"] != target and c.get("missing_rate", 0) > 0.5
        ]
        if high_missing:
            insights.append({
                "title": "High missingness columns",
                "body": f"Columns with >50% missing values: {high_missing}. Consider imputing or dropping.",
                "evidence": {"columns": high_missing},
                "source": "deterministic",
            })

        # Suspected leakage: unique_rate > 0.9 and not target
        n_rows = profile.get("shape", [0])[0] or 1
        leakage_candidates = []
        for c in columns:
            if c["name"] == target:
                continue
            unique_rate = c.get("n_unique", 0) / n_rows
            if unique_rate > 0.9:
                leakage_candidates.append(c["name"])
        if leakage_candidates:
            insights.append({
                "title": "Potential leakage columns (high unique rate)",
                "body": f"Columns with unique_rate > 0.9 (not target): {leakage_candidates}. May be IDs or timestamps.",
                "evidence": {"columns": leakage_candidates},
                "source": "deterministic",
            })

        # High cardinality categorical cols (n_unique > 20, non-numeric)
        high_card_cats = [
            c["name"] for c in columns
            if c["name"] != target
            and c.get("n_unique", 0) > 20
            and "int" not in c.get("dtype", "")
            and "float" not in c.get("dtype", "")
        ]
        if high_card_cats:
            insights.append({
                "title": "High-cardinality categorical columns",
                "body": f"Categorical cols with >20 unique values: {high_card_cats}. Use frequency or target encoding.",
                "evidence": {"columns": high_card_cats},
                "source": "deterministic",
            })

        # Datetime-typed columns
        datetime_cols = [
            c["name"] for c in columns
            if "datetime" in c.get("dtype", "") or "date" in c.get("dtype", "").lower()
        ]
        if datetime_cols:
            insights.append({
                "title": "Datetime columns detected",
                "body": f"Datetime columns: {datetime_cols}. Consider expand_datetime to extract year/month/dow/hour/is_weekend.",
                "evidence": {"columns": datetime_cols},
                "source": "deterministic",
            })

        # Class imbalance for binary tasks
        if task == "binary":
            target_col_info = next((c for c in columns if c["name"] == target), None)
            if target_col_info and "top_values" in target_col_info:
                top_vals = target_col_info["top_values"]
                if top_vals:
                    total = sum(top_vals.values())
                    minority_frac = min(top_vals.values()) / total if total > 0 else 0.5
                    if minority_frac < 0.3:
                        insights.append({
                            "title": "Class imbalance detected",
                            "body": f"Minority class fraction: {minority_frac:.2%}. Consider class-weight strategies.",
                            "evidence": {"minority_fraction": minority_frac, "distribution": top_vals},
                            "source": "deterministic",
                        })

        # Strong correlations between numeric features (detected from profile stats)
        # We can't compute full corr matrix here, but we note columns with similar stats
        numeric_cols = [c for c in columns if "stats" in c and c["name"] != target]
        if len(numeric_cols) >= 2:
            insights.append({
                "title": "Numeric features available",
                "body": f"{len(numeric_cols)} numeric features found. Consider drop_high_corr to remove redundant features.",
                "evidence": {"count": len(numeric_cols)},
                "source": "deterministic",
            })

        # --- LLM insights ---
        n_llm_insights = 0
        parse_ok = True
        analyze_prompt = build_analyze_prompt(state)
        prompt_chars = len(analyze_prompt)
        response_chars = 0
        try:
            response = model.invoke(analyze_prompt)
            content = getattr(response, "content", str(response))
            response_chars = len(content)
            parsed = parse_json_response(content)
            if "error" in parsed:
                parse_ok = False
                insights.append({
                    "title": "analyze_node JSON parse failure",
                    "body": parsed["error"],
                    "evidence": {"raw_excerpt": parsed.get("raw", "")[:500]},
                    "source": "analyze_llm_error",
                })
            else:
                llm_insights = parsed.get("insights", [])
                if isinstance(llm_insights, list):
                    for ins in llm_insights:
                        if isinstance(ins, dict):
                            ins.setdefault("source", "llm")
                            insights.append(ins)
                    n_llm_insights = len(llm_insights)
        except Exception as exc:
            parse_ok = False
            insights.append({
                "title": "analyze_node exception",
                "body": repr(exc),
                "evidence": {},
                "source": "analyze_llm_error",
            })

        # write debug log
        try:
            log_path = reports_dir / "analyze_debug.log"
            log_entry = json.dumps({
                "timestamp": _dt.datetime.utcnow().isoformat(),
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "parse_ok": parse_ok,
                "n_deterministic_insights": len([i for i in insights if i.get("source") in ("deterministic",)]),
                "n_llm_insights": n_llm_insights,
            })
            with log_path.open("a") as lf:
                lf.write(log_entry + "\n")
        except Exception:
            pass

        return {"insights": insights, "has_test_df": context.test_df is not None}

    def planner_node(state: AgentState) -> dict[str, Any]:
        if state.get("proposed_action", {}) and state["proposed_action"].get("stop"):
            return {"decision": "finish", "proposed_action": None}

        exp_log = state.get("experiment_log", [])
        dupe_warning = ""
        action: dict[str, Any] = {}
        for attempt in range(3):
            prompt = build_planner_prompt(state, SCHEMA, tolerance=tolerance)
            if dupe_warning:
                prompt = dupe_warning + "\n\n" + prompt
            response = model.invoke(prompt)
            content = getattr(response, "content", str(response))
            action = parse_json_response(content)

            if "error" in action:
                return {"last_error": action["error"], "proposed_action": None}
            if action.get("stop"):
                return {"decision": "finish", "proposed_action": action, "last_error": None}

            dupe = _find_duplicate(action, exp_log)
            if dupe is None:
                return {"proposed_action": action, "last_error": None}

            dupe_warning = (
                f"HARD REJECTION (attempt {attempt + 1}): you proposed "
                f"{action.get('operation')}({action.get('args')}) which was already "
                f"{dupe['decision'].upper()} at step {dupe['step']}. "
                "You MUST propose a genuinely different operation or meaningfully different args."
            )

        return {
            "last_error": f"Planner proposed duplicate 3 times: {action.get('operation')}",
            "proposed_action": None,
        }

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

        # determine kind, defaulting to transformer on error
        try:
            op_kind = kind_of(op)
        except KeyError:
            op_kind = "transformer"

        if op_kind == "info":
            # info tools: call with task kwarg, return observation dict directly
            try:
                obs = REGISTRY[op](context.current_df, state["target"], task=state["task"], **args)
                return {"last_observation": obs, "last_error": None}
            except Exception as e:
                return {"last_observation": None, "last_error": repr(e)}
        else:
            # transformer: inject feature_importances for drop_low_importance
            if op == "drop_low_importance":
                args = dict(args, feature_importances=evaluator.last_feature_importances_)

            try:
                df_new, transformer, observation = REGISTRY[op](context.current_df, state["target"], **args)
                context.working_df = df_new
                context.working_transformer = transformer

                # sync test df
                if context.current_test_df is not None:
                    try:
                        context.working_test_df = transformer.apply(context.current_test_df)
                    except Exception as e:
                        context.working_test_df = None
                        observation = dict(observation)
                        observation["test_apply_error"] = repr(e)

                return {"last_observation": observation, "last_error": None}
            except Exception as e:
                context.working_df = None
                context.working_transformer = None
                context.working_test_df = None
                return {"last_observation": None, "last_error": repr(e)}

    def evaluate_node(state: AgentState) -> dict[str, Any]:
        # pass-through on error or when apply didn't produce a df
        if state.get("last_error") or context.working_df is None:
            return {}

        op = (state.get("proposed_action") or {}).get("operation", "")

        # info operations don't change the df — skip CV
        try:
            if kind_of(op) == "info":
                return {}
        except KeyError:
            pass  # unknown op — fall through to CV

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

        # determine operation kind
        try:
            op_kind = kind_of(op)
        except KeyError:
            op_kind = "transformer"

        obs = dict(obs)  # make mutable
        cv_result_dict = obs.pop("_cv_result", None)
        cv_after_result: CVResult | None = None
        if cv_result_dict:
            cv_after_result = CVResult(**cv_result_dict)

        cv_before = base.mean if base else None
        cv_std_before = base.std if base else None
        cv_after = cv_after_result.mean if cv_after_result else None
        cv_std_after = cv_after_result.std if cv_after_result else None

        # determine iteration decision
        if error:
            iter_decision = "error"
            cv_delta = None
        elif op_kind == "info":
            # info ops are always "keep" (or error if obs has "error")
            if obs.get("error"):
                iter_decision = "error"
            else:
                iter_decision = "keep"
            cv_delta = None
        elif cv_after_result and base:
            kept = is_keep(cv_after_result, base, tol=tolerance)
            iter_decision = "keep" if kept else "reject"
            cv_delta = _signed_delta(cv_after_result, base)
        else:
            iter_decision = "error"
            cv_delta = None

        # LLM reflection
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
            if op_kind == "info":
                # record info tool result as insight
                info_insight = {
                    "title": f"{op} result",
                    "body": obs.get("summary", str(obs)[:200]),
                    "evidence": obs,
                    "source": "info_tool",
                }
                updates["insights"] = [info_insight]
                updates["info_tool_results"] = [{"operation": op, "args": action.get("args", {}), "result": obs}]
            else:
                # transformer keep
                context.current_df = context.working_df
                baseline_cell[0] = cv_after_result
                updates["baseline_cv_mean"] = cv_after_result.mean
                updates["baseline_cv_std"] = cv_after_result.std

                # append to fitted pipeline
                context.fitted_transformers.append(context.working_transformer)
                transformer_id = len(context.fitted_transformers) - 1
                updates["applied_pipeline"] = [{
                    "step": step,
                    "operation": op,
                    "args": action.get("args", {}),
                    "transformer_id": transformer_id,
                }]

                # promote working test df
                if context.working_test_df is not None:
                    context.current_test_df = context.working_test_df

                updates["applied_actions"] = [action]

                # re-profile on keep
                refreshed_profile = None
                try:
                    html_path = reports_dir / f"profile_step_{step}.html"
                    refreshed_profile = profile_dataset(
                        context.current_df, state["target"], html_path=html_path
                    )
                    updates["dataset_profile"] = refreshed_profile
                except Exception:
                    refreshed_profile = state.get("dataset_profile")

                # rebuild compact summary against the new df
                try:
                    updates["profile_summary"] = build_profile_summary(
                        context.current_df, state["target"], refreshed_profile or {}
                    )
                except Exception:
                    pass

        else:
            # reject or error
            context.working_df = None
            context.working_transformer = None
            context.working_test_df = None

        if new_insight:
            updates.setdefault("insights", [])
            updates["insights"] = updates.get("insights", []) + [new_insight]

        # v4: always record this attempt in the experiment log so the planner
        # next turn knows what's been tried (and won't propose duplicates).
        updates["experiment_log"] = [
            _make_experiment_entry(
                step=step,
                operation=op,
                kind=op_kind,
                args=action.get("args", {}),
                decision=iter_decision,
                cv_delta=cv_delta,
                obs=obs,
                error=error,
            )
        ]

        # stop conditions
        stop = step >= max_iterations or bool(action.get("stop"))
        updates["decision"] = "finish" if stop else "continue"

        return updates

    def final_report_node(state: AgentState) -> dict[str, Any]:
        import pickle as _pickle
        import json as _json

        prompt = build_final_report_prompt(state)
        response = model.invoke(prompt)
        report = getattr(response, "content", str(response))

        # write artifacts
        (reports_dir / "final_report.md").write_text(report)
        try:
            context.current_df.to_parquet(reports_dir / "final_dataset.parquet", index=False)
        except Exception:
            pass

        # persist fitted pipeline
        try:
            with (reports_dir / "fitted_pipeline.pkl").open("wb") as f:
                _pickle.dump(context.fitted_transformers, f)
        except Exception:
            pass

        try:
            (reports_dir / "pipeline.json").write_text(
                _json.dumps(
                    [{"operation": t.operation, "args": t.args} for t in context.fitted_transformers],
                    indent=2,
                )
            )
        except Exception:
            pass

        if context.current_test_df is not None:
            try:
                context.current_test_df.to_parquet(
                    reports_dir / "final_test_dataset.parquet", index=False
                )
            except Exception:
                pass

        return {"final_report": report}

    def submit_node(state: AgentState) -> dict[str, Any]:
        import pandas as _pd

        if context.current_test_df is None:
            return {}

        target = state["target"]
        X_train = context.current_df.drop(columns=[target], errors="ignore")
        y_train = context.current_df[target]
        X_test = context.current_test_df

        # id was extracted from test_df in runner.py but may still be in train
        if context.test_id_column_name and context.test_id_column_name in X_train.columns:
            X_train = X_train.drop(columns=[context.test_id_column_name])

        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            extra_in_test = test_cols - train_cols
            raise RuntimeError(
                f"Feature mismatch between train and test.\n"
                f"Missing in test: {missing_in_test}\nExtra in test: {extra_in_test}"
            )

        model_fitted = evaluator.fit_full(X_train, y_train)

        cat_features = evaluator._detect_cat_features(X_test)
        X_test_clean = evaluator._prepare_X(X_test, cat_features)

        if evaluator.task == "binary":
            preds = model_fitted.predict_proba(X_test_clean)[:, 1]
        elif evaluator.task == "multiclass":
            preds = model_fitted.predict(X_test_clean).ravel()
        else:
            preds = model_fitted.predict(X_test_clean)

        id_name = context.test_id_column_name or "id"
        if context.test_id_values is not None:
            id_vals = context.test_id_values.values
        else:
            id_vals = range(len(preds))

        sub = _pd.DataFrame({id_name: id_vals, target: preds})
        sub_path = reports_dir / "submission.csv"
        sub.to_csv(sub_path, index=False)

        return {"submission_path": str(sub_path)}

    def route_after_planner(state: AgentState) -> Literal["apply", "final_report"]:
        if state.get("decision") == "finish":
            return "final_report"
        return "apply"

    def route_after_reflect(state: AgentState) -> Literal["planner", "final_report"]:
        if state.get("decision") == "finish":
            return "final_report"
        return "planner"

    def route_after_final(state: AgentState) -> str:
        return "submit" if state.get("has_test_df") else "__end__"

    graph = StateGraph(AgentState)

    graph.add_node("profile", profile_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("planner", planner_node)
    graph.add_node("apply", apply_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("final_report", final_report_node)
    graph.add_node("submit", submit_node)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "analyze")
    graph.add_edge("analyze", "planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"apply": "apply", "final_report": "final_report"}
    )
    graph.add_edge("apply", "evaluate")
    graph.add_edge("evaluate", "reflect")
    graph.add_conditional_edges(
        "reflect", route_after_reflect, {"planner": "planner", "final_report": "final_report"}
    )
    graph.add_conditional_edges(
        "final_report",
        route_after_final,
        {"submit": "submit", "__end__": END},
    )
    graph.add_edge("submit", END)

    return graph.compile()
