from __future__ import annotations

import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .dataset import DatasetContext
from .evaluator import CatBoostEvaluator, is_keep, CVResult
from .preprocessor import run_preprocess
from .column_typer import detect_column_types
from .actions.registry import TRANSFORMERS
from .actions.info import sparse_linear_features, baseline_linear_model
from .actions.info_new import (
    groupby_agg, value_counts, correlation_matrix,
    describe_column, view_precomputed_stats, view_long_summary,
)
from .prompts import (
    build_description_summarise_prompt,
    build_planner_prompt,
    build_critic_prompt,
    build_final_report_prompt,
    build_ideator_prompt,
    IDEATOR_ROLES,
)

_INFO_TOOLS_NEW = {
    "groupby_agg": groupby_agg,
    "value_counts": value_counts,
    "correlation_matrix": correlation_matrix,
    "describe_column": describe_column,
    "view_precomputed_stats": view_precomputed_stats,
    "view_long_summary": view_long_summary,
}
_INFO_TOOLS_OLD = {
    "sparse_linear_features": sparse_linear_features,
    "baseline_linear_model": baseline_linear_model,
}
_ALL_INFO_TOOLS = {**_INFO_TOOLS_NEW, **_INFO_TOOLS_OLD}


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()
    # Fast path: valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Handle "Extra data" — model returned JSON + trailing explanation text.
    # raw_decode stops at the end of the first complete JSON value.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Last resort: find the first {...} block in the text
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"_parse_error": f"Could not parse JSON from response", "_raw": text[:300]}


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
    tolerance: float = 1e-4,
    reports_dir: Path = Path("reports"),
    ohe_max_cardinality: int = 12,
    numeric_unique_threshold: int = 12,
    oversample: bool = True,
    max_inner_turns: int = 15,
    debug: bool = False,
    critic_every: int = 3,
    critic_model=None,
    n_ideators: int = 0,
    ideator_model=None,
):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_cell: list[CVResult | None] = [None]

    def _dbg(*parts: Any) -> None:
        if debug:
            print(" ".join(str(p) for p in parts), flush=True)

    def _args_brief(args: dict) -> str:
        s = json.dumps(args, ensure_ascii=False)
        return s[:70] + "..." if len(s) > 70 else s

    # -----------------------------------------------------------------------
    # Phase 1: Deterministic preprocessing
    # -----------------------------------------------------------------------

    def preprocess_node(state: AgentState) -> dict[str, Any]:
        target = state["target"]
        task = evaluator.task

        _dbg(f"[preprocess] {task}, {len(context.df)} rows x {len(context.df.columns)} cols — running...")

        df_out, transformers, col_type_map, corr_stats = run_preprocess(
            df=context.df,
            target=target,
            task=task,
            ohe_max_cardinality=ohe_max_cardinality,
            numeric_unique_threshold=numeric_unique_threshold,
            oversample=oversample,
        )

        context.current_df = df_out
        context.fitted_transformers = list(transformers)

        # Apply preprocessing to test_df (upsampling excluded — no transformer)
        if context.test_df is not None:
            test_out = context.test_df.copy()
            for t in transformers:
                try:
                    test_out = t.apply(test_out)
                except Exception:
                    pass
            context.current_test_df = test_out

        # Baseline CV on preprocessed data
        X = context.current_df.drop(columns=[target], errors="ignore")
        y = context.current_df[target]
        baseline = evaluator.cv(X, y, step=0)
        baseline_cell[0] = baseline

        feature_cols = [c for c in context.current_df.columns if c != target]
        initial_memory: list[str] = []

        _dbg(
            f"[preprocess] {task},"
            f" {len(context.df)}→{len(df_out)} rows,"
            f" {len(context.df.columns)}→{len(df_out.columns)} cols,"
            f" baseline {evaluator.metric_name}={baseline.mean:.4f} ± {baseline.std:.4f}"
        )

        return {
            "task": task,
            "metric_name": evaluator.metric_name,
            "metric_direction": evaluator.metric_direction,
            "baseline_cv_mean": baseline.mean,
            "baseline_cv_std": baseline.std,
            "has_test_df": context.test_df is not None,
            "column_type_map": col_type_map,
            "target_correlation_stats": corr_stats,
            "feature_columns": feature_cols,
            "iteration_start_cv": baseline.mean,
            "current_step": 0,
            "experiment_log": [],
            "applied_pipeline": [],
            "decision": "continue",
            "final_report": None,
            "submission_path": None,
            "planner_memory": initial_memory,
            "critic_message": None,
            "current_iteration_transforms": [],
            "planner_addendum": None,
            "planner_turn_count": 0,
            "long_description_summary": None,
            "short_description_summary": None,
        }

    # -----------------------------------------------------------------------
    # Phase 2: Description summarisation (LLM, once)
    # -----------------------------------------------------------------------

    def summarise_node(state: AgentState) -> dict[str, Any]:
        description = state.get("dataset_description")
        if not description:
            _dbg("[summarise] skipped (no description)")
            return {}

        prompt = build_description_summarise_prompt(description)
        try:
            response = model.invoke(prompt)
            content = getattr(response, "content", str(response))
            parsed = parse_json_response(content)
            if "_parse_error" in parsed:
                _dbg("[summarise] parse error, skipped")
                return {}
            long_summary = parsed.get("long_summary", "")[:400]
            short_summary = parsed.get("short_summary", "")[:300]
        except Exception:
            _dbg("[summarise] exception, skipped")
            return {}

        _dbg("[summarise] summaries generated")
        initial_memory = [short_summary] if short_summary else []
        return {
            "long_description_summary": long_summary,
            "short_description_summary": short_summary,
            "planner_memory": initial_memory,
        }

    # -----------------------------------------------------------------------
    # Phase 2b: Ideator node (runs once per iteration before planner turn 1)
    # -----------------------------------------------------------------------

    _ideator_model = ideator_model if ideator_model is not None else model
    _ideator_roles = IDEATOR_ROLES[:n_ideators] if n_ideators > 0 else []

    def _call_one_ideator(role: dict[str, str], state: AgentState) -> dict[str, Any] | None:
        """Call model with one ideator role. Returns suggestion dict or None on failure."""
        try:
            prompt = build_ideator_prompt(state, role["instruction"])
            response = _ideator_model.invoke(prompt)
            content = getattr(response, "content", None) or ""
            parsed = parse_json_response(content)
            if "_parse_error" in parsed:
                return None
            suggestion = parsed.get("suggestion", "").strip()
            if not suggestion:
                return None
            return {
                "role": role["name"],
                "suggestion": suggestion,
                "rationale": parsed.get("rationale", "")[:200],
                "priority": parsed.get("priority", "medium"),
            }
        except Exception:
            return None

    def ideate_node(state: AgentState) -> dict[str, Any]:
        if not _ideator_roles:
            return {"ideator_suggestions": []}

        suggestions: list[dict[str, Any]] = []

        # Call all ideators in parallel
        with ThreadPoolExecutor(max_workers=len(_ideator_roles)) as pool:
            futures = {
                pool.submit(_call_one_ideator, role, state): role["name"]
                for role in _ideator_roles
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    suggestions.append(result)

        # Sort by priority: high → medium → low
        _priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: _priority_order.get(s.get("priority", "medium"), 1))

        step = state.get("current_step", 0) + 1
        if suggestions:
            for s in suggestions:
                _dbg(f"[ideate/{s['role']}:{s['priority']}] {s['suggestion'][:80]}")
        else:
            _dbg(f"[ideate] step {step}: no suggestions (all failed or empty)")

        return {"ideator_suggestions": suggestions}

    # -----------------------------------------------------------------------
    # Phase 3: Planner node (self-loop)
    # -----------------------------------------------------------------------

    def _call_info_tool(op: str, args: dict, state: AgentState, df: pd.DataFrame) -> dict:
        target = state["target"]
        injected: dict[str, Any] = {}
        if op in ("sparse_linear_features", "baseline_linear_model"):
            injected["task"] = state.get("task", "regression")
        elif op == "view_precomputed_stats":
            injected["target_correlation_stats"] = state.get("target_correlation_stats", {})
        elif op == "view_long_summary":
            injected["long_description_summary"] = state.get("long_description_summary")

        combined = {**injected, **args}
        try:
            if op in _ALL_INFO_TOOLS:
                return _ALL_INFO_TOOLS[op](df, target, **combined)
            return {"error": f"Unknown info tool: {op!r}"}
        except Exception as e:
            return {"error": repr(e)}

    def _apply_transformer(op: str, args: dict, state: AgentState) -> tuple[Any, Any, Any, str | None]:
        """Apply a transformer op. Returns (df_out, test_df_out, transformer, error)."""
        target = state["target"]

        if op not in TRANSFORMERS:
            return None, None, None, f"Unknown transformer: {op!r}"

        # Inject feature importances for drop_low_importance
        call_args = dict(args)
        if op == "drop_low_importance":
            call_args.setdefault("feature_importances", evaluator.last_feature_importances_)

        working_df = context.working_df
        if working_df is None:
            working_df = context.current_df.copy()

        try:
            df_out, transformer, observation = TRANSFORMERS[op](working_df, target, **call_args)
        except Exception as e:
            return None, None, None, repr(e)

        # Apply to test_df
        test_df_out = None
        if context.working_test_df is not None or context.current_test_df is not None:
            test_src = context.working_test_df if context.working_test_df is not None else context.current_test_df
            try:
                test_df_out = transformer.apply(test_src)
            except Exception as te:
                # test apply failed — observation carries warning but transform still applies on train
                observation = dict(observation)
                observation["test_apply_warning"] = repr(te)

        return df_out, test_df_out, transformer, None

    def planner_node(state: AgentState) -> dict[str, Any]:
        # Lazily initialize working_df at the start of each iteration
        if context.working_df is None:
            context.working_df = context.current_df.copy()
            context.working_transformers_this_iteration = []
            if context.current_test_df is not None:
                context.working_test_df = context.current_test_df.copy()

        turn_count = state.get("planner_turn_count", 0)
        _step = state.get("current_step", 0) + 1
        _tag = f"[step {_step} / turn {turn_count + 1}]"

        # Force submit if inner turn limit reached
        if turn_count >= max_inner_turns:
            _dbg(f"{_tag} submit (forced — turn limit {max_inner_turns} reached)")
            return {
                "planner_addendum": {
                    "submit": True,
                    "rationale": f"принудительная отправка (лимит {max_inner_turns} ходов)",
                },
            }

        # Build prompt and call LLM (retry up to 2 times on network errors)
        prompt = build_planner_prompt(state)
        text = None
        llm_error: str | None = None
        for _attempt in range(3):
            try:
                response = model.invoke(prompt)
                text = getattr(response, "content", None) or ""
                llm_error = None
                break
            except Exception as e:
                llm_error = repr(e)
                _dbg(f"{_tag} LLM attempt {_attempt + 1}/3 failed: {e}")

        if llm_error is not None:
            _dbg(f"{_tag} LLM call failed after retries")
            return {
                "planner_addendum": {"parse_error": f"LLM call failed: {llm_error}"},
                "planner_turn_count": turn_count + 1,
            }

        if not text:
                # Empty response — usually means the prompt was too long for the model.
                _dbg(f"{_tag} empty LLM response (prompt too long?) — clearing addendum")
                return {
                    "planner_addendum": {
                        "parse_error": "Пустой ответ от LLM. Промпт мог быть слишком длинным. "
                                       "Попробуй более короткое действие.",
                    },
                    "planner_turn_count": turn_count + 1,
                }

        action = parse_json_response(text)
        if "_parse_error" in action:
            _dbg(f"{_tag} parse_error: {action['_parse_error'][:80]}")
            return {
                "planner_addendum": {"parse_error": action["_parse_error"]},
                "planner_turn_count": turn_count + 1,
            }

        action_type = action.get("type", "")

        # --- stop ---
        if action_type == "stop":
            _dbg(f"{_tag} stop: {action.get('reason', '')!r}")
            return {
                "planner_addendum": {"stop": True},
                "decision": "finish",
            }

        # --- submit ---
        if action_type == "submit":
            rationale = action.get("rationale", "")
            _dbg(f"{_tag} submit: {rationale!r}")
            return {
                "planner_addendum": {
                    "submit": True,
                    "rationale": rationale,
                },
            }

        # --- read_info ---
        if action_type == "read_info":
            op = action.get("op", "")
            args = action.get("args", {})
            result = _call_info_tool(op, args, state, context.working_df)
            _dbg(f"{_tag} read_info: {op}({_args_brief(args)})")
            return {
                "planner_addendum": {"info_result": result, "info_op": op},
                "planner_turn_count": turn_count + 1,
            }

        # --- transform ---
        if action_type == "transform":
            op = action.get("op", "")
            args = action.get("args", {})
            df_out, test_df_out, transformer, error = _apply_transformer(op, args, state)

            if error:
                _dbg(f"{_tag} transform ERROR: {op}({_args_brief(args)}) → {error[:60]}")
                return {
                    "planner_addendum": {
                        "transform_error": {"op": op, "args": args, "error": error},
                    },
                    "planner_turn_count": turn_count + 1,
                }

            # Commit to working context
            context.working_df = df_out
            if test_df_out is not None:
                context.working_test_df = test_df_out
            context.working_transformers_this_iteration.append(transformer)

            # Re-detect column types for new/changed columns
            new_col_map = detect_column_types(df_out, state["target"],
                                               unique_threshold=numeric_unique_threshold)
            changed_cols: list[str] = []
            if hasattr(transformer, "args"):
                changed_cols = [c for c in df_out.columns if c not in context.current_df.columns]
                if not changed_cols and op in ("multi_col_lambda",):
                    result_col = args.get("result_column", "")
                    if result_col:
                        changed_cols = [result_col]

            new_types = {c: new_col_map.get(c, "NUMERIC") for c in changed_cols}

            new_iter_transforms = list(state.get("current_iteration_transforms", [])) + [
                {"op": op, "args": args}
            ]

            _dbg(f"{_tag} transform: {op}({_args_brief(args)})")
            return {
                "planner_addendum": {
                    "transform_applied": {
                        "op": op,
                        "args": args,
                        "changed_columns": changed_cols,
                        "new_types": new_types,
                    },
                    "all_transforms": new_iter_transforms,
                },
                "current_iteration_transforms": new_iter_transforms,
                "column_type_map": new_col_map,
                "planner_turn_count": turn_count + 1,
            }

        # --- update_memory ---
        if action_type == "update_memory":
            notes = action.get("notes", [])
            if isinstance(notes, str):
                notes = [notes]
            new_memory = list(state.get("planner_memory", [])) + [str(n) for n in notes]
            total = sum(len(n) for n in new_memory)
            while total > 2000 and len(new_memory) > 1:
                removed = new_memory.pop(0)
                total -= len(removed)
            _dbg(f"{_tag} update_memory: {notes[:2]}")
            return {
                "planner_memory": new_memory,
                "planner_addendum": {
                    "memory_updated": True,
                    "all_transforms": state.get("current_iteration_transforms", []),
                },
                "planner_turn_count": turn_count + 1,
            }

        # --- cancel ---
        if action_type == "cancel":
            n_rolled = len(state.get("current_iteration_transforms", []))
            context.working_df = context.current_df.copy()
            if context.current_test_df is not None:
                context.working_test_df = context.current_test_df.copy()
            context.working_transformers_this_iteration = []
            restored_col_map = detect_column_types(
                context.current_df, state["target"], unique_threshold=numeric_unique_threshold
            )
            _dbg(f"{_tag} cancel ({n_rolled} transforms rolled back)")
            return {
                "planner_addendum": {"cancelled": True, "rolled_back_n": n_rolled},
                "current_iteration_transforms": [],
                "column_type_map": restored_col_map,
                "planner_turn_count": turn_count + 1,
            }

        # Unknown action type
        _dbg(f"{_tag} parse_error: unknown action type {action_type!r}")
        return {
            "planner_addendum": {"parse_error": f"Unknown action type: {action_type!r}"},
            "planner_turn_count": turn_count + 1,
        }

    # -----------------------------------------------------------------------
    # Evaluate node
    # -----------------------------------------------------------------------

    def evaluate_node(state: AgentState) -> dict[str, Any]:
        target = state["target"]
        step = state.get("current_step", 0) + 1
        base = baseline_cell[0]

        working = context.working_df if context.working_df is not None else context.current_df
        X = working.drop(columns=[target], errors="ignore")
        y = working[target]

        try:
            result = evaluator.cv(X, y, step=step)
        except Exception as e:
            # CV failed — reject and log error
            context.working_df = None
            context.working_test_df = None
            context.working_transformers_this_iteration = []
            return {
                "current_step": step,
                "current_iteration_transforms": [],
                "planner_turn_count": 0,
                "planner_addendum": {
                    "new_iteration": True,
                    "step": step + 1,
                    "cv": base.mean if base else 0.0,
                    "decision": "error",
                    "delta": 0.0,
                },
                "experiment_log": [{
                    "step": step,
                    "transforms": state.get("current_iteration_transforms", []),
                    "rationale": (state.get("planner_addendum") or {}).get("rationale", ""),
                    "cv_before": base.mean if base else None,
                    "cv_after": None,
                    "delta": 0.0,
                    "decision": "error",
                    "error": repr(e),
                }],
            }

        rationale = (state.get("planner_addendum") or {}).get("rationale", "")
        iter_transforms = state.get("current_iteration_transforms", [])
        cv_before = base.mean if base else 0.0
        cv_after = result.mean
        kept = is_keep(result, base, tol=tolerance)
        delta = _signed_delta(result, base) if base else 0.0
        decision = "keep" if kept else "reject"

        log_entry = {
            "step": step,
            "transforms": iter_transforms,
            "rationale": rationale,
            "cv_before": cv_before,
            "cv_after": cv_after,
            "delta": delta,
            "decision": decision,
        }

        updates: dict[str, Any] = {
            "current_step": step,
            "current_iteration_transforms": [],
            "planner_turn_count": 0,
            "experiment_log": [log_entry],
            "planner_addendum": {
                "new_iteration": True,
                "step": step + 1,
                "cv": cv_after,
                "decision": decision,
                "delta": delta,
            },
            "iteration_start_cv": cv_after,
        }

        if kept:
            context.current_df = context.working_df
            if context.working_test_df is not None:
                context.current_test_df = context.working_test_df
            context.fitted_transformers.extend(context.working_transformers_this_iteration)
            baseline_cell[0] = result

            updates["baseline_cv_mean"] = result.mean
            updates["baseline_cv_std"] = result.std
            updates["applied_pipeline"] = [
                {"step": step, "operation": t.operation, "args": t.args}
                for t in context.working_transformers_this_iteration
            ]
            # Refresh column type map on the new current_df
            updates["column_type_map"] = detect_column_types(
                context.current_df, state["target"], unique_threshold=numeric_unique_threshold
            )

        # Reset working context for next iteration
        context.working_df = None
        context.working_test_df = None
        context.working_transformers_this_iteration = []

        # Check stop condition
        if step >= max_iterations:
            updates["decision"] = "finish"

        return updates

    # -----------------------------------------------------------------------
    # Critic node
    # -----------------------------------------------------------------------

    def critic_node(state: AgentState) -> dict[str, Any]:
        step = state.get("current_step", 0)
        exp_log = state.get("experiment_log", [])

        if step % critic_every != 0:
            _dbg(f"[critic] skipped (step {step}, runs every {critic_every})")
            return {}  # preserve last critic_message

        if len(exp_log) < 2:
            _dbg("[critic] skipped (< 2 experiments)")
            return {"critic_message": None}

        prompt = build_critic_prompt(state)
        _critic = critic_model if critic_model is not None else model
        try:
            response = _critic.invoke(prompt)
            content = getattr(response, "content", str(response))
            parsed = parse_json_response(content)
            if "_parse_error" in parsed:
                _dbg("[critic] parse error")
                return {"critic_message": None}
            message = parsed.get("message")
            if message:
                _dbg(f"[critic] {message[:120]!r}")
            else:
                _dbg("[critic] silent")
            return {"critic_message": message if message else None}
        except Exception:
            _dbg("[critic] exception")
            return {"critic_message": None}

    # -----------------------------------------------------------------------
    # Final report node
    # -----------------------------------------------------------------------

    def final_report_node(state: AgentState) -> dict[str, Any]:
        base_cv = state.get("baseline_cv_mean") or 0.0
        # Find final CV from last kept entry in experiment_log
        exp_log = state.get("experiment_log", [])
        kept_entries = [e for e in exp_log if e.get("decision") == "keep"]
        final_cv = kept_entries[-1]["cv_after"] if kept_entries else base_cv

        prompt = build_final_report_prompt(state, baseline_cv=base_cv, final_cv=final_cv)
        try:
            response = model.invoke(prompt)
            report = getattr(response, "content", str(response))
        except Exception as e:
            report = f"[Report generation failed: {e}]"

        # Write artifacts
        try:
            (reports_dir / "final_report.md").write_text(report)
        except Exception:
            pass

        try:
            context.current_df.to_parquet(reports_dir / "final_dataset.parquet", index=False)
        except Exception:
            pass

        try:
            with (reports_dir / "fitted_pipeline.pkl").open("wb") as f:
                pickle.dump(context.fitted_transformers, f)
        except Exception:
            pass

        try:
            (reports_dir / "pipeline.json").write_text(json.dumps(
                [{"operation": t.operation, "args": t.args} for t in context.fitted_transformers],
                indent=2,
            ))
        except Exception:
            pass

        if context.current_test_df is not None:
            try:
                context.current_test_df.to_parquet(
                    reports_dir / "final_test_dataset.parquet", index=False
                )
            except Exception:
                pass

        _dbg(f"[final_report] saved to {reports_dir / 'final_report.md'}")
        return {"final_report": report}

    # -----------------------------------------------------------------------
    # Submit node
    # -----------------------------------------------------------------------

    def submit_node(state: AgentState) -> dict[str, Any]:
        if context.current_test_df is None:
            return {}

        target = state["target"]
        X_train = context.current_df.drop(columns=[target], errors="ignore")
        y_train = context.current_df[target]
        X_test = context.current_test_df

        if context.test_id_column_name and context.test_id_column_name in X_train.columns:
            X_train = X_train.drop(columns=[context.test_id_column_name])

        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            extra_in_test = test_cols - train_cols
            raise RuntimeError(
                f"Feature mismatch — missing in test: {missing_in_test}; extra in test: {extra_in_test}"
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
        id_vals = context.test_id_values.values if context.test_id_values is not None else range(len(preds))

        sub = pd.DataFrame({id_name: id_vals, target: preds})
        sub_path = reports_dir / "submission.csv"
        sub.to_csv(sub_path, index=False)

        _dbg(f"[submit] submission saved to {sub_path}")
        return {"submission_path": str(sub_path)}

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    def route_after_planner(state: AgentState) -> Literal["planner", "evaluate", "final_report"]:
        addendum = state.get("planner_addendum") or {}
        if addendum.get("stop") or state.get("decision") == "finish":
            return "final_report"
        if addendum.get("submit"):
            return "evaluate"
        return "planner"

    def route_after_critic(state: AgentState) -> Literal["ideate", "planner", "final_report"]:
        if state.get("decision") == "finish":
            return "final_report"
        if _ideator_roles:
            return "ideate"
        return "planner"

    def route_after_final(state: AgentState) -> str:
        return "submit" if state.get("has_test_df") else "__end__"

    # -----------------------------------------------------------------------
    # Build graph
    # -----------------------------------------------------------------------

    graph = StateGraph(AgentState)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("ideate", ideate_node)
    graph.add_node("planner", planner_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("final_report", final_report_node)
    graph.add_node("submit", submit_node)

    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "summarise")
    # After summarise: ideate first if roles are configured, else straight to planner
    if _ideator_roles:
        graph.add_edge("summarise", "ideate")
        graph.add_edge("ideate", "planner")
    else:
        graph.add_edge("summarise", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"planner": "planner", "evaluate": "evaluate", "final_report": "final_report"},
    )
    graph.add_edge("evaluate", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"ideate": "ideate", "planner": "planner", "final_report": "final_report"},
    )
    graph.add_conditional_edges(
        "final_report",
        route_after_final,
        {"submit": "submit", "__end__": END},
    )
    graph.add_edge("submit", END)

    return graph.compile()
