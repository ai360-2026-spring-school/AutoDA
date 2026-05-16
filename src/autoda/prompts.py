import json
from typing import Any

PLANNER_PROMPT = """\
You are an iterative data-improvement agent. Your job is to propose ONE action per turn \
that is most likely to improve the cross-validation metric on the dataset.

Rules:
- Choose exactly one operation from the catalog below.
- The change will be applied to a working copy of the dataset. CatBoost CV will run and \
  the change is kept only if it improves the metric by more than {tolerance} (absolute). \
  Otherwise it is rolled back — so prefer safe, targeted changes.
- When you want to remember a hypothesis for future iterations, use `record_insight` \
  (it does NOT change the data, so it is always kept).
- If you believe no further improvement is possible, set "stop": true.

Output strict JSON — no markdown, no commentary:
{{
  "thought": "why this action should improve the metric",
  "operation": "<operation name from catalog>",
  "args": {{...}},
  "expected_effect": "brief description of expected metric change",
  "stop": false
}}

Action catalog:
{schema}

Example:
{{
  "thought": "Fare has a heavy right tail, log transform should reduce skew and help the model",
  "operation": "log_transform",
  "args": {{"columns": ["Fare"], "plus_one": true}},
  "expected_effect": "reduced skew on Fare, slight AUC gain",
  "stop": false
}}
"""

REFLECT_PROMPT = """\
You are reviewing the outcome of one data-improvement iteration.

Summarise what happened and extract any lasting insight worth keeping for future iterations.

Output strict JSON:
{{
  "conclusion": "one or two sentences describing what was found and whether the change helped",
  "insight": null
}}

OR if you have a hypothesis worth preserving for future iterations:
{{
  "conclusion": "...",
  "insight": {{
    "title": "short title",
    "body": "actionable observation for future steps",
    "evidence": {{"key": "value"}}
  }}
}}
"""

FINAL_REPORT_PROMPT = """\
Write a concise markdown report summarising the full data-improvement run.

Structure:
1. **Baseline** — initial CV score.
2. **Applied changes** — table of kept actions with metric delta.
3. **Rejected changes** — count and brief reason.
4. **Final CV score**.
5. **Key insights** — synthesise the insights ledger into 3-5 bullet points.

Use only the information provided. Do not invent facts.
"""


def build_planner_prompt(
    state: dict[str, Any],
    schema: list[dict],
    tolerance: float = 1e-4,
    last_k: int = 6,
) -> str:
    iterations = state.get("iterations", [])
    recent = iterations[-last_k:] if len(iterations) > last_k else iterations

    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    insights_str = json.dumps(state.get("insights", []), ensure_ascii=False)[:4000]
    recent_str = json.dumps(recent, ensure_ascii=False)[:8000]
    profile_str = json.dumps(state.get("dataset_profile", {}), ensure_ascii=False)[:6000]

    baseline_mean = state.get("baseline_cv_mean")
    baseline_std = state.get("baseline_cv_std")

    prompt = PLANNER_PROMPT.format(schema=schema_str, tolerance=tolerance)

    context = f"""
--- Context ---
Goal: {state.get("goal")}
Target: {state.get("target")}
Task: {state.get("task")}
Metric: {state.get("metric_name")} (direction: {state.get("metric_direction")}, higher-is-better={state.get("metric_direction") == "max"})
Tolerance: {tolerance}
Baseline CV: mean={baseline_mean}, std={baseline_std}
Current step: {state.get("current_step", 0)}
No-improve streak: {state.get("no_improve_streak", 0)}

Dataset profile:
{profile_str}

Insights ledger:
{insights_str}

Recent iterations (last {last_k}):
{recent_str}
--- End Context ---
"""
    return (prompt + context)[:16000]


def build_reflect_prompt(
    state: dict[str, Any],
    cv_before: float | None,
    cv_after: float | None,
    cv_delta: float | None,
    decision: str,
) -> str:
    action = json.dumps(state.get("proposed_action", {}), ensure_ascii=False)
    observation = json.dumps(state.get("last_observation", {}), ensure_ascii=False)[:4000]

    return f"""{REFLECT_PROMPT}

--- Iteration Data ---
Action: {action}
Observation: {observation}
CV before: {cv_before}
CV after: {cv_after}
CV delta (signed improvement): {cv_delta}
Decision: {decision}
Error: {state.get("last_error")}
--- End ---
"""


def build_final_report_prompt(state: dict[str, Any]) -> str:
    iterations_str = json.dumps(state.get("iterations", []), ensure_ascii=False)[:16000]
    insights_str = json.dumps(state.get("insights", []), ensure_ascii=False)[:4000]
    applied_str = json.dumps(state.get("applied_actions", []), ensure_ascii=False)[:6000]

    return f"""{FINAL_REPORT_PROMPT}

--- Run Summary ---
Goal: {state.get("goal")}
Target: {state.get("target")}
Metric: {state.get("metric_name")} ({state.get("metric_direction")})
Baseline CV: {state.get("baseline_cv_mean")}
Final CV: {state.get("baseline_cv_mean")}
Applied actions: {applied_str}
Insights: {insights_str}
All iterations: {iterations_str}
--- End ---
"""
