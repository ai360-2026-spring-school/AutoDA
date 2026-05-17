import json
from typing import Any

from .profile_summary import format_profile_summary
from .experiment_log import dedup_experiments, format_experiment_log

DESCRIPTION_TOKEN_BUDGET = 1200  # characters
PROMPT_HARD_CAP = 18000           # rough char cap on the assembled prompt
DESCRIPTION_CAP = 1500            # chars of user description re-injected each turn

SUMMARIZE_DESCRIPTION_PROMPT = """\
You are summarizing a user's free-text description of a tabular dataset.

Goal: produce a compact summary (max ~250 words) that preserves
- the business meaning of the dataset and the target,
- any domain-specific column semantics the user mentioned,
- any data-quality caveats or known issues,
- competition / scoring conventions if mentioned.

Drop fluff, examples that don't illustrate a rule, and anything not useful
for an automated cleaning / feature-engineering agent.

Output plain text. No JSON, no markdown headers.
"""


def summarize_description(model, text: str) -> str:
    response = model.invoke(f"{SUMMARIZE_DESCRIPTION_PROMPT}\n\n--- Description ---\n{text}\n--- End ---")
    return getattr(response, "content", str(response)).strip()

PLANNER_PROMPT = """\
You are an iterative data-improvement agent for a tabular ML competition.
Each turn you propose ONE action that should improve cross-validated metric on the training dataset.

How acceptance works:
- TRANSFORMER ops are applied to a copy of the dataset, then a fixed-config CatBoost is
  re-fit under 5-fold CV. The change is KEPT iff the metric improves by more than
  {tolerance} (absolute). Otherwise it is rolled back automatically.
- INFO ops are read-only probes (no CV, no data change). Their output is added to the
  insights ledger so you can use it next turn. Use them sparingly — they cost a turn.
- If you have nothing useful left to try, return "stop": true with a brief reason.

Read the action catalog carefully — every entry has a "use when" line that tells you
exactly when it's appropriate. Match your action to the signal in PROFILE SUMMARY and
to what you've already learned in INSIGHTS LEDGER.

Avoid waste:
- The EXPERIMENTS LOG shows what's already been tried. Do NOT repeat a (operation, args)
  combination that has already been KEPT or REJECTED — try a variation or pick a different op.
- Pay attention to ERRORS — same args will fail the same way.

Output STRICT JSON — no markdown, no commentary, no code fences:
{{
  "thought": "why this action should improve the metric, grounded in the profile / insights",
  "operation": "<operation name from catalog>",
  "args": {{...}},
  "expected_effect": "brief sentence on expected metric change",
  "stop": false
}}

ACTION CATALOG (the closed set of things you can do):
{catalog}
"""

REFLECT_PROMPT = """\
You just observed the outcome of one data-improvement iteration. Your job:
1. Write a one- or two-sentence "conclusion" — what happened and whether it helped.
2. OPTIONALLY add a lasting "insight" that future iterations should know. Examples:
   - "Fare is heavy-tailed; log_transform did not help on its own but maybe with binarize_missing".
   - "drop_columns(Cabin) was a net negative — Cabin's missingness pattern carries signal".
   - "target_encode_oof(Race) lifted AUC by 0.005 — good target for related encoders".
   - "baseline_linear_model CV ~ {{value}} — non-linear FE still has headroom".
Insights that just restate the action / delta are useless; only add an insight when there's
a genuine cross-iteration lesson.

Output STRICT JSON (no markdown / fences):
{{
  "conclusion": "...",
  "insight": null
}}
OR with an insight:
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

ANALYZE_PROMPT = """\
You are reviewing the initial profile of a tabular dataset BEFORE any cleaning or FE.
You will see (a) a compact structured PROFILE SUMMARY (already curated by deterministic
checks — DROP-candidates, high-skew, high-card cats, datetime cols, suspected leakage,
class balance, high-correlation pairs) and (b) the raw ydata profile beneath it.

Your job: produce 3-7 actionable, NON-OBVIOUS structured insights that will help an
iterative agent improve CV. Concretely:
- Do NOT just restate what's in the profile summary — the planner already sees that summary
  every turn. Insights should add interpretation or hypotheses.
- Suggest concrete next steps tied to specific columns (e.g. "Fare → log_transform then
  group_aggregate by Embarked").
- Call out suspected leakage / id-like columns by name.
- For classification with class imbalance, note whether the metric is rank-based (AUC) or
  threshold-based.
- If the dataset description mentions a known caveat (e.g. "Normalized_TyreLife removed
  to avoid trivial prediction"), tie that into your insights.

Output strict JSON: {"insights": [{"title": "...", "body": "...", "evidence": {...}}, ...]}
Return raw JSON only. Do NOT wrap in ```json fences.
"""


def _format_catalog(schema: Any) -> str:
    """Render the catalog. Accepts the rich list-of-dicts produced by
    ``actions.catalog.build_catalog`` (preferred) or a plain JSON-serialisable
    list (back-compat with the old hand-edited SCHEMA)."""
    if not schema:
        return "(no actions)"
    if isinstance(schema, list) and schema and isinstance(schema[0], dict) and "summary" in schema[0]:
        # rich catalog — render with format_catalog
        from .actions.catalog import format_catalog
        return format_catalog(schema)
    # fallback: dump JSON
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_planner_prompt(
    state: dict[str, Any],
    schema: list[dict],
    tolerance: float = 1e-4,
    last_k: int = 6,
) -> str:
    catalog_str = _format_catalog(schema)
    insights_str = json.dumps(state.get("insights", []), ensure_ascii=False)[:4000]

    # v4: compact profile summary instead of the raw JSON dump
    profile_summary = state.get("profile_summary") or {}
    profile_summary_str = format_profile_summary(profile_summary)

    # v4: deduped experiments tried so far — prevents repeats
    experiment_log = dedup_experiments(state.get("experiment_log", []), limit=40)
    experiment_log_str = format_experiment_log(experiment_log)

    baseline_mean = state.get("baseline_cv_mean")
    baseline_std = state.get("baseline_cv_std")
    metric_name = state.get("metric_name")
    direction = state.get("metric_direction")

    prompt = PLANNER_PROMPT.format(catalog=catalog_str, tolerance=tolerance)

    context = f"""
==================== CONTEXT ====================
Goal: {state.get("goal")}
Target column: {state.get("target")}
Task: {state.get("task")}  ({metric_name}, direction={direction}, higher-is-better={direction == "max"})
Acceptance tolerance: {tolerance}
Current step: {state.get("current_step", 0)}
Has test df: {state.get("has_test_df", False)}
Current baseline CV: mean={baseline_mean}, std={baseline_std}

DATASET DESCRIPTION (user-provided):
{(state.get("dataset_description") or "(none)")[:DESCRIPTION_CAP]}

PROFILE SUMMARY (key signal — refer to this for column-level decisions):
{profile_summary_str}

INSIGHTS LEDGER (built up across iterations — read it before deciding):
{insights_str}

EXPERIMENTS LOG (what's already been tried — do NOT repeat exact combinations):
{experiment_log_str}
================================================
"""
    return (prompt + context)[:PROMPT_HARD_CAP]


def build_reflect_prompt(
    state: dict[str, Any],
    cv_before: float | None,
    cv_after: float | None,
    cv_delta: float | None,
    decision: str,
) -> str:
    action = json.dumps(state.get("proposed_action", {}), ensure_ascii=False)
    observation = json.dumps(state.get("last_observation", {}), ensure_ascii=False)[:4000]

    # v4: reflect also gets the curated profile summary so insights can
    # reference actual dataset properties (e.g. "Fare is still right-skewed").
    profile_summary_str = format_profile_summary(state.get("profile_summary") or {})

    return f"""{REFLECT_PROMPT}

==================== ITERATION DATA ====================
Target: {state.get("target")}
Metric: {state.get("metric_name")} ({state.get("metric_direction")})

DATASET DESCRIPTION (user-provided):
{(state.get("dataset_description") or "(none)")[:DESCRIPTION_CAP]}

PROFILE SUMMARY (use to ground insights in actual dataset properties):
{profile_summary_str}

ACTION just attempted: {action}
OBSERVATION returned: {observation}
CV before:  {cv_before}
CV after:   {cv_after}
CV delta (signed improvement): {cv_delta}
Decision:   {decision}
Error:      {state.get("last_error")}
========================================================
"""


def build_analyze_prompt(state: dict[str, Any]) -> str:
    # analyze gets both the curated summary AND a slice of the raw profile,
    # since this is the one node that can really "see" the dataset before iteration starts.
    profile_summary_str = format_profile_summary(state.get("profile_summary") or {})
    raw_profile_str = json.dumps(state.get("dataset_profile", {}), ensure_ascii=False)[:6000]
    target = state.get("target", "")
    task = state.get("task", "")

    return f"""{ANALYZE_PROMPT}

Output strict JSON: {{"insights": [{{"title": "...", "body": "...", "evidence": {{}}}}, ...]}}
Return raw JSON only. Do NOT wrap in ```json fences.

==================== DATASET PROFILE ====================
Target: {target}
Task: {task}
Has test df: {state.get("has_test_df", False)}

DATASET DESCRIPTION (user-provided):
{(state.get("dataset_description") or "(none)")[:DESCRIPTION_CAP]}

PROFILE SUMMARY (already curated; build on it, do not just restate it):
{profile_summary_str}

RAW YDATA PROFILE (excerpt — for the detail the summary may have dropped):
{raw_profile_str}
=========================================================
"""


def build_final_report_prompt(state: dict[str, Any]) -> str:
    iterations_str = json.dumps(state.get("iterations", []), ensure_ascii=False)[:14000]
    insights_str = json.dumps(state.get("insights", []), ensure_ascii=False)[:4000]
    applied_str = json.dumps(state.get("applied_actions", []), ensure_ascii=False)[:6000]
    info_str = json.dumps(state.get("info_tool_results", []), ensure_ascii=False)[:3000]

    # Baseline = iteration 0's cv_after; final = the current running baseline_cv_mean.
    iterations = state.get("iterations", [])
    baseline = iterations[0].get("cv_after") if iterations else None
    final = state.get("baseline_cv_mean")

    return f"""{FINAL_REPORT_PROMPT}

==================== RUN SUMMARY ====================
Goal: {state.get("goal")}
Target: {state.get("target")}
Metric: {state.get("metric_name")} ({state.get("metric_direction")})
Baseline CV (iteration 0): {baseline}
Final CV (after all kept changes): {final}

DATASET DESCRIPTION (user-provided):
{(state.get("dataset_description") or "(none)")[:DESCRIPTION_CAP]}

Applied (kept) actions in order:
{applied_str}

Info-tool calls (read-only probes):
{info_str}

Insights ledger:
{insights_str}

All iterations (full log):
{iterations_str}
=====================================================
"""
