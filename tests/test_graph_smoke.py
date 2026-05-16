"""Smoke test: 3-iteration run with a scripted fake LLM (no real API call)."""
import json
import types
from pathlib import Path
import pandas as pd
import pytest

from autoda.dataset import DatasetContext
from autoda.evaluator import CatBoostEvaluator
from autoda.graph import build_graph


class ScriptedModel:
    """Returns canned planner/reflect JSON in a round-robin sequence."""

    PLANNER_RESPONSES = [
        # step 1: keep — impute missing (should be safe on Titanic)
        json.dumps({
            "thought": "Age has missing values; impute with median",
            "operation": "impute_missing",
            "args": {"columns": ["Age"], "strategy": "median"},
            "expected_effect": "removes NaNs, slight AUC improvement",
            "stop": False,
        }),
        # step 2: reject — log transform on a column with zeros (will error or reject)
        json.dumps({
            "thought": "Fare might benefit from log transform",
            "operation": "log_transform",
            "args": {"columns": ["Fare"], "plus_one": True},
            "expected_effect": "reduces skew",
            "stop": False,
        }),
        # step 3: finish
        json.dumps({
            "thought": "No more obvious improvements",
            "operation": "record_insight",
            "args": {"title": "done", "body": "No further improvements expected", "evidence": {}},
            "expected_effect": "none",
            "stop": True,
        }),
    ]

    REFLECT_RESPONSE = json.dumps({
        "conclusion": "Action completed.",
        "insight": None,
    })

    REPORT_RESPONSE = "# Final Report\n\nSome findings."

    def __init__(self):
        self._plan_calls = 0

    def invoke(self, prompt: str):
        if "Action catalog" in prompt or '"operation"' not in prompt[:200]:
            # planner prompt
            idx = min(self._plan_calls, len(self.PLANNER_RESPONSES) - 1)
            self._plan_calls += 1
            content = self.PLANNER_RESPONSES[idx]
        elif "conclusion" in prompt:
            content = self.REFLECT_RESPONSE
        else:
            content = self.REPORT_RESPONSE

        return types.SimpleNamespace(content=content)


@pytest.fixture
def titanic_df():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        return pd.read_csv(url)
    except Exception:
        pytest.skip("Cannot reach titanic CSV (no network)")


def test_smoke(titanic_df, tmp_path):
    target = "Survived"
    df = titanic_df

    context = DatasetContext(dataset_id="smoke", df=df, target=target)

    # use iterations=100 to keep the baseline CV fast
    evaluator = CatBoostEvaluator.auto(df[target], n_splits=3)
    # monkey-patch COMMON_PARAMS for speed
    import autoda.evaluator as ev_mod
    original_common = ev_mod.COMMON_PARAMS.copy()
    ev_mod.COMMON_PARAMS = {**original_common, "iterations": 100, "early_stopping_rounds": 10}

    model = ScriptedModel()

    app = build_graph(
        model=model,
        context=context,
        evaluator=evaluator,
        max_iterations=3,
        patience=4,
        tolerance=1e-4,
        reports_dir=tmp_path / "reports",
    )

    initial_state = {
        "goal": "smoke test",
        "target": target,
        "task": evaluator.task,
        "metric_name": evaluator.metric_name,
        "metric_direction": evaluator.metric_direction,
        "dataset_id": "smoke",
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

    final_state = None
    for state in app.stream(initial_state, stream_mode="values"):
        final_state = state

    # restore
    ev_mod.COMMON_PARAMS = original_common

    assert final_state is not None
    assert final_state.get("final_report"), "expected a final report"

    iterations = final_state.get("iterations", [])
    assert len(iterations) >= 2, f"expected at least 2 iterations, got {len(iterations)}"

    cv_history = tmp_path / "reports" / "cv_history.jsonl"
    assert cv_history.exists(), "cv_history.jsonl not created"
    lines = cv_history.read_text().strip().splitlines()
    assert len(lines) >= 2, f"expected at least 2 CV records, got {len(lines)}"
