# AutoDA

Iterative LangGraph agent that improves a pandas dataset for ML — cleaning, feature engineering, encoding, and selection — guided by a fixed-config CatBoost cross-validation loop.

Each iteration the LLM proposes one change from a closed action catalog. The change is applied to a working copy of the dataset, CatBoost CV runs, and the change is **kept only if the metric improves** above a tolerance threshold. Otherwise it is rolled back. The agent stops when it hits a budget (max iterations, no-improvement streak, or LLM-issued stop).

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or with dev tools:
pip install -e ".[dev]"
```

## Environment variables

Copy `.env.example` to `.env` and fill in your Timeweb credentials:

```
TIMEWEB_API_TOKEN=...
TIMEWEB_AGENT_ID=...        # or set TIMEWEB_BASE_URL directly
TIMEWEB_MODEL=timeweb-agent # optional, defaults to "timeweb-agent"
```

## Quick start

```python
import pandas as pd
from autoda import PDAgent

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

agent = PDAgent(
    provider="timeweb",
    max_iterations=12,
    patience=4,
)

result = agent.run(
    df=df,
    goal="Improve CV for survival prediction",
    target="Survived",
)

print(f"baseline {result.baseline_cv:.4f} -> final {result.final_cv:.4f}")
print(result.report)
```

Per-step output:
```
[step 1] keep    impute_missing(columns=['Age'], strategy=median)
         roc_auc: 0.8321 -> 0.8389  (+0.0068)
[step 2] reject  log_transform(columns=['Fare'])
         roc_auc: 0.8389 -> 0.8381  (-0.0008)
[step 3] insight "Fare has a heavy right tail; binning may help"
```

## Reports

Artifacts are written to `reports/` (gitignored):

| File | Contents |
|---|---|
| `reports/profile_initial.html` | ydata-profiling HTML for the original dataset |
| `reports/profile_step_N.html` | Profile after each kept change |
| `reports/cv_history.jsonl` | One JSON line per CV run (step, mean, std, fold scores) |
| `reports/final_report.md` | Markdown summary: baseline → final score, kept/rejected actions, insights |
| `reports/final_dataset.parquet` | The improved dataset as a Parquet file |

## `AgentResult` fields

| Field | Type | Description |
|---|---|---|
| `report` | `str` | Markdown final report |
| `iterations` | `list[Iteration]` | Full per-step log |
| `applied_actions` | `list[dict]` | Kept actions only |
| `insights` | `list[dict]` | LLM-recorded hypotheses |
| `baseline_cv` | `float` | Initial CV score |
| `final_cv` | `float` | Best achieved CV score |
| `final_df` | `pd.DataFrame` | Dataset after kept changes |
| `raw_state` | `dict` | Full LangGraph state |

## Design docs

- [`docs/PLAN.md`](docs/PLAN.md) — architecture and design decisions
- [`docs/TASKS.md`](docs/TASKS.md) — implementation task list
