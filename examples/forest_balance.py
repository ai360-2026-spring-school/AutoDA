"""
AutoDA: ForestBeamSearch на Balance Scale — 5 деревьев последовательно.
n_ideas=12, beam_width=6, max_depth=9
"""

import pandas as pd
from pathlib import Path
from autoda import BeamSearchAgent
from autoda.beam_viz import save_all
from autoda.evaluator import CatBoostEvaluator
from autoda.models.factory import make_model
from autoda.preprocessor import run_preprocess
from autoda.beam import set_llm_concurrency

TRAIN = Path("test_data/balance_scale/train.csv")
TEST  = Path("test_data/balance_scale/test.csv")

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)

description = """
Balance Scale. Мультиклассификация: class (L/B/R).
Признаки: left_weight, left_distance, right_weight, right_distance.
Физика: момент силы = weight × distance.
Класс R если right_moment > left_moment, L если left > right, B если равны.
"""

df_prep, _, _, _ = run_preprocess(
    train, target="class", task="multiclass", oversample=True
)
test_prep = test.drop(columns=["class"], errors="ignore").copy()

evaluator = CatBoostEvaluator.auto(train["class"], n_splits=5)
model = make_model(provider="gigachat")

# Один запрос к LLM одновременно — снижаем 429
set_llm_concurrency(1)

agent = BeamSearchAgent(
    planner_model=model,
    implementer_model=model,
    evaluator=evaluator,
    beam_width=6,
    n_ideas=12,
    max_depth=9,
    max_impl_turns=5,
    n_trees=5,
    sequential=True,
    tolerance=0.001,
    debug=True,
)

result = agent.run(
    df=df_prep,
    target="class",
    goal="Минимизировать mlogloss. Ключ: момент силы = weight×distance.",
    task="multiclass",
    metric_name="mlogloss",
    metric_direction="min",
    test_df=test_prep,
    description=description,
    reports_dir="reports/forest_balance",
)

save_all(result)

print(f"\n{'─'*50}")
print(f"Baseline: {result.root_cv:.4f}")
print(f"Best:     {result.best_cv:.4f}  (-{result.improvement_pct:.1f}%)")
print(f"Best path: {result.best_node.branch_summary()}")
