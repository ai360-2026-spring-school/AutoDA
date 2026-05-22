"""
AutoDA example: BeamSearchAgent on Balance Scale.

Demonstrates tree/beam search over FE space:
  - Planner generates N ideas per node (no implementation)
  - N parallel implementers each apply one idea (clean context, no failure log)
  - Prune: keep top beam_width children by CV
  - Repeat for max_depth levels

Best result so far: mlogloss 0.352 → 0.110  (-69%)
with beam_width=5, n_ideas=10, max_depth=3
"""

import pandas as pd
from pathlib import Path
from autoda import BeamSearchAgent
from autoda.beam_viz import save_all
from autoda.evaluator import CatBoostEvaluator
from autoda.models.factory import make_model
from autoda.preprocessor import run_preprocess

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

# Preprocess once (OHE, impute, oversample)
df_prep, _, _, _ = run_preprocess(
    train, target="class", task="multiclass", oversample=True
)
test_prep = test.drop(columns=["class"], errors="ignore").copy()

evaluator = CatBoostEvaluator.auto(train["class"], n_splits=5)
model = make_model(provider="gigachat")

agent = BeamSearchAgent(
    planner_model=model,
    implementer_model=model,
    evaluator=evaluator,
    beam_width=5,    # keep top-5 branches per level
    n_ideas=10,      # rule of thumb: n_ideas = beam_width * 2
    max_depth=3,     # levels of chained transforms
    max_impl_turns=5,
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
    reports_dir="reports/beam_balance",
)

# Print ASCII tree + save PNG to reports/beam_balance/search_tree.png
save_all(result)

print(f"\n{'─'*50}")
print(f"Baseline: {result.root_cv:.4f}")
print(f"Best:     {result.best_cv:.4f}  (-{result.improvement_pct:.1f}%)")
print(f"Best path: {result.best_node.branch_summary()}")
