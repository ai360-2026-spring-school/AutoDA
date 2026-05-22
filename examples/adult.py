"""
AutoDA example: Adult Census Income dataset (UCI / OpenML).

~39K строк, 13 признаков (числовые + категориальные), бинарная классификация.
Шумность: пропуски в workclass / occupation / native-country, сильно скошенные
capital-gain и capital-loss (>90% нулей), высококардинальный native-country.

Данные уже лежат в test_data/adult/. Для первого скачивания:
    python -c "from sklearn.datasets import fetch_openml; fetch_openml('adult', version=2)"
"""

import pandas as pd
from pathlib import Path
from autoda import PDAgent

TRAIN = Path("test_data/adult/train_mini.csv")   # 10% для быстрого теста
TEST  = Path("test_data/adult/test.csv")

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)

test_features = test.drop(columns=["income"])

description = """
Adult Census Income (UCI). Предсказать, зарабатывает ли человек >50K в год (income=1).
Числовые: age, education-num, capital-gain, capital-loss, hours-per-week.
Категориальные: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
Особенности: capital-gain и capital-loss сильно перекошены (>90% нулей, редкие большие выбросы).
Пропуски в workclass, occupation, native-country (~6%).
"""

agent = PDAgent(
    provider="gigachat",
    max_iterations=10,
    tolerance=-0.001,
    critic_every=3,
    debug=True,
)
res = agent.run(
    df=train,
    target="income",
    goal="Максимизировать ROC AUC для предсказания дохода >50K",
    test_df=test_features,
    description=description,
    ohe_max_cardinality=12,
    oversample=True,
)

print(f"\nbaseline {res.baseline_cv:.4f} -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"experiment log: {len(res.experiment_log)} iterations")

if res.submission_path:
    print(f"submission: {res.submission_path}")
    if res.submission_df is not None:
        print(res.submission_df.head())

print("\n--- Report ---")
print(res.report)
