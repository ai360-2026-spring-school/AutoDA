"""
AutoDA example: Medical Insurance Costs.

1070 строк в train, 268 в test, 6 признаков, регрессия.
Бейзлайн CatBoost на сырых фичах: R² ≈ 0.84 (из-за smoker).
Ключевой FE: smoker × bmi — взаимодействие даёт +3-5% R², т.к. затраты
курящих людей с высоким BMI резко нелинейны. Также: age^2, log(charges) как таргет.
"""

import pandas as pd
from pathlib import Path
from autoda import PDAgent

TRAIN = Path("test_data/insurance/train.csv")
TEST  = Path("test_data/insurance/test.csv")

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
test_features = test.drop(columns=["charges"])

description = """
Medical Insurance Costs (Kaggle/GitHub). Регрессия: предсказать годовые медицинские
расходы (charges, USD) для физических лиц.

Признаки:
  age      — возраст (18–64)
  sex      — пол (male/female)
  bmi      — индекс массы тела (Body Mass Index)
  children — число детей на иждивении
  smoker   — курит ли человек (yes/no)
  region   — регион США (northeast/northwest/southeast/southwest)

Целевая переменная: charges — медицинские расходы в долларах.

Ключевые особенности:
  - smoker является мощнейшим предиктором: курильщики тратят в ~3-4x больше
  - Взаимодействие smoker × bmi особенно важно: курящие + ожирение = экстремальные расходы
  - charges сильно скошен вправо, log-трансформация таргета может помочь
  - age и bmi имеют нелинейную связь с charges (степенные зависимости)
"""

agent = PDAgent(
    provider="gigachat",
    max_iterations=15,
    tolerance=0.001,
    critic_every=3,
    debug=True,
)

res = agent.run(
    df=train,
    target="charges",
    goal="Максимизировать R² для предсказания медицинских расходов. "
         "Исследуй взаимодействия между smoker и bmi, степенные признаки по age и bmi.",
    test_df=test_features,
    description=description,
)

print(f"\n{'─'*50}")
print(f"Baseline:           {res.baseline_cv:.4f}")
print(f"После FE агента:    {res.final_cv:.4f}  ({res.final_cv - res.baseline_cv:+.4f})")
kept     = [e for e in res.experiment_log if e.get("decision") == "keep"]
rejected = [e for e in res.experiment_log if e.get("decision") == "reject"]
print(f"Принято / отклонено: {len(kept)} / {len(rejected)}")
if res.submission_path:
    print(f"Submission: {res.submission_path}")
print()
print(res.report)
