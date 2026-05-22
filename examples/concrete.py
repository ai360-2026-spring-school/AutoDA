"""
AutoDA example: Concrete Compressive Strength (UCI).

824 строки в train, 206 в test, 8 числовых признаков, регрессия.
Базовый CatBoost на сырых фичах: RMSE ≈ 6.5 МПа.
Ключевой FE: cement/water ratio (водоцементное отношение) — фундаментальная
формула строительной химии; log(age+1) — прочность растёт логарифмически со временем;
binder_total = cement + slag + fly_ash.
"""

import pandas as pd
from pathlib import Path
from autoda import PDAgent

TRAIN = Path("test_data/concrete/train.csv")
TEST  = Path("test_data/concrete/test.csv")

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
test_features = test.drop(columns=["compressive_strength"])

description = """
Concrete Compressive Strength (UCI). Регрессия: предсказать прочность бетона
на сжатие (compressive_strength, МПа) по составу смеси.

Компоненты смеси (кг/м³):
  cement             — цемент
  blast_furnace_slag — доменный шлак (замена части цемента)
  fly_ash            — зола-унос (замена части цемента)
  water              — вода
  superplasticizer   — суперпластификатор (добавка)
  coarse_aggregate   — крупный заполнитель (щебень)
  fine_aggregate     — мелкий заполнитель (песок)
  age                — возраст образца в днях (1–365)

Целевая переменная: compressive_strength — прочность на сжатие в МПа.

Строительно-химические закономерности:
  - cement/water (водоцементное отношение) — главный параметр прочности бетона
  - Суммарное вяжущее: cement + blast_furnace_slag + fly_ash
  - Прочность растёт логарифмически с возрастом: log(age+1)
  - Эффективность пластификатора зависит от водоцементного отношения
  - Доля заполнителей относительно всей смеси влияет на плотность
"""

agent = PDAgent(
    provider="gigachat",
    max_iterations=15,
    tolerance=0.05,
    critic_every=3,
    debug=True,
    metric="rmse",
)

res = agent.run(
    df=train,
    target="compressive_strength",
    goal="Минимизировать RMSE. Создавай отношения компонентов: cement/water, "
         "суммарное вяжущее, log(age). Думай как строительный химик.",
    test_df=test_features,
    description=description,
)

print(f"\n{'─'*50}")
print(f"Baseline RMSE:      {res.baseline_cv:.4f}")
print(f"После FE агента:    {res.final_cv:.4f}  ({res.final_cv - res.baseline_cv:+.4f})")
kept     = [e for e in res.experiment_log if e.get("decision") == "keep"]
rejected = [e for e in res.experiment_log if e.get("decision") == "reject"]
print(f"Принято / отклонено: {len(kept)} / {len(rejected)}")
if res.submission_path:
    print(f"Submission: {res.submission_path}")
print()
print(res.report)
