"""
AutoDA example: Red Wine Quality (UCI).

1279 строк, 11 числовых признаков, бинарная классификация (quality >= 6 → 1).
Нет пропусков, нет категориальных — предобработка минимальна, весь прирост
должен идти от feature engineering.

Ключевые FE-возможности для LLM:
  - free_sulfur_dioxide / total_sulfur_dioxide (связанная доля SO2)
  - volatile_acidity / fixed_acidity          (доля летучей кислотности)
  - alcohol / density                          (объёмная «концентрация»)
  - fixed_acidity / pH                         (интенсивность кислоты)
  - sulphates * alcohol                        (антисептик × крепость)

Базовые baseline (без FE):
  CatBoost:      ≈ 0.868 roc_auc  (дерево само строит взаимодействия)
  LogisticReg:   ≈ 0.798 roc_auc  (линейная, нужны явные признаки)

Разрыв ~0.07 — это потолок, который FE на логрег должен закрыть.
"""

import pandas as pd
from pathlib import Path
from autoda import PDAgent

TRAIN = Path("test_data/wine/train.csv")
TEST  = Path("test_data/wine/test.csv")

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
test_features = test.drop(columns=["quality"])

BASE_MODEL = "catboost"
print(f"=== AutoDA + {BASE_MODEL} ===\n")

description = """
Red Wine Quality (UCI). Бинарная классификация: quality >= 6 → хорошее вино (1), иначе 0.
Все признаки числовые, физико-химические:
  fixed_acidity, volatile_acidity, citric_acid — виды кислотности
  residual_sugar — остаточный сахар
  chlorides — соли хлора
  free_sulfur_dioxide, total_sulfur_dioxide — диоксид серы (консервант)
  density — плотность
  pH — кислотность шкала
  sulphates — сульфаты (антисептик)
  alcohol — крепость

Полезные соотношения: free/total SO2, volatile/fixed acidity, alcohol/density.
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
    target="quality",
    goal="Максимизировать ROC AUC. Добавляй по ОДНОМУ новому признаку за итерацию — "
         "так видно что именно помогает. Фокус на соотношениях и произведениях признаков.",
    test_df=test_features,
    description=description,
    oversample=True,
)

kept     = [e for e in res.experiment_log if e.get("decision") == "keep"]
rejected = [e for e in res.experiment_log if e.get("decision") == "reject"]
print(f"\n{'─'*45}")
print(f"Baseline:            {res.baseline_cv:.4f}")
print(f"После FE агента:     {res.final_cv:.4f}  ({res.final_cv - res.baseline_cv:+.4f})")
print(f"Принято / отклонено: {len(kept)} / {len(rejected)}")
if res.submission_path:
    print(f"Submission: {res.submission_path}")
print()
print(res.report)
