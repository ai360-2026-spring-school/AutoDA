"""
AutoDA example: Balance Scale (UCI).

500 строк в train, 125 в test, 4 признака, мультиклассовая классификация.
Бейзлайн CatBoost на сырых фичах: ≈ 85% accuracy.
Ключевой FE: left_weight × left_distance и right_weight × right_distance —
точная физическая формула момента силы. LLM-FE показывал +13.4% accuracy
(85.6% → 99.0%) именно на этом датасете.
"""

import pandas as pd
from pathlib import Path
from autoda import PDAgent

TRAIN = Path("test_data/balance_scale/train.csv")
TEST  = Path("test_data/balance_scale/test.csv")

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
test_features = test.drop(columns=["class"])

description = """
Balance Scale (UCI). Мультиклассовая классификация: предсказать, в какую сторону
наклонится рычажные весы (class: L=влево, R=вправо, B=сбалансированы).

Признаки:
  left_weight    — вес на левой чаше (1–5)
  left_distance  — расстояние до центра, левая сторона (1–5)
  right_weight   — вес на правой чаше (1–5)
  right_distance — расстояние до центра, правая сторона (1–5)

Физический принцип (момент силы):
  left_moment  = left_weight  × left_distance
  right_moment = right_weight × right_distance
  Если left_moment > right_moment  → L (наклон влево)
  Если left_moment < right_moment  → R (наклон вправо)
  Если left_moment == right_moment → B (равновесие)

Без произведений модель не может найти эту закономерность в явном виде.
Мультипликативные признаки: left_moment, right_moment, moment_diff, moment_ratio.
"""

agent = PDAgent(
    provider="gigachat",
    max_iterations=12,
    tolerance=0.002,
    critic_every=3,
    debug=True,
)

res = agent.run(
    df=train,
    target="class",
    goal="Максимизировать accuracy. Ключ к задаче — создать произведения "
         "weight × distance для каждой стороны весов (момент силы). "
         "Это прямой физический закон, скрытый в сырых признаках.",
    test_df=test_features,
    description=description,
)

print(f"\n{'─'*50}")
print(f"Baseline (mlogloss): {res.baseline_cv:.4f}")
print(f"После FE агента:     {res.final_cv:.4f}  ({res.final_cv - res.baseline_cv:+.4f})")
kept     = [e for e in res.experiment_log if e.get("decision") == "keep"]
rejected = [e for e in res.experiment_log if e.get("decision") == "reject"]
print(f"Принято / отклонено: {len(kept)} / {len(rejected)}")
if res.submission_path:
    print(f"Submission: {res.submission_path}")
print()
print(res.report)
