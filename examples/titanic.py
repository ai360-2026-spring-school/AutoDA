"""
AutoDA new architecture showcase: deterministic preprocessing,
description summarisation, multi-turn planner loop, and critic.
Uses the F1 pit-stop dataset.
"""

import pandas as pd
from autoda import PDAgent

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
train = df.sample(frac=0.8, random_state=0)
test = df.drop(train.index).drop(columns=["Survived"])

description = [
    """
The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS 
Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough 
lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of 
people were more likely to survive than others.
In this challenge, we ask you to build a predictive model that answers the question: 
“what sorts of people were more likely to survive?” using passenger data 
(ie name, age, gender, socio-economic class, etc).
"""
]

agent = PDAgent(
    provider="gigachat",
    max_iterations=50,
    tolerance=0.001,
    metric="f1",
    max_inner_turns=15,
    debug=True,
)

res = agent.run(
    df=train,
    target="Survived",
    goal="Improve F1 metric for survival prediction",
    test_df=test,
    description=description,
    ohe_max_cardinality=12,
    oversample=True,
)

print(f"baseline {res.baseline_cv:.4f} -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"experiment log entries: {len(res.experiment_log)}")

if res.final_test_df is not None:
    print(f"test transformed shape: {res.final_test_df.shape}")
if res.submission_path:
    print(f"submission: {res.submission_path}")
    if res.submission_df is not None:
        print(res.submission_df.head())

print("\nColumn type map (sample):")
for col, kind in list(res.column_type_map.items())[:10]:
    print(f"  {col}: {kind}")

print("\n--- Report ---")
print(res.report)