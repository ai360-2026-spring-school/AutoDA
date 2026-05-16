"""
AutoDA v3 showcase: demonstrates reports subfolder, description injection,
and submission writing. Uses the Titanic dataset as a stand-in.
Swap the URL and target/id for your actual dataset.
"""

import pandas as pd
from autoda import PDAgent

train = pd.read_csv("test_data/pitstop/train.csv")
test = pd.read_csv("test_data/pitstop/test.csv")

description = ["""
               The dataset for this competition (both train and test) 
               was inspired by F1 strategy dataset. Feature distributions 
               are close to, but not exactly the same, as the original, 
               and we intentionally remove Normalized_TyreLife which makes 
               the prediction trivial. Feel free to use the original 
               dataset as part of this competition, both to explore 
               differences as well as to see whether incorporating the 
               original in training improves model performance."""]

agent = PDAgent(provider="gigachat", max_iterations=12)
res = agent.run(
    df=train,
    target="PitNextLap",
    goal="Improve CV for Formula-1 pitstop prediction",
    test_df=test,
    description=description,
)

print(f"baseline {res.baseline_cv:.4f} -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"info-tool calls: {len(res.info_tool_results)}")
if res.final_test_df is not None:
    print(f"test transformed shape: {res.final_test_df.shape}")
if res.submission_path:
    print(f"submission: {res.submission_path}")
    print(res.submission_df.head())

print(res.report)
