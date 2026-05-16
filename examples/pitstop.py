import pandas as pd
from autoda import PDAgent

train = pd.read_csv("test_data/pitstop/train.csv")
test = pd.read_csv("test_data/pitstop/test.csv")

agent = PDAgent(provider="timeweb", max_iterations=12, tolerance=1e-4)
res = agent.run(
    df=train,
    target="PitNextLap",
    goal="Improve CV for pit prediction",
    test_df=test,
)

print(f"baseline {res.baseline_cv:.4f} -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"info-tool calls: {len(res.info_tool_results)}")
if res.final_test_df is not None:
    print(f"test transformed shape: {res.final_test_df.shape}")
print(res.report)
