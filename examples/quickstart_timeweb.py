import pandas as pd
from autoda import PDAgent, replay

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
train = df.sample(frac=0.8, random_state=0)
test  = df.drop(train.index).drop(columns=["Survived"])

agent = PDAgent(provider="timeweb", max_iterations=12, tolerance=1e-4)
res = agent.run(df=train, target="Survived", goal="Improve CV for survival prediction", test_df=test)

print(f"baseline {res.baseline_cv:.4f} -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"info-tool calls: {len(res.info_tool_results)}")
if res.final_test_df is not None:
    print(f"test transformed shape: {res.final_test_df.shape}")
print(res.report)
