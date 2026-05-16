import pandas as pd
from autoda import PDAgent

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
train = df.sample(frac=0.8, random_state=0)
test = df.drop(train.index).drop(columns=["Survived"])

agent = PDAgent(provider="gigachat", max_iterations=12)
res = agent.run(
    df=train,
    target="Survived",
    goal="Improve CV for survival prediction",
    test_df=test,
    description="Titanic passenger survival. Target is Survived (0/1). PassengerId is a row id, not a feature.",
)

print(f"baseline {res.baseline_cv:.4f} -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"info-tool calls: {len(res.info_tool_results)}")
if res.final_test_df is not None:
    print(f"test shape: {res.final_test_df.shape}")
if res.submission_path:
    print(f"submission: {res.submission_path}")
print(res.report)
