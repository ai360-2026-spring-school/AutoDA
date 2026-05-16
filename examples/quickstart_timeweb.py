import pandas as pd
from autoda import PDAgent

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

agent = PDAgent(
    provider="timeweb",
    max_iterations=12,
    patience=4,
)

result = agent.run(
    df=df,
    goal="Improve CV for survival prediction",
    target="Survived",
)

print(f"baseline {result.baseline_cv:.4f} -> final {result.final_cv:.4f}")
print(result.report)
