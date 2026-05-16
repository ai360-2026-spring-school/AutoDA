import pandas as pd
from autoda import PDAgent

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

agent = PDAgent(
    provider="timeweb",
    model="timeweb-agent",  # может игнорироваться Timeweb, но нужен клиенту
    max_iterations=6,
    temperature=0,
    max_tokens=1000,
)

result = agent.run(
    df=df,
    goal="Проведи первичный анализ датасета",
    target="Survived",
)

print(result.report)
