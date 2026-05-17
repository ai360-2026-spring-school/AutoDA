"""
AutoDA v3 showcase: demonstrates reports subfolder, description injection,
and submission writing. Uses the Titanic dataset as a stand-in.
Swap the URL and target/id for your actual dataset.
"""

import pandas as pd
from autoda import PDAgent

train = pd.read_csv("test_data/pitstop/train.csv")
test = pd.read_csv("test_data/pitstop/test.csv")

description = [
    """
    This dataset provides a lap-level view of Formula 1 races, 
    designed specifically for race strategy analysis and machine learning applications.
It transforms raw telemetry data into a structured format with 
engineered features that capture tire degradation, race progression, 
and driver performance dynamics.
Lap-by-lap race data for each driver
Tire compound and tire life tracking
Lap time and degradation metrics
Position changes across laps
Race progress indicators
Pit stop detection
Target variable: PitNextLap (predict whether a driver will pit next lap)"""
]

agent = PDAgent(provider="gigachat", max_iterations=10, tolerance=-0.001, metric="f1")
res = agent.run(
    df=train,
    target="PitNextLap",
    goal="Improve metrics",
    test_df=test,
    description=description,
)

print(f"baseline {res.baseline_cv:.4 -> final {res.final_cv:.4f}")
print(f"applied pipeline: {len(res.applied_pipeline)} steps")
print(f"info-tool calls: {len(res.info_tool_results)}")
if res.final_test_df is not None:
    print(f"test transformed shape: {res.final_test_df.shape}")
if res.submission_path:
    print(f"submission: {res.submission_path}")
    print(res.submission_df.head())

print(res.report)
