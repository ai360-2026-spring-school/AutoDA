"""
AutoDA new architecture showcase: deterministic preprocessing,
description summarisation, multi-turn planner loop, and critic.
Uses the F1 pit-stop dataset.
"""

import pandas as pd
from autoda import PDAgent

train = pd.read_csv("test_data/pitstop/train_mini.csv")
test = pd.read_csv("test_data/pitstop/test.csv")

description = [
    """
    This dataset provides a lap-level view of Formula 1 races,
    designed specifically for race strategy analysis and machine learning applications.
    It transforms raw telemetry data into a structured format with
    engineered features that capture tire degradation, race progression,
    and driver performance dynamics.
    Lap-by-lap race data for each driver.
    Tire compound and tire life tracking.
    Lap time and degradation metrics.
    Position changes across laps.
    Race progress indicators.
    Pit stop detection.
    Target variable: PitNextLap (predict whether a driver will pit next lap).
    """
]

agent = PDAgent(
    provider="gigachat",
    max_iterations=10,
    tolerance=0.001,
    metric="f1",
    max_inner_turns=15,
    debug=True
)

res = agent.run(
    df=train,
    target="PitNextLap",
    goal="Improve F1 metric for pit stop prediction",
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
