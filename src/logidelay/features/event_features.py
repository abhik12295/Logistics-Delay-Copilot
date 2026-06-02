from __future__ import annotations

import numpy as np
import pandas as pd


TIME_COLUMNS = [
    "assigned_time",
    "accepted_time",
    "pickup_time",
    "completed_time",
    "promised_delivery_time",
]


def _minutes_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 60


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create delay and event-sequence features from logistics event logs.
    """
    data = df.copy()

    for col in TIME_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    data["acceptance_gap_minutes"] = _minutes_between(
        data["accepted_time"], data["assigned_time"]
    )
    data["pickup_gap_minutes"] = _minutes_between(
        data["pickup_time"], data["accepted_time"]
    )
    data["execution_time_minutes"] = _minutes_between(
        data["completed_time"], data["pickup_time"]
    )
    data["delay_minutes"] = _minutes_between(
        data["completed_time"], data["promised_delivery_time"]
    )

    data["delay_minutes"] = data["delay_minutes"].fillna(999)
    data["delay_minutes"] = data["delay_minutes"].clip(lower=0)

    data["is_delayed"] = data["delay_minutes"] > 0

    data["delay_category"] = pd.cut(
        data["delay_minutes"],
        bins=[-1, 0, 60, 240, np.inf],
        labels=["On Time", "Slight Delay", "Moderate Delay", "Severe Delay"],
    ).astype(str)

    data["event_sequence_abnormality_score"] = 0.0

    missing_completed = data["completed_time"].isna()
    impossible_sequence = (
        (data["accepted_time"] < data["assigned_time"])
        | (data["pickup_time"] < data["accepted_time"])
        | (
            data["completed_time"].notna()
            & (data["completed_time"] < data["pickup_time"])
        )
    )

    data.loc[missing_completed | impossible_sequence, "event_sequence_abnormality_score"] = 1.0

    data["time_window_violation_score"] = np.where(
        data["delay_minutes"] > 0,
        np.minimum(data["delay_minutes"] / 240, 1.0),
        0.0,
    )

    data["workload_pressure_score"] = np.minimum(
        data["courier_workload_2h"].fillna(0) / 20,
        1.0,
    )

    median_execution = data["execution_time_minutes"].median(skipna=True)
    if pd.isna(median_execution) or median_execution == 0:
        median_execution = 120

    data["route_execution_instability_score"] = np.minimum(
        data["execution_time_minutes"].fillna(median_execution) / (median_execution * 3),
        1.0,
    )

    return data