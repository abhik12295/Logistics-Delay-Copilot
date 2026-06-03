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


def haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Calculate approximate straight-line distance between two lat/lng points.

    This is used as a country-neutral distance feature. In the final research
    dataset, this can be replaced or supplemented with actual route distance
    if trajectory data is available.
    """
    earth_radius_km = 6371.0

    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def add_event_features(df: pd.DataFrame, expected_speed_kmph: float = 25.0) -> pd.DataFrame:
    """
    Create delay, event-sequence, workload, and distance-aware route features
    from logistics event logs.
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

    data.loc[
        missing_completed | impossible_sequence,
        "event_sequence_abnormality_score",
    ] = 1.0

    data["time_window_violation_score"] = np.where(
        data["delay_minutes"] > 0,
        np.minimum(data["delay_minutes"] / 240, 1.0),
        0.0,
    )

    data["workload_pressure_score"] = np.minimum(
        data["courier_workload_2h"].fillna(0) / 20,
        1.0,
    )

    has_coordinates = {
        "origin_lat",
        "origin_lng",
        "destination_lat",
        "destination_lng",
    }.issubset(data.columns)

    if has_coordinates:
        data["distance_km"] = haversine_distance_km(
            data["origin_lat"],
            data["origin_lng"],
            data["destination_lat"],
            data["destination_lng"],
        )

        # Avoid zero-distance divide issues for nearby synthetic points.
        data["distance_km"] = data["distance_km"].clip(lower=0.25)

        data["expected_execution_time_minutes"] = (
            data["distance_km"] / expected_speed_kmph
        ) * 60

        data["distance_adjusted_execution_ratio"] = (
            data["execution_time_minutes"].fillna(data["expected_execution_time_minutes"])
            / data["expected_execution_time_minutes"]
        )

        data["route_execution_instability_score"] = np.minimum(
            data["distance_adjusted_execution_ratio"] / 2,
            1.0,
        )
    else:
        data["distance_km"] = np.nan
        data["expected_execution_time_minutes"] = np.nan
        data["distance_adjusted_execution_ratio"] = np.nan

        median_execution = data["execution_time_minutes"].median(skipna=True)
        if pd.isna(median_execution) or median_execution == 0:
            median_execution = 120

        data["route_execution_instability_score"] = np.minimum(
            data["execution_time_minutes"].fillna(median_execution)
            / (median_execution * 3),
            1.0,
        )

    return data