from __future__ import annotations

import numpy as np
import pandas as pd

from logidelay.features.event_features import haversine_distance_km


TIME_COLUMNS = [
    "accepted_time",
    "service_window_start_time",
    "promised_delivery_time",
    "completed_time",
]


def _minutes_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 60


def _clip_score(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def add_service_window_breach_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the supervised ML target.

    Target:
        service_window_breach = 1 if completed_time > promised_delivery_time
        service_window_breach = 0 otherwise

    For LaDe-P:
        completed_time = pickup_time
        promised_delivery_time = time_window_end

    This label is used only for training/evaluation. It must not be used as a
    model feature during proactive prediction.
    """
    data = df.copy()

    for col in ["completed_time", "promised_delivery_time"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")

    data["service_window_breach"] = (
        data["completed_time"] > data["promised_delivery_time"]
    ).astype(int)

    return data


def add_proactive_breach_features(
    df: pd.DataFrame,
    expected_speed_kmph: float = 25.0,
    workload_capacity: float = 20.0,
) -> pd.DataFrame:
    """
    Create proactive features available at task acceptance time.

    These features are designed to avoid data leakage. They should only use
    information available before pickup completion.

    Safe proactive features include:
        - accepted_time
        - service_window_start_time
        - promised_delivery_time
        - distance/location
        - courier workload
        - city/zone/task context
        - historical risk features computed without target leakage
    """
    data = df.copy()

    for col in TIME_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    required_time_cols = {"accepted_time", "promised_delivery_time"}
    missing_time_cols = required_time_cols - set(data.columns)

    if missing_time_cols:
        raise ValueError(
            f"Missing required time columns for proactive features: {missing_time_cols}"
        )

    # Time pressure features available at acceptance time.
    data["time_to_window_end_minutes"] = _minutes_between(
        data["promised_delivery_time"],
        data["accepted_time"],
    )

    if "service_window_start_time" in data.columns:
        data["time_to_window_start_minutes"] = _minutes_between(
            data["service_window_start_time"],
            data["accepted_time"],
        )
        data["service_window_length_minutes"] = _minutes_between(
            data["promised_delivery_time"],
            data["service_window_start_time"],
        )
    else:
        data["time_to_window_start_minutes"] = np.nan
        data["service_window_length_minutes"] = np.nan

    # Calendar/context features available at acceptance time.
    data["accept_hour"] = data["accepted_time"].dt.hour
    data["accept_dayofweek"] = data["accepted_time"].dt.dayofweek
    data["accept_month"] = data["accepted_time"].dt.month

    # Distance features.
    has_coordinates = {
        "origin_lat",
        "origin_lng",
        "destination_lat",
        "destination_lng",
    }.issubset(data.columns)

    if has_coordinates:
        if "distance_km" not in data.columns:
            data["distance_km"] = haversine_distance_km(
                data["origin_lat"],
                data["origin_lng"],
                data["destination_lat"],
                data["destination_lng"],
            )

        data["distance_km"] = pd.to_numeric(
            data["distance_km"], errors="coerce"
        ).clip(lower=0.25)

    else:
        data["distance_km"] = np.nan

    data["expected_travel_time_minutes"] = (
        data["distance_km"] / expected_speed_kmph
    ) * 60

    # Main operational feature:
    # how much time remains after estimated travel time.
    data["feasibility_margin_minutes"] = (
        data["time_to_window_end_minutes"] - data["expected_travel_time_minutes"]
    )

    # Normalized proactive pressure scores.
    # Less remaining time means higher pressure.
    data["time_pressure_score"] = 1 - (
        data["time_to_window_end_minutes"] / 240
    )
    data["time_pressure_score"] = _clip_score(data["time_pressure_score"])

    data["workload_pressure_score"] = (
        pd.to_numeric(data.get("courier_workload_2h", 0), errors="coerce").fillna(0)
        / workload_capacity
    )
    data["workload_pressure_score"] = _clip_score(data["workload_pressure_score"])

    # Negative or small feasibility margin means high pressure.
    # If margin >= 120 minutes, pressure is near 0.
    # If margin <= 0 minutes, pressure is 1.
    data["distance_feasibility_pressure_score"] = 1 - (
        data["feasibility_margin_minutes"] / 120
    )
    data["distance_feasibility_pressure_score"] = _clip_score(
        data["distance_feasibility_pressure_score"]
    )

    return data


def add_historical_breach_rates(
    df: pd.DataFrame,
    target_col: str = "service_window_breach",
) -> pd.DataFrame:
    """
    Add simple historical breach-rate features.

    Important:
    This function uses expanding averages shifted by one row within each group.
    That avoids using the current row's label to predict itself.

    These features approximate prior courier/zone risk at the time of prediction.
    """
    data = df.copy()

    if target_col not in data.columns:
        raise ValueError(
            f"Missing target column '{target_col}'. "
            "Run add_service_window_breach_label first."
        )

    if "accepted_time" in data.columns:
        data["accepted_time"] = pd.to_datetime(data["accepted_time"], errors="coerce")
        data = data.sort_values("accepted_time")

    data[target_col] = pd.to_numeric(data[target_col], errors="coerce").fillna(0)

    global_prior = float(data[target_col].mean())

    if "courier_id" in data.columns:
        data["historical_courier_breach_rate"] = (
            data.groupby("courier_id")[target_col]
            .transform(lambda s: s.shift().expanding().mean())
            .fillna(global_prior)
        )
    else:
        data["historical_courier_breach_rate"] = global_prior

    if "zone_id" in data.columns:
        data["historical_zone_breach_rate"] = (
            data.groupby("zone_id")[target_col]
            .transform(lambda s: s.shift().expanding().mean())
            .fillna(global_prior)
        )
    else:
        data["historical_zone_breach_rate"] = global_prior

    if "city" in data.columns:
        data["historical_city_breach_rate"] = (
            data.groupby("city")[target_col]
            .transform(lambda s: s.shift().expanding().mean())
            .fillna(global_prior)
        )
    else:
        data["historical_city_breach_rate"] = global_prior

    return data


def add_intervention_urgency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an intervention urgency score for dispatch prioritization.

    In rare-event logistics prediction, calibrated probabilities are often small.
    For example, a 7% breach probability may be very high when the base breach rate
    is below 1%. Therefore, the urgency score uses the model risk percentile/rank
    rather than relying only on raw probability magnitude.

    Formula:
        Intervention Urgency Score =
            0.65 * model_risk_rank_score
          + 0.20 * time_pressure_score
          + 0.10 * distance_feasibility_pressure_score
          + 0.05 * workload_pressure_score

    If predicted_breach_probability is unavailable, the function falls back to a
    rule-based risk pressure score.
    """
    data = df.copy()

    if "predicted_breach_probability" in data.columns:
        predicted_probability = pd.to_numeric(
            data["predicted_breach_probability"],
            errors="coerce",
        ).fillna(0)

        data["predicted_breach_probability"] = predicted_probability

        # Percentile rank: highest-risk task approaches 1.0.
        data["model_risk_rank_score"] = predicted_probability.rank(
            method="average",
            pct=True,
        )

    else:
        # Temporary fallback before ML predictions exist.
        fallback_risk_score = (
            0.45 * data["time_pressure_score"].fillna(0)
            + 0.35 * data["distance_feasibility_pressure_score"].fillna(0)
            + 0.20 * data["workload_pressure_score"].fillna(0)
        )

        data["model_risk_rank_score"] = fallback_risk_score.rank(
            method="average",
            pct=True,
        )

    data["intervention_urgency_score"] = (
        0.65 * data["model_risk_rank_score"].fillna(0)
        + 0.20 * data["time_pressure_score"].fillna(0)
        + 0.10 * data["distance_feasibility_pressure_score"].fillna(0)
        + 0.05 * data["workload_pressure_score"].fillna(0)
    )

    data["intervention_urgency_score"] = _clip_score(
        data["intervention_urgency_score"]
    )

    data["intervention_priority"] = pd.cut(
        data["intervention_urgency_score"],
        bins=[-0.01, 0.35, 0.60, 0.80, 1.00],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    return data


def build_proactive_breach_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end helper for creating the proactive breach modeling dataset.
    """
    data = add_service_window_breach_label(df)
    data = add_proactive_breach_features(data)
    data = add_historical_breach_rates(data)
    data = add_intervention_urgency_score(data)

    return data