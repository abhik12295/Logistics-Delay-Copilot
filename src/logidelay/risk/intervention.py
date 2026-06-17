from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def classify_breach_risk(probability: float) -> str:
    """
    Convert predicted breach probability into a human-readable risk level.
    """
    probability = _safe_float(probability)

    if probability >= 0.75:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.25:
        return "Medium"
    return "Low"


def identify_primary_risk_factor(row: pd.Series) -> str:
    """
    Identify the strongest operational risk factor for dispatcher explanation.
    """
    time_pressure = _safe_float(row.get("time_pressure_score"))
    workload_pressure = _safe_float(row.get("workload_pressure_score"))
    distance_pressure = _safe_float(row.get("distance_feasibility_pressure_score"))

    risk_scores = {
        "time pressure": time_pressure,
        "courier workload pressure": workload_pressure,
        "distance feasibility pressure": distance_pressure,
    }

    return max(risk_scores, key=risk_scores.get)


def generate_dispatch_recommendation(row: pd.Series) -> str:
    """
    Generate a practical dispatcher recommendation from model evidence.
    """
    breach_probability = _safe_float(row.get("predicted_breach_probability"))
    urgency_score = _safe_float(row.get("intervention_urgency_score"))
    time_left = _safe_float(row.get("time_to_window_end_minutes"))
    feasibility_margin = _safe_float(row.get("feasibility_margin_minutes"))
    workload = _safe_float(row.get("courier_workload_2h"))
    distance_km = _safe_float(row.get("distance_km"))
    expected_travel = _safe_float(row.get("expected_travel_time_minutes"))

    risk_level = classify_breach_risk(breach_probability)
    primary_factor = identify_primary_risk_factor(row)

    if risk_level in {"Critical", "High"}:
        if feasibility_margin < 0:
            action = (
                "Prioritize immediate dispatch intervention. The estimated travel time "
                "exceeds the remaining time before the service-window deadline. "
                "Consider reassignment to a closer courier or direct escalation."
            )
        elif workload >= 15:
            action = (
                "Review courier workload and consider rebalancing. The task has high "
                "breach risk and the courier appears heavily loaded."
            )
        elif time_left <= 30:
            action = (
                "Prioritize this pickup immediately. The task is close to the service-window "
                "deadline and should be monitored until completion."
            )
        else:
            action = (
                "Monitor this task closely and confirm courier progress. The model indicates "
                "elevated breach probability based on current time, distance, and workload signals."
            )

    elif risk_level == "Medium":
        action = (
            "Monitor this pickup task. No immediate reassignment is required, but dispatch "
            "should check progress if the service window continues to narrow."
        )

    else:
        action = (
            "No immediate intervention is required. Continue normal monitoring unless the "
            "task status changes."
        )

    return (
        f"Risk level: {risk_level}. "
        f"Primary risk factor: {primary_factor}. "
        f"Predicted breach probability is {breach_probability:.1%}, with an intervention "
        f"urgency score of {urgency_score:.2f}. "
        f"Time remaining before window end is {time_left:.1f} minutes, estimated travel "
        f"time is {expected_travel:.1f} minutes, feasibility margin is "
        f"{feasibility_margin:.1f} minutes, distance is {distance_km:.2f} km, and "
        f"courier workload in the recent 2-hour window is {workload:.0f}. "
        f"Recommended action: {action}"
    )


def generate_short_action(row: pd.Series) -> str:
    """
    Short action text for intervention queue table.
    """
    breach_probability = _safe_float(row.get("predicted_breach_probability"))
    feasibility_margin = _safe_float(row.get("feasibility_margin_minutes"))
    time_left = _safe_float(row.get("time_to_window_end_minutes"))
    workload = _safe_float(row.get("courier_workload_2h"))

    risk_level = classify_breach_risk(breach_probability)

    if risk_level in {"Critical", "High"}:
        if feasibility_margin < 0:
            return "Reassign or escalate immediately"
        if time_left <= 30:
            return "Prioritize pickup now"
        if workload >= 15:
            return "Review workload and rebalance"
        return "Monitor closely"

    if risk_level == "Medium":
        return "Monitor progress"

    return "No immediate action"