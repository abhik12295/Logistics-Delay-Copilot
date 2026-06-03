from __future__ import annotations

import pandas as pd

from logidelay.copilot.recommendation_engine import recommend_action


def generate_explanation(row: pd.Series) -> str:
    """
    Free GenAI-style grounded explanation.

    This does not use any paid API. It generates a structured natural-language
    explanation from event facts, diagnosis output, severity score, and distance-aware
    route execution evidence.
    """
    root_cause = row.get("root_cause_label", "Unknown")
    severity = row.get("severity_class", "Unknown")
    action = recommend_action(root_cause, severity)

    delay_minutes = float(row.get("delay_minutes", 0))
    acceptance_gap = float(row.get("acceptance_gap_minutes", 0))
    pickup_gap = float(row.get("pickup_gap_minutes", 0))
    workload_score = float(row.get("workload_pressure_score", 0))
    event_score = float(row.get("event_sequence_abnormality_score", 0))
    severity_score = float(row.get("operational_exception_severity", 0))

    distance_km = row.get("distance_km", None)
    expected_execution = row.get("expected_execution_time_minutes", None)
    actual_execution = row.get("execution_time_minutes", None)
    distance_ratio = row.get("distance_adjusted_execution_ratio", None)

    evidence = (
        f"The delivery has a delay of {delay_minutes:.0f} minutes. "
        f"The task acceptance gap is {acceptance_gap:.0f} minutes, "
        f"the pickup/task start gap is {pickup_gap:.0f} minutes, "
        f"the workload pressure score is {workload_score:.2f}, "
        f"and the event abnormality score is {event_score:.2f}."
    )

    distance_evidence = ""
    if pd.notna(distance_km) and pd.notna(expected_execution) and pd.notna(actual_execution):
        distance_evidence = (
            f" The route distance is approximately {float(distance_km):.2f} km. "
            f"The expected execution time based on distance is "
            f"{float(expected_execution):.0f} minutes, while the actual execution time is "
            f"{float(actual_execution):.0f} minutes."
        )

        if pd.notna(distance_ratio):
            distance_evidence += (
                f" The distance-adjusted execution ratio is "
                f"{float(distance_ratio):.2f}."
            )

    explanation = (
        f"Likely root cause: {root_cause}. "
        f"Operational Exception Severity is {severity} "
        f"with a score of {severity_score:.2f}. "
        f"{evidence}"
        f"{distance_evidence} "
        f"Recommended action: {action}"
    )

    return explanation