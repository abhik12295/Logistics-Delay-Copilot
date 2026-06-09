from __future__ import annotations

import pandas as pd


def diagnose_root_cause(row: pd.Series) -> str:
    """
    Transparent weak-labeling rules for logistics delay diagnosis.

    The logic first checks for event-data issues. If the task is not delayed and
    the event sequence is valid, the record is labeled as No significant delay.
    This prevents normal waiting time before a service window from being
    incorrectly labeled as an operational delay.
    """
    if row.get("event_sequence_abnormality_score", 0) >= 1:
        return "Event-data inconsistency"

    if row.get("delay_minutes", 0) <= 0:
        return "No significant delay"

    if row.get("acceptance_gap_minutes", 0) >= 75:
        return "Courier acceptance delay"

    if row.get("pickup_gap_minutes", 0) >= 90:
        return "Pickup/task start delay"

    if row.get("workload_pressure_score", 0) >= 0.65:
        return "Workload pressure"

    if row.get("route_execution_instability_score", 0) >= 0.75:
        return "Route execution instability"

    if row.get("delay_minutes", 0) > 0:
        return "Delivery/service execution delay"

    return "No significant delay"


def add_root_cause_labels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["root_cause_label"] = data.apply(diagnose_root_cause, axis=1)
    return data