from __future__ import annotations

import numpy as np
import pandas as pd


def add_operational_exception_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Operational Exception Severity.

    OES is a country-neutral severity score that measures how serious an
    abnormal logistics event is based on delay duration, time-window violation,
    event abnormality, workload pressure, and route execution instability.
    """
    data = df.copy()

    delay_duration_score = np.minimum(data["delay_minutes"].fillna(0) / 360, 1.0)

    data["operational_exception_severity"] = (
        0.35 * delay_duration_score
        + 0.25 * data["time_window_violation_score"].fillna(0)
        + 0.15 * data["event_sequence_abnormality_score"].fillna(0)
        + 0.15 * data["workload_pressure_score"].fillna(0)
        + 0.10 * data["route_execution_instability_score"].fillna(0)
    )

    data["operational_exception_severity"] = data[
        "operational_exception_severity"
    ].clip(0, 1)

    data["severity_class"] = pd.cut(
        data["operational_exception_severity"],
        bins=[-0.001, 0.24, 0.49, 0.74, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    return data