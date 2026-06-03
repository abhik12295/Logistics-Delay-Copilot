from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.copilot.explanation_engine import generate_explanation
from logidelay.copilot.recommendation_engine import recommend_action
from logidelay.utils.app_helpers import load_data_from_sidebar

st.set_page_config(page_title="Copilot Explanation", page_icon="🤖", layout="wide")

st.title("🤖 Copilot Explanation")
st.markdown(
    """
This page demonstrates the free GenAI-style explanation layer.
The current MVP uses a grounded template engine, not a paid AI API.
"""
)

df = load_data_from_sidebar(str(ROOT_DIR))

selected_package = st.selectbox("Select package", df["package_id"].tolist())
record = df[df["package_id"] == selected_package].iloc[0]

st.subheader("Diagnosis Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Root Cause", record["root_cause_label"])
col2.metric("Severity", record["severity_class"])
col3.metric("OES", f"{record['operational_exception_severity']:.2f}")

st.subheader("Planner-Facing Explanation")
st.info(generate_explanation(record))

st.subheader("Recommended Action")
st.success(
    recommend_action(
        record["root_cause_label"],
        record["severity_class"],
    )
)

st.subheader("Grounded Evidence Used")
evidence = {
    "Distance km": record["distance_km"],
    "Expected execution time minutes": record["expected_execution_time_minutes"],
    "Actual execution time minutes": record["execution_time_minutes"],
    "Distance-adjusted execution ratio": record["distance_adjusted_execution_ratio"],
    "Delay minutes": record["delay_minutes"],
    "Acceptance gap minutes": record["acceptance_gap_minutes"],
    "Pickup gap minutes": record["pickup_gap_minutes"],
    "Courier workload 2h": record["courier_workload_2h"],
    "Workload pressure score": record["workload_pressure_score"],
    "Event abnormality score": record["event_sequence_abnormality_score"],
    "Time-window violation score": record["time_window_violation_score"],
    "Route execution instability score": record["route_execution_instability_score"],
}

st.json(evidence)