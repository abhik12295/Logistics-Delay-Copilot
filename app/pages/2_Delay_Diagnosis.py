from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.copilot.explanation_engine import generate_explanation
from logidelay.utils.app_helpers import load_sample_data

st.set_page_config(page_title="Delay Diagnosis", page_icon="🧭", layout="wide")

st.title("🧭 Delay Diagnosis")
st.markdown(
    """
Select a delivery/package record and inspect the diagnosed delay cause,
supporting event evidence, and planner-facing explanation.
"""
)

df = load_sample_data(str(ROOT_DIR))

severity_filter = st.multiselect(
    "Filter by severity",
    options=sorted(df["severity_class"].dropna().unique().tolist()),
    default=sorted(df["severity_class"].dropna().unique().tolist()),
)

cause_filter = st.multiselect(
    "Filter by root cause",
    options=sorted(df["root_cause_label"].dropna().unique().tolist()),
    default=sorted(df["root_cause_label"].dropna().unique().tolist()),
)

filtered = df[
    df["severity_class"].isin(severity_filter)
    & df["root_cause_label"].isin(cause_filter)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

selected_package = st.selectbox(
    "Select package",
    filtered["package_id"].tolist(),
)

record = filtered[filtered["package_id"] == selected_package].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Delay Category", record["delay_category"])
col2.metric("Root Cause", record["root_cause_label"])
col3.metric("Severity", record["severity_class"])
col4.metric("OES Score", f"{record['operational_exception_severity']:.2f}")

st.divider()

st.subheader("Event Timeline")

timeline_cols = [
    "assigned_time",
    "accepted_time",
    "pickup_time",
    "completed_time",
    "promised_delivery_time",
]

timeline = pd.DataFrame(
    {
        "event": timeline_cols,
        "timestamp": [record.get(col) for col in timeline_cols],
    }
)

st.dataframe(timeline, use_container_width=True)

st.subheader("Evidence Features")

evidence_cols = [
    "acceptance_gap_minutes",
    "pickup_gap_minutes",
    "execution_time_minutes",
    "delay_minutes",
    "courier_workload_2h",
    "workload_pressure_score",
    "event_sequence_abnormality_score",
    "time_window_violation_score",
    "route_execution_instability_score",
]

st.dataframe(pd.DataFrame([record[evidence_cols]]), use_container_width=True)

st.subheader("Copilot Explanation")
st.info(generate_explanation(record))