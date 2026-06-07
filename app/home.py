from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.copilot.explanation_engine import generate_explanation
from logidelay.utils.app_helpers import load_data_from_sidebar
from logidelay.copilot.explanation_engine import generate_explanation
from logidelay.diagnosis.weak_labeler import add_root_cause_labels
from logidelay.features.event_features import add_event_features
from logidelay.severity.scoring import add_operational_exception_severity


st.set_page_config(
    page_title="LogiDelay Copilot",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 LogiDelay Copilot")
st.subheader("Explainable AI for Logistics Delay Diagnosis")

st.markdown(
    """
LogiDelay Copilot is a research prototype for diagnosing logistics delivery delays
using event logs and **Operational Exception Severity**.

The system identifies delayed deliveries, diagnoses the likely operational cause,
assigns a severity score, and generates a planner-facing explanation with a
recommended action.

This MVP uses free/open-source Python logic only. No paid AI API is required.
"""
)

# sample_path = ROOT_DIR / "data" / "sample" / "sample_logistics_events.csv"

# if not sample_path.exists():
#     st.warning(
#         "Sample data not found. Run: `uv run python scripts/prepare_sample_data.py`"
#     )
#     st.stop()

# df = pd.read_csv(sample_path)

# # Recalculate if columns are missing
# required_cols = {
#     "delay_minutes",
#     "operational_exception_severity",
#     "severity_class",
#     "root_cause_label",
# }
# if not required_cols.issubset(df.columns):
#     df = add_event_features(df)
#     df = add_operational_exception_severity(df)
#     df = add_root_cause_labels(df)
df = load_data_from_sidebar(str(ROOT_DIR))

st.sidebar.download_button(
    label="Download current dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="logidelay_current_dataset.csv",
    mime="text/csv",
)

total_deliveries = len(df)
delayed_count = int(df["is_delayed"].sum())
avg_delay = df["delay_minutes"].mean()
critical_count = int((df["severity_class"] == "Critical").sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_deliveries:,}")
col2.metric("Delayed Records", f"{delayed_count:,}")
col3.metric("Avg Delay Minutes", f"{avg_delay:.1f}")
col4.metric("Critical Exceptions", f"{critical_count:,}")

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("### Delay Category Distribution")
    delay_chart = (
        df["delay_category"]
        .value_counts()
        .rename_axis("delay_category")
        .reset_index(name="count")
    )
    fig = px.bar(
        delay_chart,
        x="delay_category",
        y="count",
        text="count",
        title="Deliveries by Delay Category",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Operational Exception Severity")
    severity_chart = (
        df["severity_class"]
        .value_counts()
        .rename_axis("severity_class")
        .reset_index(name="count")
    )
    fig = px.bar(
        severity_chart,
        x="severity_class",
        y="count",
        text="count",
        title="Deliveries by Severity Class",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("### Delivery Diagnosis Explorer")

package_options = df["package_id"].tolist()
selected_package = st.selectbox("Select a package", package_options)

record = df[df["package_id"] == selected_package].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Root Cause", record["root_cause_label"])
c2.metric("Severity", record["severity_class"])
c3.metric("OES Score", f"{record['operational_exception_severity']:.2f}")

st.markdown("#### Event Evidence")

evidence_cols = [
    "package_id",
    "courier_id",
    "city",
    "zone_id",
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
    "distance_km",
    "assigned_time",
    "accepted_time",
    "pickup_time",
    "completed_time",
    "promised_delivery_time",
    "expected_execution_time_minutes",
    "distance_adjusted_execution_ratio",
    "acceptance_gap_minutes",
    "pickup_gap_minutes",
    "execution_time_minutes",
    "delay_minutes",
    "courier_workload_2h",
    "root_cause_label",
    "severity_class",
]

available_cols = [col for col in evidence_cols if col in df.columns]
st.dataframe(pd.DataFrame([record[available_cols]]), use_container_width=True)

st.markdown("#### Copilot Explanation")
st.info(generate_explanation(record))

st.divider()

st.markdown("### Research Pipeline")

st.code(
    """
Logistics Event Logs
→ Event Feature Engineering
→ Delay Detection
→ Root-Cause Weak Labeling
→ Operational Exception Severity Scoring
→ Free GenAI-style Explanation
→ Planner Recommendation
""",
    language="text",
)

st.markdown(
    """
### Next Research Output

Use the **Research Results** page to generate paper-ready summary tables for:

- delay distribution
- root-cause distribution
- severity distribution
- average OES by root cause
- top exception examples
"""
)