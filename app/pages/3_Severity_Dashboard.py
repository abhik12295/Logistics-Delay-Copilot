from __future__ import annotations

from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.utils.app_helpers import load_data_from_sidebar

st.set_page_config(page_title="Severity Dashboard", page_icon="🚦", layout="wide")

st.title("🚦 Operational Exception Severity Dashboard")

st.markdown(
    """
Operational Exception Severity measures how serious an abnormal logistics event is.
It combines delay duration, time-window violation, event abnormality, workload pressure,
and route execution instability.
"""
)

df = load_data_from_sidebar(str(ROOT_DIR))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average OES", f"{df['operational_exception_severity'].mean():.2f}")
col2.metric("High Exceptions", f"{(df['severity_class'] == 'High').sum():,}")
col3.metric("Critical Exceptions", f"{(df['severity_class'] == 'Critical').sum():,}")
col4.metric("Max Delay Minutes", f"{df['delay_minutes'].max():.0f}")

st.divider()

left, right = st.columns(2)

with left:
    severity_counts = df["severity_class"].value_counts().reset_index()
    severity_counts.columns = ["severity_class", "count"]

    fig = px.bar(
        severity_counts,
        x="severity_class",
        y="count",
        text="count",
        title="Severity Class Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        df,
        x="severity_class",
        y="delay_minutes",
        title="Delay Minutes by Severity Class",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Top High-Severity Records")

top_records = df.sort_values(
    "operational_exception_severity", ascending=False
).head(25)

display_cols = [
    "package_id",
    "courier_id",
    "city",
    "zone_id",
    "delay_minutes",
    "root_cause_label",
    "severity_class",
    "operational_exception_severity",
]

st.dataframe(top_records[display_cols], use_container_width=True)