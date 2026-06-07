from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.utils.app_helpers import load_data_from_sidebar

st.set_page_config(page_title="Research Results", page_icon="📄", layout="wide")

st.title("📄 Research Results Summary")

st.markdown(
    """
This page converts the app outputs into research-friendly summary tables.
These tables can later support the experiment section of the IEEE BigData paper.
"""
)

df = load_data_from_sidebar(str(ROOT_DIR))

st.subheader("1. Dataset Summary")

summary = {
    "Total records": len(df),
    "Unique couriers/carriers": df["courier_id"].nunique(),
    "Unique cities": df["city"].nunique(),
    "Unique zones": df["zone_id"].nunique(),
    "Delayed records": int(df["is_delayed"].sum()),
    "Delayed percentage": round(float(df["is_delayed"].mean() * 100), 2),
    "Average delay minutes": round(float(df["delay_minutes"].mean()), 2),
    "Average OES": round(float(df["operational_exception_severity"].mean()), 3),
    "Critical exceptions": int((df["severity_class"] == "Critical").sum()),
}

summary_df = pd.DataFrame(
    [{"Metric": key, "Value": value} for key, value in summary.items()]
)

st.dataframe(summary_df, use_container_width=True)

st.download_button(
    label="Download dataset summary CSV",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="logidelay_dataset_summary.csv",
    mime="text/csv",
)

st.divider()

st.subheader("2. Delay Category Distribution")

delay_distribution = (
    df["delay_category"]
    .value_counts()
    .rename_axis("delay_category")
    .reset_index(name="count")
)

delay_distribution["percentage"] = (
    delay_distribution["count"] / delay_distribution["count"].sum() * 100
).round(2)

st.dataframe(delay_distribution, use_container_width=True)

fig = px.bar(
    delay_distribution,
    x="delay_category",
    y="count",
    text="percentage",
    title="Delay Category Distribution",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("3. Root-Cause Distribution")

root_cause_distribution = (
    df["root_cause_label"]
    .value_counts()
    .rename_axis("root_cause_label")
    .reset_index(name="count")
)

root_cause_distribution["percentage"] = (
    root_cause_distribution["count"] / root_cause_distribution["count"].sum() * 100
).round(2)

st.dataframe(root_cause_distribution, use_container_width=True)

fig = px.bar(
    root_cause_distribution,
    x="root_cause_label",
    y="count",
    text="percentage",
    title="Root-Cause Distribution",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("4. Severity Distribution")

severity_distribution = (
    df["severity_class"]
    .value_counts()
    .rename_axis("severity_class")
    .reset_index(name="count")
)

severity_distribution["percentage"] = (
    severity_distribution["count"] / severity_distribution["count"].sum() * 100
).round(2)

st.dataframe(severity_distribution, use_container_width=True)

fig = px.bar(
    severity_distribution,
    x="severity_class",
    y="count",
    text="percentage",
    title="Operational Exception Severity Distribution",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("5. Average OES by Root Cause")

oes_by_cause = (
    df.groupby("root_cause_label", as_index=False)
    .agg(
        records=("package_id", "count"),
        avg_delay_minutes=("delay_minutes", "mean"),
        avg_oes=("operational_exception_severity", "mean"),
        avg_distance_km=("distance_km", "mean"),
        avg_distance_adjusted_ratio=("distance_adjusted_execution_ratio", "mean"),
    )
    .sort_values("avg_oes", ascending=False)
)

numeric_cols = [
    "avg_delay_minutes",
    "avg_oes",
    "avg_distance_km",
    "avg_distance_adjusted_ratio",
]

for col in numeric_cols:
    if col in oes_by_cause.columns:
        oes_by_cause[col] = oes_by_cause[col].round(3)

st.dataframe(oes_by_cause, use_container_width=True)

fig = px.bar(
    oes_by_cause,
    x="root_cause_label",
    y="avg_oes",
    text="avg_oes",
    title="Average Operational Exception Severity by Root Cause",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("6. Top Exception Examples")

top_n = st.slider("Number of top exceptions to show", min_value=5, max_value=50, value=10)

top_exceptions = df.sort_values(
    "operational_exception_severity",
    ascending=False,
).head(top_n)

display_cols = [
    "package_id",
    "courier_id",
    "city",
    "zone_id",
    "distance_km",
    "delay_minutes",
    "distance_adjusted_execution_ratio",
    "root_cause_label",
    "severity_class",
    "operational_exception_severity",
]

available_cols = [col for col in display_cols if col in top_exceptions.columns]

st.dataframe(top_exceptions[available_cols], use_container_width=True)

st.download_button(
    label="Download top exception examples CSV",
    data=top_exceptions[available_cols].to_csv(index=False).encode("utf-8"),
    file_name="logidelay_top_exceptions.csv",
    mime="text/csv",
)

st.divider()

st.subheader("7. Download Full Processed Dataset")

st.download_button(
    label="Download full processed dataset CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="logidelay_processed_dataset.csv",
    mime="text/csv",
)