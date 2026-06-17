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

from logidelay.risk.intervention import (
    generate_dispatch_recommendation,
    generate_short_action,
)

st.set_page_config(
    page_title="Proactive Risk Copilot",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Proactive Risk Copilot")

st.markdown(
    """
This page represents the new research direction of the project:

**Proactive service-window breach prevention in last-mile logistics.**

Instead of only explaining delays after they happen, this page uses a machine learning
model to predict which pickup tasks are likely to miss their service window and ranks
them for dispatcher intervention.
"""
)

DATA_PATH = ROOT_DIR / "data" / "processed" / "lade_p_breach_model_predictions_with_urgency.csv"

if not DATA_PATH.exists():
    st.error(
        "Proactive breach prediction file was not found.\n\n"
        "Run these commands first:\n\n"
        "`uv run python scripts/prepare_breach_dataset.py`\n\n"
        "`uv run python scripts/train_breach_model.py`"
    )
    st.stop()


@st.cache_data
def load_prediction_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "service_window_breach",
        "predicted_breach_probability",
        "predicted_breach",
        "intervention_urgency_score",
        "time_to_window_end_minutes",
        "distance_km",
        "expected_travel_time_minutes",
        "feasibility_margin_minutes",
        "courier_workload_2h",
        "time_pressure_score",
        "distance_feasibility_pressure_score",
        "workload_pressure_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "accepted_time" in df.columns:
        df["accepted_time"] = pd.to_datetime(df["accepted_time"], errors="coerce")

    if "promised_delivery_time" in df.columns:
        df["promised_delivery_time"] = pd.to_datetime(
            df["promised_delivery_time"],
            errors="coerce",
        )

    if "intervention_priority" not in df.columns:
        df["intervention_priority"] = pd.cut(
            df["intervention_urgency_score"],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.00],
            labels=["Low", "Medium", "High", "Critical"],
        ).astype(str)

    df["recommended_action"] = df.apply(generate_short_action, axis=1)

    return df


df = load_prediction_data(DATA_PATH)

st.info(
    """
The current table is generated from the model test split. It includes actual breach
labels for evaluation, predicted breach probability, and intervention urgency scores.
"""
)

st.divider()

# Sidebar filters
st.sidebar.header("Risk Filters")

priority_options = ["All"] + sorted(df["intervention_priority"].dropna().unique().tolist())
selected_priority = st.sidebar.selectbox("Intervention priority", priority_options)

city_options = ["All"] + sorted(df["city"].dropna().astype(str).unique().tolist())
selected_city = st.sidebar.selectbox("City", city_options)

risk_threshold = st.sidebar.slider(
    "Minimum predicted breach probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

filtered_df = df.copy()

if selected_priority != "All":
    filtered_df = filtered_df[filtered_df["intervention_priority"] == selected_priority]

if selected_city != "All":
    filtered_df = filtered_df[filtered_df["city"].astype(str) == selected_city]

filtered_df = filtered_df[
    filtered_df["predicted_breach_probability"] >= risk_threshold
].copy()

filtered_df = filtered_df.sort_values(
    "intervention_urgency_score",
    ascending=False,
)

# KPIs
st.subheader("Risk Summary")

total_tasks = len(filtered_df)
actual_breaches = int(filtered_df["service_window_breach"].sum())
breach_rate = filtered_df["service_window_breach"].mean() if total_tasks else 0
avg_predicted_risk = (
    filtered_df["predicted_breach_probability"].mean() if total_tasks else 0
)
high_priority_count = int(
    filtered_df["intervention_priority"].isin(["High", "Critical"]).sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tasks in View", f"{total_tasks:,}")
col2.metric("Actual Breaches", f"{actual_breaches:,}", f"{breach_rate:.2%}")
col3.metric("Avg Predicted Risk", f"{avg_predicted_risk:.2%}")
col4.metric("High/Critical Priority", f"{high_priority_count:,}")

st.divider()

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Predicted Breach Probability Distribution")
    fig = px.histogram(
        filtered_df,
        x="predicted_breach_probability",
        nbins=30,
        title="Predicted Service-Window Breach Probability",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Intervention Priority Distribution")
    priority_counts = (
        filtered_df["intervention_priority"]
        .value_counts()
        .reset_index()
    )
    priority_counts.columns = ["intervention_priority", "count"]

    fig = px.bar(
        priority_counts,
        x="intervention_priority",
        y="count",
        title="Intervention Priority",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Intervention queue
st.subheader("Ranked Intervention Queue")

queue_cols = [
    "package_id",
    "city",
    "zone_id",
    "service_window_breach",
    "predicted_breach_probability",
    "intervention_urgency_score",
    "intervention_priority",
    "time_to_window_end_minutes",
    "distance_km",
    "expected_travel_time_minutes",
    "feasibility_margin_minutes",
    "courier_workload_2h",
    "recommended_action",
]

available_queue_cols = [col for col in queue_cols if col in filtered_df.columns]

queue_df = filtered_df[available_queue_cols].copy()
queue_df.insert(0, "rank", range(1, len(queue_df) + 1))

st.dataframe(
    queue_df.head(100),
    use_container_width=True,
)

st.caption(
    """
This table is designed like a dispatcher queue. The highest urgency tasks appear first.
The actual breach label is shown only because this is an evaluation dataset.
In a live system, the actual outcome would not be known at prediction time.
"""
)

st.divider()

# Task explorer
st.subheader("Task Risk Explorer")

if len(filtered_df) == 0:
    st.warning("No tasks match the selected filters.")
    st.stop()

selected_package = st.selectbox(
    "Select a package/task",
    filtered_df["package_id"].astype(str).tolist(),
)

record = filtered_df[filtered_df["package_id"].astype(str) == selected_package].iloc[0]

explorer_col1, explorer_col2, explorer_col3, explorer_col4 = st.columns(4)

explorer_col1.metric(
    "Predicted Breach Risk",
    f"{record['predicted_breach_probability']:.2%}",
)
explorer_col2.metric(
    "Urgency Score",
    f"{record['intervention_urgency_score']:.2f}",
)
explorer_col3.metric(
    "Priority",
    str(record["intervention_priority"]),
)
explorer_col4.metric(
    "Actual Breach",
    "Yes" if int(record["service_window_breach"]) == 1 else "No",
)

st.markdown("### Dispatcher Recommendation")
st.success(generate_dispatch_recommendation(record))

st.markdown("### Risk Evidence")

evidence_cols = [
    "package_id",
    "courier_id",
    "city",
    "zone_id",
    "accepted_time",
    "promised_delivery_time",
    "time_to_window_end_minutes",
    "distance_km",
    "expected_travel_time_minutes",
    "feasibility_margin_minutes",
    "courier_workload_2h",
    "time_pressure_score",
    "distance_feasibility_pressure_score",
    "workload_pressure_score",
    "predicted_breach_probability",
    "intervention_urgency_score",
    "intervention_priority",
    "service_window_breach",
]

evidence = {
    col: record[col]
    for col in evidence_cols
    if col in record.index
}

st.json(evidence)

st.divider()

st.subheader("Research Interpretation")

st.markdown(
    """
This page supports the new research framing:

> Can AI predict service-window breach risk before task completion and help dispatchers
> prioritize interventions?

The key research output is no longer only a delay explanation. The system now creates:

- a supervised breach label,
- proactive non-leakage features,
- predicted breach probability,
- intervention urgency score,
- ranked dispatch queue,
- dispatcher-facing recommendation.

This makes the project a stronger AI/ML logistics decision-support system.
"""
)