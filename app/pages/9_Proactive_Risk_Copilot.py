from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from logidelay.copilot.local_llm_engine import generate_dispatcher_recommendation

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Proactive Risk Copilot",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Proactive Risk Copilot")

st.markdown(
    """
# Dispatch Intervention Queue

This page demonstrates the operational copilot for **proactive service-window breach
prevention**.

The copilot ranks pickup tasks by intervention urgency and generates a dispatcher-ready
recommendation using structured model evidence.

The goal is to help answer:

> **Which pickup tasks should dispatch review first, and what should be done before the service-window breach occurs?**
"""
)

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_predictions_with_urgency.csv"
)

if not PREDICTIONS_PATH.exists():
    st.warning(
        """
The proactive breach prediction file was not found.

Run these commands first:

```bash
uv run python scripts/prepare_breach_dataset.py
uv run python scripts/train_breach_model.py
```

Then refresh this page.
"""
    )
    st.stop()


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "service_window_breach",
        "predicted_breach_probability",
        "predicted_breach",
        "model_risk_rank_score",
        "intervention_urgency_score",
        "time_to_window_start_minutes",
        "time_to_window_end_minutes",
        "service_window_length_minutes",
        "distance_km",
        "expected_travel_time_minutes",
        "feasibility_margin_minutes",
        "courier_workload_2h",
        "time_pressure_score",
        "distance_feasibility_pressure_score",
        "workload_pressure_score",
        "historical_courier_breach_rate",
        "historical_zone_breach_rate",
        "historical_city_breach_rate",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    datetime_cols = [
        "accepted_time",
        "assigned_time",
        "service_window_start_time",
        "promised_delivery_time",
        "pickup_time",
        "completed_time",
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "intervention_priority" not in df.columns:
        df["intervention_priority"] = pd.cut(
            df["intervention_urgency_score"],
            bins=[-0.01, 0.35, 0.60, 0.80, 1.00],
            labels=["Low", "Medium", "High", "Critical"],
        ).astype(str)

    return df


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_probability(value: Any) -> str:
    return f"{safe_float(value):.2%}"


def format_score(value: Any) -> str:
    return f"{safe_float(value):.3f}"


def classify_risk(row: pd.Series) -> str:
    priority = str(row.get("intervention_priority", "")).strip()

    if priority in {"Critical", "High", "Medium", "Low"}:
        return priority

    urgency = safe_float(row.get("intervention_urgency_score"))

    if urgency >= 0.80:
        return "Critical"
    if urgency >= 0.60:
        return "High"
    if urgency >= 0.35:
        return "Medium"
    return "Low"


def identify_primary_risk_factor(row: pd.Series) -> str:
    risk_factors = {
        "Time pressure": safe_float(row.get("time_pressure_score")),
        "Distance feasibility": safe_float(row.get("distance_feasibility_pressure_score")),
        "Courier workload": safe_float(row.get("workload_pressure_score")),
        "Model risk rank": safe_float(row.get("model_risk_rank_score")),
    }

    primary_factor = max(risk_factors, key=risk_factors.get)
    return primary_factor


def generate_dispatch_recommendation(row: pd.Series) -> dict[str, str]:
    risk_level = classify_risk(row)
    primary_factor = identify_primary_risk_factor(row)

    probability = safe_float(row.get("predicted_breach_probability"))
    urgency = safe_float(row.get("intervention_urgency_score"))
    model_rank = safe_float(row.get("model_risk_rank_score"))
    time_remaining = safe_float(row.get("time_to_window_end_minutes"))
    margin = safe_float(row.get("feasibility_margin_minutes"))
    workload = safe_float(row.get("courier_workload_2h"))
    distance = safe_float(row.get("distance_km"))

    if risk_level == "Critical":
        action = (
            "Immediate dispatcher review is recommended. Confirm courier availability, "
            "check nearby backup capacity, and consider task reassignment if the courier "
            "cannot complete the pickup within the service window."
        )
    elif risk_level == "High":
        action = (
            "Prioritize this task for dispatcher review. Contact the courier if needed, "
            "monitor route progress closely, and prepare reassignment if feasibility worsens."
        )
    elif risk_level == "Medium":
        action = (
            "Monitor this task. It does not require immediate escalation, but dispatch should "
            "review it if workload increases or the service-window margin becomes tighter."
        )
    else:
        action = (
            "No immediate intervention is required. Continue normal monitoring unless new "
            "events increase the predicted breach risk."
        )

    evidence_items = [
        f"Predicted breach probability: {probability:.2%}",
        f"Model risk percentile/rank score: {model_rank:.3f}",
        f"Intervention urgency score: {urgency:.3f}",
        f"Primary risk factor: {primary_factor}",
        f"Time remaining to service-window end: {time_remaining:.1f} minutes",
        f"Feasibility margin: {margin:.1f} minutes",
        f"Distance estimate: {distance:.2f} km",
        f"Courier workload in 2-hour window: {workload:.0f} tasks",
    ]

    if risk_level in {"Critical", "High"}:
        dispatcher_note = (
            f"This task is ranked as {risk_level.lower()} priority because the model risk "
            f"and operational pressure signals indicate elevated breach potential."
        )
    else:
        dispatcher_note = (
            f"This task is ranked as {risk_level.lower()} priority. Current evidence does "
            f"not indicate immediate escalation, but the task remains available for monitoring."
        )

    return {
        "risk_level": risk_level,
        "primary_factor": primary_factor,
        "dispatcher_note": dispatcher_note,
        "recommended_action": action,
        "evidence": "\n".join(f"- {item}" for item in evidence_items),
    }


def build_topk_summary(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "service_window_breach",
    k: int = 50,
) -> dict[str, float]:
    sorted_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    topk = sorted_df.head(k)

    total_breaches = int(sorted_df[label_col].sum())
    captured_breaches = int(topk[label_col].sum())

    precision_k = captured_breaches / k if k else 0
    recall_k = captured_breaches / total_breaches if total_breaches else 0

    random_expected = (k / len(sorted_df)) * total_breaches if len(sorted_df) else 0
    lift = captured_breaches / random_expected if random_expected else 0

    return {
        "captured_breaches": captured_breaches,
        "precision_k": precision_k,
        "recall_k": recall_k,
        "lift": lift,
    }


predictions_df = load_predictions(PREDICTIONS_PATH)

predictions_df = predictions_df.sort_values(
    "intervention_urgency_score",
    ascending=False,
).reset_index(drop=True)

predictions_df["dispatch_rank"] = predictions_df.index + 1

st.divider()

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------

st.sidebar.header("Queue Filters")

available_cities = sorted(predictions_df["city"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect(
    "City",
    options=available_cities,
    default=available_cities,
)

priority_order = ["Critical", "High", "Medium", "Low"]
available_priorities = [
    priority
    for priority in priority_order
    if priority in predictions_df["intervention_priority"].dropna().unique().tolist()
]

selected_priorities = st.sidebar.multiselect(
    "Intervention priority",
    options=available_priorities,
    default=available_priorities,
)

max_records = st.sidebar.slider(
    "Records to show in queue",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
)

show_actual_labels = st.sidebar.checkbox(
    "Show actual breach labels for research evaluation",
    value=True,
)

filtered_df = predictions_df.copy()

if selected_cities:
    filtered_df = filtered_df[filtered_df["city"].isin(selected_cities)]

if selected_priorities:
    filtered_df = filtered_df[
        filtered_df["intervention_priority"].isin(selected_priorities)
    ]

filtered_df = filtered_df.sort_values(
    "intervention_urgency_score",
    ascending=False,
).reset_index(drop=True)

filtered_df["dispatch_rank"] = filtered_df.index + 1

# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------

st.subheader("Dispatch Queue Summary")

top50_summary = build_topk_summary(
    predictions_df,
    score_col="intervention_urgency_score",
    k=50,
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tasks in View", f"{len(filtered_df):,}")
col2.metric(
    "Avg Predicted Risk",
    format_probability(filtered_df["predicted_breach_probability"].mean()),
)
col3.metric("Top-50 Breaches", int(top50_summary["captured_breaches"]))
col4.metric("Top-50 Lift", f"{top50_summary['lift']:.2f}x")

col5, col6, col7, col8 = st.columns(4)

high_priority_count = int(
    filtered_df["intervention_priority"].isin(["Critical", "High"]).sum()
)

actual_breaches_in_view = int(filtered_df["service_window_breach"].sum())

col5.metric("High/Critical Tasks", f"{high_priority_count:,}")
col6.metric("Actual Breaches in View", f"{actual_breaches_in_view:,}")
col7.metric("Urgency Precision@50", f"{top50_summary['precision_k']:.2%}")
col8.metric("Urgency Recall@50", f"{top50_summary['recall_k']:.2%}")

st.success(
    f"""
The intervention urgency queue captures **{int(top50_summary['captured_breaches'])}**
actual breaches in the top 50 ranked tasks, with **Precision@50 =
{top50_summary['precision_k']:.2%}**, **Recall@50 = {top50_summary['recall_k']:.2%}**,
and **Lift@50 = {top50_summary['lift']:.2f}x** over random selection.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Visual overview
# ---------------------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
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
        title="Tasks by Intervention Priority",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Predicted Breach Probability")

    fig = px.histogram(
        filtered_df,
        x="predicted_breach_probability",
        nbins=30,
        title="Predicted Breach Probability Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Dispatch queue
# ---------------------------------------------------------------------

st.subheader("Ranked Dispatch Intervention Queue")

queue_cols = [
    "dispatch_rank",
    "package_id",
    "city",
    "zone_id",
    "intervention_priority",
    "predicted_breach_probability",
    "model_risk_rank_score",
    "intervention_urgency_score",
    "time_to_window_end_minutes",
    "distance_km",
    "feasibility_margin_minutes",
    "courier_workload_2h",
]

if show_actual_labels:
    queue_cols.insert(5, "service_window_breach")

available_queue_cols = [
    col for col in queue_cols if col in filtered_df.columns
]

queue_df = filtered_df.head(max_records)[available_queue_cols]

st.dataframe(queue_df, use_container_width=True)

st.caption(
    """
The actual breach label is displayed only for research validation. In a real deployment,
dispatchers would only see model risk, urgency ranking, evidence, and recommendation.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Task-level copilot recommendation
# ---------------------------------------------------------------------

st.subheader("Task-Level Dispatcher Recommendation")

if filtered_df.empty:
    st.warning("No tasks match the selected filters.")
    st.stop()

task_options = filtered_df["package_id"].astype(str).tolist()

selected_package_id = st.selectbox(
    "Select a task",
    options=task_options,
    index=0,
)

selected_row = filtered_df[
    filtered_df["package_id"].astype(str) == selected_package_id
].iloc[0]

# recommendation = generate_dispatch_recommendation(selected_row)
task_evidence_for_llm = {
    col: selected_row.get(col)
    for col in [
        "package_id",
        "city",
        "zone_id",
        "intervention_priority",
        "predicted_breach_probability",
        "model_risk_rank_score",
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
}

st.markdown("### Recommendation Engine")

use_ollama = st.checkbox(
    "Use local Ollama GenAI recommendation",
    value=False,
)

ollama_model_name = st.text_input(
    "Ollama model name",
    value="llama3.2:3b",
)

llm_recommendation = generate_dispatcher_recommendation(
    task_evidence=task_evidence_for_llm,
    use_ollama=use_ollama,
    model_name=ollama_model_name,
)

recommendation = generate_dispatch_recommendation(selected_row)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric("Risk Level", recommendation["risk_level"])
summary_col2.metric(
    "Predicted Risk",
    format_probability(selected_row.get("predicted_breach_probability")),
)
summary_col3.metric(
    "Urgency Score",
    format_score(selected_row.get("intervention_urgency_score")),
)
summary_col4.metric(
    "Dispatch Rank",
    int(selected_row.get("dispatch_rank")),
)

st.markdown("### Copilot Summary")

st.markdown("### GenAI / Grounded Recommendation")

source_label = (
    "Local Ollama GenAI"
    if llm_recommendation.source == "ollama"
    else "Rule-based fallback"
)

st.caption(f"Recommendation source: {source_label}")

st.info(llm_recommendation.risk_summary)

st.markdown("### Evidence Summary")
st.write(llm_recommendation.evidence_summary)

st.markdown("### Recommended Dispatch Action")
st.success(llm_recommendation.recommended_action)

st.markdown("### Dispatcher Note")
st.write(llm_recommendation.dispatcher_note)

st.markdown("### Recommendation Confidence")
st.write(llm_recommendation.confidence)

st.markdown("### Rule-Based Baseline Recommendation")
st.info(recommendation["dispatcher_note"])

st.markdown("### Rule-Based Baseline Action")
st.success(recommendation["recommended_action"])


st.markdown("### Evidence Used by Copilot")

st.code(recommendation["evidence"], language="text")

st.markdown("### Structured Task Evidence")

evidence_cols = [
    "package_id",
    "city",
    "zone_id",
    "service_window_breach",
    "predicted_breach_probability",
    "model_risk_rank_score",
    "intervention_urgency_score",
    "intervention_priority",
    "time_to_window_end_minutes",
    "time_pressure_score",
    "distance_km",
    "expected_travel_time_minutes",
    "feasibility_margin_minutes",
    "distance_feasibility_pressure_score",
    "courier_workload_2h",
    "workload_pressure_score",
    "historical_courier_breach_rate",
    "historical_zone_breach_rate",
    "historical_city_breach_rate",
]

available_evidence_cols = [
    col for col in evidence_cols if col in selected_row.index
]

task_evidence_df = pd.DataFrame(
    [selected_row[available_evidence_cols].to_dict()]
)

st.dataframe(task_evidence_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Research explanation
# ---------------------------------------------------------------------

st.subheader("Research Interpretation")

st.markdown(
    """
This page represents the operational side of the proposed framework.

The breach model estimates service-window risk. The intervention urgency score then
combines model risk ranking with time pressure, feasibility pressure, and workload
pressure to produce a dispatcher-facing queue.

The recommendation section is currently rule-grounded and evidence-grounded. In the
next stage, the same structured evidence can be passed to a local LLM, such as Ollama,
to generate more natural dispatcher recommendations while preserving factual grounding.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------

st.subheader("Download Dispatch Queue")

download_df = filtered_df.head(max_records).copy()

st.download_button(
    label="Download filtered dispatch queue",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="proactive_dispatch_queue.csv",
    mime="text/csv",
)