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

st.set_page_config(
    page_title="Dataset Integration",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Dataset Integration")

st.markdown(
    """
# Cainiao LaDe-P Dataset Integration

This page documents how the public Cainiao LaDe-P pickup logistics dataset is mapped
into the standardized event-log schema used by LogiDelay Copilot.

The goal is to support reproducible research for proactive service-window breach
prediction.
"""
)

STANDARDIZED_SAMPLE_PATH = (
    ROOT_DIR / "data" / "processed" / "lade_p_standardized_sample.csv"
)

PROACTIVE_SAMPLE_PATH = (
    ROOT_DIR / "data" / "processed" / "lade_p_proactive_breach_sample.csv"
)

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_predictions_with_urgency.csv"
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def convert_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data = df.copy()

    for col in cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


st.divider()

# ---------------------------------------------------------------------
# Dataset purpose
# ---------------------------------------------------------------------
st.subheader("Purpose of Dataset Integration")

st.markdown(
    """
Public logistics datasets often use different field names and formats. This project
maps the raw public dataset into a consistent logistics schema so that feature
engineering, model training, dashboarding, and paper evaluation remain reproducible.

For this research, the most important dataset is **LaDe-P**, because it contains pickup
service-window fields:

```text
accept_time
time_window_start
time_window_end
pickup_time
```

These fields allow us to define a proactive breach prediction task.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Dataset mapping
# ---------------------------------------------------------------------

st.subheader("LaDe-P Field Mapping")

mapping_df = pd.DataFrame(
    [
        {
            "Raw LaDe-P Field": "order_id",
            "Standard Field": "package_id",
            "Purpose": "Unique pickup task or package identifier",
        },
        {
            "Raw LaDe-P Field": "courier_id",
            "Standard Field": "courier_id",
            "Purpose": "Courier responsible for the pickup task",
        },
        {
            "Raw LaDe-P Field": "city",
            "Standard Field": "city",
            "Purpose": "City where the pickup task occurs",
        },
        {
            "Raw LaDe-P Field": "region_id",
            "Standard Field": "zone_id",
            "Purpose": "Operational zone or region identifier",
        },
        {
            "Raw LaDe-P Field": "accept_time",
            "Standard Field": "assigned_time / accepted_time",
            "Purpose": "Time when the courier accepted the task",
        },
        {
            "Raw LaDe-P Field": "time_window_start",
            "Standard Field": "service_window_start_time",
            "Purpose": "Beginning of the pickup service window",
        },
        {
            "Raw LaDe-P Field": "time_window_end",
            "Standard Field": "promised_delivery_time",
            "Purpose": "End of the pickup service window / promised deadline",
        },
        {
            "Raw LaDe-P Field": "pickup_time",
            "Standard Field": "pickup_time / completed_time",
            "Purpose": "Actual pickup completion time",
        },
        {
            "Raw LaDe-P Field": "accept_gps_lat",
            "Standard Field": "origin_lat",
            "Purpose": "Courier latitude at acceptance time",
        },
        {
            "Raw LaDe-P Field": "accept_gps_lng",
            "Standard Field": "origin_lng",
            "Purpose": "Courier longitude at acceptance time",
        },
        {
            "Raw LaDe-P Field": "lat",
            "Standard Field": "destination_lat",
            "Purpose": "Pickup location latitude",
        },
        {
            "Raw LaDe-P Field": "lng",
            "Standard Field": "destination_lng",
            "Purpose": "Pickup location longitude",
        },
    ]
)

st.dataframe(mapping_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Standard schema
# ---------------------------------------------------------------------

st.subheader("Standardized Event-Log Schema")

st.markdown(
    """
The standardized schema gives the project a consistent data structure across feature
engineering, model training, evaluation, and dashboard pages.
"""
)

st.code(
    """
package_id
courier_id
city
zone_id
assigned_time
accepted_time
service_window_start_time
promised_delivery_time
pickup_time
completed_time
origin_lat
origin_lng
destination_lat
destination_lng
courier_workload_2h
""",
    language="text",
)

st.divider()

# ---------------------------------------------------------------------
# Target label
# ---------------------------------------------------------------------

st.subheader("Service-Window Breach Label")

st.markdown(
    """
The target variable is created by comparing the actual pickup completion time against
the service-window end time.

A task is labeled as a breach when the pickup occurs after the promised service-window
deadline.
"""
)

st.code(
    """
service_window_breach = 1 if completed_time > promised_delivery_time else 0
""",
    language="text",
)

st.markdown(
    """
This label is used only as the supervised learning target and for evaluation. It is not
used as a model input feature.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Leakage-safe feature design
# ---------------------------------------------------------------------

st.subheader("Leakage-Safe Prediction Setup")

st.markdown(
    """
The proactive model is designed to operate at the time of task acceptance. Therefore,
features must be available before the pickup outcome is known.

The model excludes post-event fields such as:

* actual pickup completion outcome,
* delay minutes,
* delay category,
* root-cause label,
* severity score,
* operational exception severity,
* event-sequence abnormality scores.

This ensures that model evaluation reflects a realistic proactive dispatch setting.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Dataset preview and statistics
# ---------------------------------------------------------------------

st.subheader("Processed Dataset Preview")

if not STANDARDIZED_SAMPLE_PATH.exists():
    st.warning(
        """
The standardized LaDe-P sample was not found.

Run:

```bash
uv run python scripts/prepare_lade_p_sample.py
```
"""
    )
else:
    standardized_df = load_csv(STANDARDIZED_SAMPLE_PATH)

    st.markdown("### Standardized LaDe-P Sample")

    st.dataframe(standardized_df.head(20), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{len(standardized_df):,}")
    col2.metric("Columns", f"{len(standardized_df.columns):,}")

    if "city" in standardized_df.columns:
        col3.metric("Cities", f"{standardized_df['city'].nunique():,}")
    else:
        col3.metric("Cities", "N/A")

    if "courier_id" in standardized_df.columns:
        col4.metric("Couriers", f"{standardized_df['courier_id'].nunique():,}")
    else:
        col4.metric("Couriers", "N/A")

st.divider()

# ---------------------------------------------------------------------
# Proactive feature dataset
# ---------------------------------------------------------------------

st.subheader("Proactive Breach Feature Dataset")

if not PROACTIVE_SAMPLE_PATH.exists():
    st.warning(
        """
The proactive breach feature dataset was not found.

Run:

```bash
uv run python scripts/prepare_breach_dataset.py
```
"""
    )
else:
    proactive_df = load_csv(PROACTIVE_SAMPLE_PATH)

    proactive_df = convert_numeric_columns(
        proactive_df,
        [
            "service_window_breach",
            "time_to_window_end_minutes",
            "distance_km",
            "expected_travel_time_minutes",
            "feasibility_margin_minutes",
            "courier_workload_2h",
            "time_pressure_score",
            "distance_feasibility_pressure_score",
            "workload_pressure_score",
            "intervention_urgency_score",
        ],
    )

    st.dataframe(proactive_df.head(20), use_container_width=True)

    total_rows = len(proactive_df)
    total_breaches = int(proactive_df["service_window_breach"].sum())
    breach_rate = proactive_df["service_window_breach"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{total_rows:,}")
    col2.metric("Service-Window Breaches", f"{total_breaches:,}")
    col3.metric("Breach Rate", f"{breach_rate:.2%}")

    if "intervention_priority" in proactive_df.columns:
        high_count = int(
            proactive_df["intervention_priority"].isin(["High", "Critical"]).sum()
        )
        col4.metric("High/Critical Priority", f"{high_count:,}")
    else:
        col4.metric("High/Critical Priority", "N/A")

    st.markdown("### Breach Distribution")

    breach_counts = (
        proactive_df["service_window_breach"]
        .value_counts()
        .reset_index()
    )
    breach_counts.columns = ["service_window_breach", "count"]

    fig = px.bar(
        breach_counts,
        x="service_window_breach",
        y="count",
        title="Service-Window Breach Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Proactive Features")

    feature_cols = [
        "package_id",
        "city",
        "zone_id",
        "service_window_breach",
        "time_to_window_end_minutes",
        "distance_km",
        "expected_travel_time_minutes",
        "feasibility_margin_minutes",
        "courier_workload_2h",
        "time_pressure_score",
        "distance_feasibility_pressure_score",
        "workload_pressure_score",
        "intervention_urgency_score",
        "intervention_priority",
    ]

    available_feature_cols = [
        col for col in feature_cols if col in proactive_df.columns
    ]

    st.dataframe(
        proactive_df[available_feature_cols].head(50),
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------
# Prediction output integration
# ---------------------------------------------------------------------

st.subheader("Prediction Output Integration")

if not PREDICTIONS_PATH.exists():
    st.info(
        """
The final prediction output file was not found yet.

Run:

```bash
uv run python scripts/train_breach_model.py
```
"""
    )
else:
    predictions_df = load_csv(PREDICTIONS_PATH)

    predictions_df = convert_numeric_columns(
        predictions_df,
        [
            "service_window_breach",
            "predicted_breach_probability",
            "model_risk_rank_score",
            "intervention_urgency_score",
        ],
    )

    sorted_predictions = predictions_df.sort_values(
        "intervention_urgency_score",
        ascending=False,
    ).reset_index(drop=True)

    top50 = sorted_predictions.head(50)
    captured_breaches = int(top50["service_window_breach"].sum())
    total_breaches = int(sorted_predictions["service_window_breach"].sum())
    precision_50 = captured_breaches / 50
    recall_50 = captured_breaches / total_breaches if total_breaches else 0
    expected_random = (50 / len(sorted_predictions)) * total_breaches
    lift_50 = captured_breaches / expected_random if expected_random else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Prediction Rows", f"{len(predictions_df):,}")
    col2.metric("Top-50 Captured Breaches", captured_breaches)
    col3.metric("Urgency Precision@50", f"{precision_50:.2%}")
    col4.metric("Urgency Lift@50", f"{lift_50:.2f}x")

    st.success(
        f"""
The final prediction output integrates model probability, model risk rank, and
intervention urgency score. The urgency-ranked top 50 queue captures
**{captured_breaches} actual breaches**, with **Precision@50 = {precision_50:.2%}**,
**Recall@50 = {recall_50:.2%}**, and **Lift@50 = {lift_50:.2f}x**.
"""
    )

    prediction_cols = [
        "package_id",
        "city",
        "zone_id",
        "service_window_breach",
        "predicted_breach_probability",
        "model_risk_rank_score",
        "intervention_urgency_score",
        "intervention_priority",
    ]

    available_prediction_cols = [
        col for col in prediction_cols if col in sorted_predictions.columns
    ]

    st.markdown("### Final Prediction Output Preview")

    st.dataframe(
        sorted_predictions[available_prediction_cols].head(50),
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------
# Reproducibility commands
# ---------------------------------------------------------------------

st.subheader("Reproducibility Commands")

st.code(
    """
# Prepare LaDe-P standardized sample
uv run python scripts/prepare_lade_p_sample.py

# Build proactive breach feature dataset
uv run python scripts/prepare_breach_dataset.py

# Train calibrated breach prediction models
uv run python scripts/train_breach_model.py

# Run Streamlit app
uv run streamlit run app/home.py
""",
    language="bash",
)