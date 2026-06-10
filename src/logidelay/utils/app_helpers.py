from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from logidelay.diagnosis.weak_labeler import add_root_cause_labels
from logidelay.features.event_features import add_event_features
from logidelay.severity.scoring import add_operational_exception_severity


REQUIRED_RAW_COLUMNS = {
    "package_id",
    "courier_id",
    "city",
    "zone_id",
    "assigned_time",
    "accepted_time",
    "pickup_time",
    "completed_time",
    "promised_delivery_time",
    "courier_workload_2h",
}


@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    return pd.read_csv(csv_path)


@st.cache_data
def load_sample_data(root_dir: str) -> pd.DataFrame:
    sample_path = Path(root_dir) / "data" / "sample" / "sample_logistics_events.csv"
    return load_csv_data(str(sample_path))


@st.cache_data
def load_lade_p_data(root_dir: str) -> pd.DataFrame:
    lade_path = Path(root_dir) / "data" / "processed" / "lade_p_standardized_sample.csv"
    return load_csv_data(str(lade_path))


def validate_input_data(df: pd.DataFrame) -> list[str]:
    """
    Validate whether uploaded data contains the minimum required columns.
    """
    missing_cols = sorted(REQUIRED_RAW_COLUMNS - set(df.columns))
    return missing_cols


def process_event_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full LogiDelay pipeline on raw event data.
    """
    data = add_event_features(df)
    data = add_operational_exception_severity(data)
    data = add_root_cause_labels(data)
    return data


def ensure_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a dataset has all derived LogiDelay fields.

    If derived columns are missing, run the processing pipeline.
    """
    required_processed_cols = {
        "delay_minutes",
        "delay_category",
        "is_delayed",
        "distance_km",
        "distance_adjusted_execution_ratio",
        "route_execution_instability_score",
        "operational_exception_severity",
        "severity_class",
        "root_cause_label",
    }

    if required_processed_cols.issubset(df.columns):
        return df

    return process_event_data(df)


def load_data_from_sidebar(root_dir: str) -> pd.DataFrame:
    """
    Shared Streamlit sidebar loader.

    Allows user to select synthetic sample data, LaDe-P public sample, or upload CSV.
    """
    st.sidebar.header("Data Source")

    data_source = st.sidebar.radio(
        "Choose data source",
        options=[
            "Use synthetic sample data",
            "Use LaDe-P public sample",
            "Upload CSV",
        ],
    )

    if data_source == "Use synthetic sample data":
        try:
            df = load_sample_data(root_dir)
        except FileNotFoundError:
            st.sidebar.error("Synthetic sample data not found.")
            st.error("Run this command first:")
            st.code("uv run python scripts/prepare_sample_data.py", language="bash")
            st.stop()

        st.sidebar.success("Using synthetic sample logistics event data.")
        return ensure_processed_data(df)

    if data_source == "Use LaDe-P public sample":
        try:
            df = load_lade_p_data(root_dir)
        except FileNotFoundError:
            st.sidebar.error("LaDe-P public sample not found.")
            st.error("Run this command first:")
            st.code("uv run python scripts/prepare_lade_p_sample.py", language="bash")
            st.stop()

        st.sidebar.success("Using LaDe-P public benchmark sample.")
        st.sidebar.caption(
            "This sample represents pickup service-task records from the public LaDe-P dataset."
        )
        return ensure_processed_data(df)

    uploaded_file = st.sidebar.file_uploader(
        "Upload logistics event CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        st.sidebar.info("Upload a CSV file or select a built-in dataset.")
        st.stop()

    raw_df = pd.read_csv(uploaded_file)

    missing_cols = validate_input_data(raw_df)
    if missing_cols:
        st.sidebar.error("Uploaded file is missing required columns.")
        st.error("Missing required columns:")
        st.code("\n".join(missing_cols), language="text")
        st.markdown("### Required minimum columns")
        st.code("\n".join(sorted(REQUIRED_RAW_COLUMNS)), language="text")
        st.stop()

    processed_df = ensure_processed_data(raw_df)
    st.sidebar.success("Uploaded CSV processed successfully.")
    return processed_df