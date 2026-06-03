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
def load_sample_data(root_dir: str) -> pd.DataFrame:
    sample_path = Path(root_dir) / "data" / "sample" / "sample_logistics_events.csv"

    if not sample_path.exists():
        raise FileNotFoundError(
            "Sample data not found. Run: uv run python scripts/prepare_sample_data.py"
        )

    return pd.read_csv(sample_path)


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


def load_data_from_sidebar(root_dir: str) -> pd.DataFrame:
    """
    Shared Streamlit sidebar loader.

    Allows user to select sample data or upload a CSV file.
    """
    st.sidebar.header("Data Source")

    data_source = st.sidebar.radio(
        "Choose data source",
        options=["Use sample data", "Upload CSV"],
    )

    if data_source == "Use sample data":
        df = load_sample_data(root_dir)
        st.sidebar.success("Using sample logistics event data.")
        return df

    uploaded_file = st.sidebar.file_uploader(
        "Upload logistics event CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        st.sidebar.info("Upload a CSV file or switch back to sample data.")
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

    processed_df = process_event_data(raw_df)
    st.sidebar.success("Uploaded CSV processed successfully.")
    return processed_df