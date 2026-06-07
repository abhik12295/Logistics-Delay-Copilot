from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.data.schema_adapter import (
    OPTIONAL_STANDARD_COLUMNS,
    REQUIRED_STANDARD_COLUMNS,
    STANDARD_COLUMNS,
    validate_standard_schema,
)

st.set_page_config(page_title="Dataset Integration", page_icon="🧩", layout="wide")

st.title("🧩 Dataset Integration")
st.markdown(
    """
This page defines the standard LogiDelay input schema.

The goal is to make the research framework country-neutral and dataset-flexible.
Any public or enterprise logistics dataset can be used if it can be mapped into
this standard event-log structure.
"""
)

st.subheader("Standard LogiDelay Schema")

schema_df = pd.DataFrame(
    {
        "column_name": STANDARD_COLUMNS,
        "required": [
            "Yes" if col in REQUIRED_STANDARD_COLUMNS else "Optional"
            for col in STANDARD_COLUMNS
        ],
        "description": [
            "Unique package, shipment, order, or delivery identifier",
            "Courier, carrier, driver, or delivery resource identifier",
            "City or service area",
            "Operational zone, region, or cluster",
            "Task type such as delivery or pickup",
            "Origin latitude",
            "Origin longitude",
            "Destination latitude",
            "Destination longitude",
            "Time when the task was assigned or created",
            "Time when the courier/carrier accepted the task",
            "Time when pickup or task execution started",
            "Time when delivery/task was completed",
            "Promised or expected delivery completion time",
            "Number of courier tasks in a recent operating window",
        ],
    }
)

st.dataframe(schema_df, use_container_width=True)

st.divider()

st.subheader("Why This Schema Is Country-Neutral")

st.markdown(
    """
The framework does not depend on one country's logistics system. It depends on
universal event-log concepts:

- task assignment
- task acceptance
- pickup or task start
- completion
- promised service time
- workload
- distance or location
- event gaps

These concepts are common across last-mile delivery, freight transportation,
parcel delivery, and TMS-style logistics systems.
"""
)

st.divider()

st.subheader("Validate a CSV Against the Standard Schema")

uploaded_file = st.file_uploader(
    "Upload a CSV to validate schema only",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    validation = validate_standard_schema(df)

    if validation.is_valid:
        st.success("CSV matches the required LogiDelay schema.")

        col1, col2 = st.columns(2)
        col1.markdown("#### Available optional columns")
        col1.write(validation.available_optional_columns)

        col2.markdown("#### Missing optional columns")
        col2.write(validation.missing_optional_columns)

    else:
        st.error("CSV is missing required columns.")
        st.markdown("#### Missing required columns")
        st.code("\n".join(validation.missing_required_columns), language="text")

        st.markdown("#### Required columns")
        st.code("\n".join(REQUIRED_STANDARD_COLUMNS), language="text")

st.divider()

st.subheader("Example Raw-to-Standard Mapping")

st.markdown(
    """
For a real public dataset, raw column names may not match our app directly.
We will map them into the standard schema.

Example:

```text
raw package column       → package_id
raw courier column       → courier_id
raw accept timestamp     → accepted_time
raw finish timestamp     → completed_time
raw promised timestamp   → promised_delivery_time
raw origin coordinates   → origin_lat, origin_lng
raw destination coords    → destination_lat, destination_lng
```
"""
)