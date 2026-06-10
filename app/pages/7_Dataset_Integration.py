# from __future__ import annotations

# from pathlib import Path
# import sys

# import pandas as pd
# import streamlit as st

# ROOT_DIR = Path(__file__).resolve().parents[2]
# SRC_DIR = ROOT_DIR / "src"

# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# from logidelay.data.schema_adapter import (
#     OPTIONAL_STANDARD_COLUMNS,
#     REQUIRED_STANDARD_COLUMNS,
#     STANDARD_COLUMNS,
#     validate_standard_schema,
# )

# st.set_page_config(page_title="Dataset Integration", page_icon="🧩", layout="wide")

# st.title("🧩 Dataset Integration")
# st.markdown(
#     """
# This page defines the standard LogiDelay input schema.

# The goal is to make the research framework country-neutral and dataset-flexible.
# Any public or enterprise logistics dataset can be used if it can be mapped into
# this standard event-log structure.
# """
# )

# st.subheader("Standard LogiDelay Schema")

# schema_df = pd.DataFrame(
#     {
#         "column_name": STANDARD_COLUMNS,
#         "required": [
#             "Yes" if col in REQUIRED_STANDARD_COLUMNS else "Optional"
#             for col in STANDARD_COLUMNS
#         ],
#         "description": [
#             "Unique package, shipment, order, or delivery identifier",
#             "Courier, carrier, driver, or delivery resource identifier",
#             "City or service area",
#             "Operational zone, region, or cluster",
#             "Task type such as delivery or pickup",
#             "Origin latitude",
#             "Origin longitude",
#             "Destination latitude",
#             "Destination longitude",
#             "Time when the task was assigned or created",
#             "Time when the courier/carrier accepted the task",
#             "Time when pickup or task execution started",
#             "Time when delivery/task was completed",
#             "Promised or expected delivery completion time",
#             "Number of courier tasks in a recent operating window",
#         ],
#     }
# )

# st.dataframe(schema_df, use_container_width=True)

# st.divider()

# st.subheader("Why This Schema Is Country-Neutral")

# st.markdown(
#     """
# The framework does not depend on one country's logistics system. It depends on
# universal event-log concepts:

# - task assignment
# - task acceptance
# - pickup or task start
# - completion
# - promised service time
# - workload
# - distance or location
# - event gaps

# These concepts are common across last-mile delivery, freight transportation,
# parcel delivery, and TMS-style logistics systems.
# """
# )

# st.divider()

# st.subheader("Validate a CSV Against the Standard Schema")

# uploaded_file = st.file_uploader(
#     "Upload a CSV to validate schema only",
#     type=["csv"],
# )

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     validation = validate_standard_schema(df)

#     if validation.is_valid:
#         st.success("CSV matches the required LogiDelay schema.")

#         col1, col2 = st.columns(2)
#         col1.markdown("#### Available optional columns")
#         col1.write(validation.available_optional_columns)

#         col2.markdown("#### Missing optional columns")
#         col2.write(validation.missing_optional_columns)

#     else:
#         st.error("CSV is missing required columns.")
#         st.markdown("#### Missing required columns")
#         st.code("\n".join(validation.missing_required_columns), language="text")

#         st.markdown("#### Required columns")
#         st.code("\n".join(REQUIRED_STANDARD_COLUMNS), language="text")

# st.divider()

# st.subheader("Example Raw-to-Standard Mapping")

# st.markdown(
#     """
# For a real public dataset, raw column names may not match our app directly.
# We will map them into the standard schema.

# Example:

# ```text
# raw package column       → package_id
# raw courier column       → courier_id
# raw accept timestamp     → accepted_time
# raw finish timestamp     → completed_time
# raw promised timestamp   → promised_delivery_time
# raw origin coordinates   → origin_lat, origin_lng
# raw destination coords    → destination_lat, destination_lng
# ```
# """
# )

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
This page defines the standard LogiDelay input schema and explains how public or
enterprise logistics datasets can be mapped into the same country-neutral event-log structure.

The purpose of this layer is to keep the research framework **dataset-flexible**.
Different logistics datasets may use different column names, but once they are mapped
into the standard schema, the same delay diagnosis, distance reasoning, severity scoring,
and explanation pipeline can be applied.
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
            "Origin latitude or task-start latitude",
            "Origin longitude or task-start longitude",
            "Destination latitude or service-location latitude",
            "Destination longitude or service-location longitude",
            "Time when the task was assigned or created",
            "Time when the courier/carrier accepted the task",
            "Time when pickup or task execution started",
            "Time when delivery, pickup, or service task was completed",
            "Promised or expected service completion time",
            "Number of courier/carrier tasks in a recent operating window",
        ],
    }
)

st.dataframe(schema_df, use_container_width=True)

st.divider()

st.subheader("Built-In Public Dataset: LaDe-P Sample")

st.markdown(
    """
The app includes support for a standardized **LaDe-P public benchmark sample**.

LaDe-P is used as the first real public-data validation case because it contains
courier pickup service-task records with:

- order ID
- city and region
- courier ID
- acceptance time
- pickup time
- service time window
- pickup/customer location coordinates
- courier GPS coordinates when available

In LogiDelay Copilot, LaDe-P is mapped into the standard schema and used to evaluate
**logistics service-task delay diagnosis**.

For this dataset, the interpretation is:

`pickup_time > time_window_end` means the pickup service task was delayed.

The pickup event is treated as the task completion event:

- `pickup_time` becomes `completed_time`
- `time_window_end` becomes `promised_delivery_time`

This allows the same pipeline to calculate service-window delay, distance-adjusted
execution behavior, workload pressure, root-cause weak labels, and Operational
Exception Severity.
"""
)

st.divider()

st.subheader("Why This Schema Is Country-Neutral")

st.markdown(
    """
The framework does not depend on one country's logistics system. It depends on
universal event-log concepts:

- task assignment or creation
- task acceptance
- pickup, delivery, or service start
- completion
- promised service time
- workload
- distance or location
- event gaps
- service-window deviation

These concepts are common across last-mile delivery, parcel pickup, freight transportation,
courier operations, and TMS-style logistics systems.

The public LaDe-P sample is used as one empirical benchmark, but the framework itself
is designed to be reusable for other countries and logistics networks when comparable
event-log data is available.
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

        st.markdown("#### Preview")
        st.dataframe(df.head(20), use_container_width=True)

    else:
        st.error("CSV is missing required columns.")

        st.markdown("#### Missing required columns")
        st.code("\n".join(validation.missing_required_columns), language="text")

        st.markdown("#### Required columns")
        st.code("\n".join(REQUIRED_STANDARD_COLUMNS), language="text")

        st.markdown("#### Optional columns")
        st.code("\n".join(OPTIONAL_STANDARD_COLUMNS), language="text")

st.divider()

st.subheader("Example Raw-to-Standard Mapping")

st.markdown(
    """
For a real public dataset, raw column names may not match the app directly.
The adapter layer maps raw dataset columns into the standard LogiDelay schema.
"""
)

st.markdown("#### Generic Mapping Example")

generic_mapping_df = pd.DataFrame(
    {
        "raw_dataset_field": [
            "raw package/order column",
            "raw courier/carrier column",
            "raw city column",
            "raw region/zone column",
            "raw accept timestamp",
            "raw task-start timestamp",
            "raw finish/completion time",
            "raw promised timestamp",
            "raw origin coordinates",
            "raw destination coordinates",
        ],
        "standard_logidelay_field": [
            "package_id",
            "courier_id",
            "city",
            "zone_id",
            "accepted_time",
            "pickup_time",
            "completed_time",
            "promised_delivery_time",
            "origin_lat, origin_lng",
            "destination_lat, destination_lng",
        ],
    }
)

st.dataframe(generic_mapping_df, use_container_width=True)

st.markdown("#### LaDe-P Mapping Used in This Project")

lade_mapping_df = pd.DataFrame(
    {
        "LaDe-P field": [
            "order_id",
            "courier_id",
            "city",
            "region_id",
            "accept_time",
            "accept_time",
            "pickup_time",
            "pickup_time",
            "time_window_end",
            "accept_gps_lat",
            "accept_gps_lng",
            "lat",
            "lng",
        ],
        "LogiDelay standard field": [
            "package_id",
            "courier_id",
            "city",
            "zone_id",
            "assigned_time",
            "accepted_time",
            "pickup_time",
            "completed_time",
            "promised_delivery_time",
            "origin_lat",
            "origin_lng",
            "destination_lat",
            "destination_lng",
        ],
        "reason": [
            "Order ID is the task/package identifier",
            "Courier ID identifies the delivery resource",
            "City is the service area",
            "Region ID is used as the operational zone",
            "Accept time is used as task assignment time",
            "Accept time is also the task acceptance time",
            "Pickup time is the service execution event",
            "Pickup time represents completion of the pickup task",
            "End of service window is the promised completion time",
            "Courier GPS latitude at acceptance is used as origin latitude",
            "Courier GPS longitude at acceptance is used as origin longitude",
            "Customer/pickup location latitude is destination latitude",
            "Customer/pickup location longitude is destination longitude",
        ],
    }
)

st.dataframe(lade_mapping_df, use_container_width=True)

st.markdown(
    """
Once mapped, the same pipeline can calculate distance, delay, root cause,
Operational Exception Severity, and planner-facing explanations.
"""
)

st.divider()

st.subheader("Distance-Aware Integration")

st.markdown(
    """
Distance is optional but preferred. If latitude and longitude columns are available,
the app calculates approximate distance using the Haversine formula.
"""
)

st.code(
    """
distance_km = haversine(origin_lat, origin_lng, destination_lat, destination_lng)

expected_execution_time_minutes = distance_km / expected_speed_kmph * 60

distance_adjusted_execution_ratio =
    actual_execution_time_minutes / expected_execution_time_minutes
""",
    language="text",
)

st.markdown(
    """
This helps the framework distinguish naturally long routes from abnormal execution delays.

For example, a 90-minute task may be normal for a long-distance route, but abnormal
for a short route. Distance-aware reasoning prevents the model from treating every
long execution time as the same type of exception.
"""
)