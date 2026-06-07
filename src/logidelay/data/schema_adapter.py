from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STANDARD_COLUMNS = [
    "package_id",
    "courier_id",
    "city",
    "zone_id",
    "task_type",
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
    "assigned_time",
    "accepted_time",
    "pickup_time",
    "completed_time",
    "promised_delivery_time",
    "courier_workload_2h",
]


REQUIRED_STANDARD_COLUMNS = [
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
]


OPTIONAL_STANDARD_COLUMNS = [
    "task_type",
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
]


@dataclass
class SchemaValidationResult:
    is_valid: bool
    missing_required_columns: list[str]
    available_optional_columns: list[str]
    missing_optional_columns: list[str]


def validate_standard_schema(df: pd.DataFrame) -> SchemaValidationResult:
    """
    Validate whether a dataframe follows the standard LogiDelay schema.
    """
    columns = set(df.columns)

    missing_required = [
        col for col in REQUIRED_STANDARD_COLUMNS if col not in columns
    ]

    available_optional = [
        col for col in OPTIONAL_STANDARD_COLUMNS if col in columns
    ]

    missing_optional = [
        col for col in OPTIONAL_STANDARD_COLUMNS if col not in columns
    ]

    return SchemaValidationResult(
        is_valid=len(missing_required) == 0,
        missing_required_columns=missing_required,
        available_optional_columns=available_optional,
        missing_optional_columns=missing_optional,
    )


def standardize_column_names(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Convert raw dataset column names into the standard LogiDelay schema.

    column_mapping format:
        {
            "raw_column_name": "standard_column_name"
        }

    Example:
        {
            "pkg_id": "package_id",
            "courier": "courier_id",
            "accept_time": "accepted_time"
        }
    """
    data = df.copy()
    data = data.rename(columns=column_mapping)
    return data


def add_missing_optional_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add safe defaults for optional columns if they are missing.
    """
    data = df.copy()

    if "task_type" not in data.columns:
        data["task_type"] = "delivery"

    if "city" not in data.columns:
        data["city"] = "Unknown"

    if "zone_id" not in data.columns:
        data["zone_id"] = "Unknown"

    return data


def standardize_logistics_dataset(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Main adapter function.

    It optionally renames raw columns and then validates the standardized dataset.
    """
    data = df.copy()

    if column_mapping:
        data = standardize_column_names(data, column_mapping)

    data = add_missing_optional_defaults(data)

    validation = validate_standard_schema(data)

    if not validation.is_valid:
        raise ValueError(
            "Dataset does not match required LogiDelay schema. "
            f"Missing required columns: {validation.missing_required_columns}"
        )

    return data