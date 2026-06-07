from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from logidelay.data.schema_adapter import standardize_logistics_dataset


@dataclass
class PublicDatasetConfig:
    """
    Configuration for mapping a public logistics dataset into LogiDelay schema.

    This adapter is intentionally dataset-flexible. The final LaDe/public dataset
    column names can be mapped here after inspecting the downloaded files.
    """

    input_path: Path
    output_path: Path
    column_mapping: dict[str, str]


DEFAULT_PUBLIC_COLUMN_MAPPING = {
    # Update these after inspecting the actual public dataset columns.
    # Raw column name              Standard LogiDelay column
    "raw_package_id": "package_id",
    "raw_courier_id": "courier_id",
    "raw_city": "city",
    "raw_zone_id": "zone_id",
    "raw_task_type": "task_type",
    "raw_origin_lat": "origin_lat",
    "raw_origin_lng": "origin_lng",
    "raw_destination_lat": "destination_lat",
    "raw_destination_lng": "destination_lng",
    "raw_assigned_time": "assigned_time",
    "raw_accepted_time": "accepted_time",
    "raw_pickup_time": "pickup_time",
    "raw_completed_time": "completed_time",
    "raw_promised_delivery_time": "promised_delivery_time",
    "raw_courier_workload_2h": "courier_workload_2h",
}


def read_public_dataset(input_path: str | Path) -> pd.DataFrame:
    """
    Read a public logistics dataset file.

    Supports CSV and Parquet for now.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file type: {path.suffix}. Use CSV or Parquet."
    )


def infer_basic_workload(
    df: pd.DataFrame,
    courier_col: str = "courier_id",
    time_col: str = "assigned_time",
    window: str = "2h",
) -> pd.DataFrame:
    """
    Estimate courier workload using number of tasks assigned within a rolling time window.

    This is useful when the raw dataset does not directly provide workload.
    """
    data = df.copy()

    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.sort_values([courier_col, time_col])

    workload_values = []

    for _, group in data.groupby(courier_col, sort=False):
        group = group.copy()
        group = group.set_index(time_col)

        workload = (
            group[courier_col]
            .rolling(window)
            .count()
            .fillna(1)
            .astype(int)
        )

        workload_values.append(workload.reset_index(drop=True))

    data["courier_workload_2h"] = pd.concat(
        workload_values,
        ignore_index=True,
    ).values

    return data


def prepare_public_dataset(
    input_path: str | Path,
    output_path: str | Path,
    column_mapping: dict[str, str],
    create_workload_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Convert a public logistics dataset into the standard LogiDelay schema.
    """
    raw_df = read_public_dataset(input_path)

    mapped_df = raw_df.rename(columns=column_mapping)

    if create_workload_if_missing and "courier_workload_2h" not in mapped_df.columns:
        mapped_df = infer_basic_workload(mapped_df)

    standardized = standardize_logistics_dataset(mapped_df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output, index=False)

    return standardized