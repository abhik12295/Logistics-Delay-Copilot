from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from logidelay.diagnosis.weak_labeler import add_root_cause_labels
from logidelay.features.event_features import add_event_features
from logidelay.severity.scoring import add_operational_exception_severity


LADE_P_SPLITS = [
    "pickup_jl",
    "pickup_cq",
    "pickup_yt",
    "pickup_sh",
    "pickup_hz",
]


def _parse_lade_time(value: object, year: int = 2024) -> pd.Timestamp:
    """
    Convert LaDe time values like '08-21 16:30:00' into a full timestamp.

    LaDe does not include year in the timestamp string, so we add a fixed
    reference year for consistent time-difference calculations.
    """
    if value is None or pd.isna(value):
        return pd.NaT

    return pd.to_datetime(f"{year}-{value}", format="%Y-%m-%d %H:%M:%S", errors="coerce")


def _first_valid(row: pd.Series, columns: list[str]) -> object:
    for col in columns:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            return value
    return None


def convert_lade_p_to_logidelay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert LaDe-P pickup records into the standard LogiDelay schema.
    """
    records = []

    for _, row in df.iterrows():
        accept_time = _parse_lade_time(row.get("accept_time"))
        pickup_time = _parse_lade_time(row.get("pickup_time"))
        time_window_end = _parse_lade_time(row.get("time_window_end"))

        origin_lat = _first_valid(row, ["accept_gps_lat", "pickup_gps_lat", "lat"])
        origin_lng = _first_valid(row, ["accept_gps_lng", "pickup_gps_lng", "lng"])

        destination_lat = _first_valid(row, ["lat", "pickup_gps_lat"])
        destination_lng = _first_valid(row, ["lng", "pickup_gps_lng"])

        records.append(
            {
                "package_id": f"P-{row.get('order_id')}",
                "courier_id": str(row.get("courier_id")),
                "city": row.get("city"),
                "zone_id": str(row.get("region_id")),
                "task_type": "pickup",
                "origin_lat": origin_lat,
                "origin_lng": origin_lng,
                "destination_lat": destination_lat,
                "destination_lng": destination_lng,
                "assigned_time": accept_time,
                "accepted_time": accept_time,
                "pickup_time": pickup_time,
                "completed_time": pickup_time,
                "promised_delivery_time": time_window_end,
                "aoi_id": row.get("aoi_id"),
                "aoi_type": row.get("aoi_type"),
                "ds": row.get("ds"),
            }
        )

    standardized = pd.DataFrame(records)

    standardized["assigned_time"] = pd.to_datetime(standardized["assigned_time"], errors="coerce")
    standardized["accepted_time"] = pd.to_datetime(standardized["accepted_time"], errors="coerce")
    standardized["pickup_time"] = pd.to_datetime(standardized["pickup_time"], errors="coerce")
    standardized["completed_time"] = pd.to_datetime(standardized["completed_time"], errors="coerce")
    standardized["promised_delivery_time"] = pd.to_datetime(
        standardized["promised_delivery_time"], errors="coerce"
    )

    return standardized


def add_workload_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate courier workload as number of tasks accepted within a rolling 2-hour window.
    """
    data = df.copy()
    data = data.sort_values(["courier_id", "accepted_time"])

    parts = []

    for _, group in data.groupby("courier_id", sort=False):
        group = group.copy()
        group = group.set_index("accepted_time")

        group["courier_workload_2h"] = (
            group["package_id"]
            .rolling("2h")
            .count()
            .fillna(1)
            .astype(int)
        )

        group = group.reset_index()
        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def load_lade_p_sample(rows_per_split: int = 2000) -> pd.DataFrame:
    """
    Load a manageable sample from each LaDe-P split.
    """
    frames = []

    for split in LADE_P_SPLITS:
        print(f"Loading split: {split}")

        dataset = load_dataset(
            "Cainiao-AI/LaDe-P",
            split=split,
            streaming=True,
        )

        rows = []
        for idx, row in enumerate(dataset):
            if idx >= rows_per_split:
                break
            rows.append(row)

        split_df = pd.DataFrame(rows)
        split_df["source_split"] = split
        frames.append(split_df)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    output_path = Path("data/processed/lade_p_standardized_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_lade_p_sample(rows_per_split=2000)
    standardized = convert_lade_p_to_logidelay(raw_df)
    standardized = add_workload_feature(standardized)

    # Drop records that cannot support required time calculations.
    standardized = standardized.dropna(
        subset=[
            "accepted_time",
            "completed_time",
            "promised_delivery_time",
            "origin_lat",
            "origin_lng",
            "destination_lat",
            "destination_lng",
        ]
    )

    processed = add_event_features(standardized)
    processed = add_operational_exception_severity(processed)
    processed = add_root_cause_labels(processed)

    processed.to_csv(output_path, index=False)

    print(f"Saved standardized LaDe-P sample to: {output_path}")
    print(f"Rows: {len(processed):,}")
    print("\nDelay category distribution:")
    print(processed["delay_category"].value_counts())
    print("\nRoot cause distribution:")
    print(processed["root_cause_label"].value_counts())
    print("\nSeverity distribution:")
    print(processed["severity_class"].value_counts())


if __name__ == "__main__":
    main()