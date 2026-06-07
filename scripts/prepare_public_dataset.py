from __future__ import annotations

from pathlib import Path

from logidelay.data.lade_adapter import (
    DEFAULT_PUBLIC_COLUMN_MAPPING,
    prepare_public_dataset,
)


def main() -> None:
    """
    Prepare a real public logistics dataset after updating the column mapping.

    Usage:
        1. Place raw dataset file in data/raw/
        2. Update DEFAULT_PUBLIC_COLUMN_MAPPING in lade_adapter.py
        3. Update input_path belows
        4. Run:
           uv run python scripts/prepare_public_dataset.py
    """

    input_path = Path("data/raw/public_logistics_raw.csv")
    output_path = Path("data/processed/public_logistics_standardized.csv")

    df = prepare_public_dataset(
        input_path=input_path,
        output_path=output_path,
        column_mapping=DEFAULT_PUBLIC_COLUMN_MAPPING,
    )

    print(f"Prepared public dataset with {len(df):,} records.")
    print(f"Saved standardized dataset to: {output_path}")


if __name__ == "__main__":
    main()