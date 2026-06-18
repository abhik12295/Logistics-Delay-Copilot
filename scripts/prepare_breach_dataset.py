from __future__ import annotations

from pathlib import Path

import pandas as pd

from logidelay.risk.breach_features import build_proactive_breach_dataset


def main() -> None:
    input_path = Path("data/processed/lade_p_standardized_sample.csv")
    output_path = Path("data/processed/lade_p_proactive_breach_sample.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            "Standardized LaDe-P sample not found. "
            "Run: uv run python scripts/prepare_lade_p_sample.py"
        )

    df = pd.read_csv(input_path)
    proactive_df = build_proactive_breach_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    proactive_df.to_csv(output_path, index=False)

    print(f"Saved proactive breach dataset to: {output_path}")
    print(f"Rows: {len(proactive_df):,}")

    print("\nService-window breach distribution:")
    print(proactive_df["service_window_breach"].value_counts())

    print("\nService-window breach percentage:")
    print(proactive_df["service_window_breach"].mean() * 100)

    print("\nIntervention priority distribution:")
    print(proactive_df["intervention_priority"].value_counts())

    preview_cols = [
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

    available_cols = [col for col in preview_cols if col in proactive_df.columns]

    print("\nPreview:")
    print(proactive_df[available_cols].head(10))


if __name__ == "__main__":
    main()