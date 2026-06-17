from __future__ import annotations

from pathlib import Path

import pandas as pd

from logidelay.risk.breach_model import (
    save_breach_model_artifacts,
    train_and_evaluate_breach_models,
)
from logidelay.risk.breach_features import add_intervention_urgency_score


def main() -> None:
    input_path = Path("data/processed/lade_p_proactive_breach_sample.csv")
    output_prediction_path = Path(
        "data/processed/lade_p_breach_model_predictions_with_urgency.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "Proactive breach dataset not found. "
            "Run: uv run python scripts/prepare_breach_dataset.py"
        )

    df = pd.read_csv(input_path)

    best_result, all_results, train_df, test_df = train_and_evaluate_breach_models(df)

    save_breach_model_artifacts(best_result, all_results)

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": result.model_name,
                **result.metrics,
            }
            for result in all_results
        ]
    )

    print("\nModel comparison:")
    print(metrics_df.sort_values("average_precision_pr_auc", ascending=False))

    print("\nBest model:")
    print(best_result.model_name)

    print("\nBest model metrics:")
    for key, value in best_result.metrics.items():
        print(f"{key}: {value}")

    prediction_df = best_result.predictions.copy()

    # Add urgency score using predicted probability.
    prediction_df = add_intervention_urgency_score(prediction_df)

    prediction_df = prediction_df.sort_values(
        "intervention_urgency_score",
        ascending=False,
    )

    output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_prediction_path, index=False)

    print(f"\nSaved ranked prediction output to: {output_prediction_path}")

    print("\nTop 20 intervention queue:")
    display_cols = [
        "package_id",
        "city",
        "zone_id",
        "service_window_breach",
        "predicted_breach_probability",
        "intervention_urgency_score",
        "intervention_priority",
        "time_to_window_end_minutes",
        "distance_km",
        "expected_travel_time_minutes",
        "feasibility_margin_minutes",
        "courier_workload_2h",
    ]

    available_cols = [col for col in display_cols if col in prediction_df.columns]
    print(prediction_df[available_cols].head(20))


if __name__ == "__main__":
    main()