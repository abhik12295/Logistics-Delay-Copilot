from __future__ import annotations

from pathlib import Path

import pandas as pd

from logidelay.diagnosis.classifier import save_model, train_evaluate_classifier
from logidelay.diagnosis.weak_labeler import add_root_cause_labels
from logidelay.features.event_features import add_event_features
from logidelay.severity.scoring import add_operational_exception_severity


def main() -> None:
    input_path = Path("data/sample/sample_logistics_events.csv")
    model_path = Path("models/root_cause_baseline_model.joblib")
    metrics_path = Path("models/root_cause_baseline_metrics.txt")
    confusion_matrix_path = Path("models/root_cause_confusion_matrix.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            "Sample data not found. Run: uv run python scripts/prepare_sample_data.py"
        )

    df = pd.read_csv(input_path)

    required_cols = {
        "distance_km",
        "operational_exception_severity",
        "root_cause_label",
    }

    if not required_cols.issubset(df.columns):
        df = add_event_features(df)
        df = add_operational_exception_severity(df)
        df = add_root_cause_labels(df)

    model, result = train_evaluate_classifier(df)

    save_model(model, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        f"Accuracy: {result.accuracy:.4f}\n\n"
        f"{result.classification_report_text}",
        encoding="utf-8",
    )

    result.confusion_matrix_df.to_csv(confusion_matrix_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(result.classification_report_text)


if __name__ == "__main__":
    main()