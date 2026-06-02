from __future__ import annotations

from pathlib import Path

from logidelay.data.sample_generator import generate_sample_logistics_events
from logidelay.diagnosis.weak_labeler import add_root_cause_labels
from logidelay.features.event_features import add_event_features
from logidelay.severity.scoring import add_operational_exception_severity


def main() -> None:
    output_path = Path("data/sample/sample_logistics_events.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sample_logistics_events(n_rows=1000)
    df = add_event_features(df)
    df = add_operational_exception_severity(df)
    df = add_root_cause_labels(df)

    df.to_csv(output_path, index=False)
    print(f"Saved sample data to {output_path}")


if __name__ == "__main__":
    main()