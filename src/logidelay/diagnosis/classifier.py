from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "distance_km",
    "expected_execution_time_minutes",
    "distance_adjusted_execution_ratio",
    "acceptance_gap_minutes",
    "pickup_gap_minutes",
    "execution_time_minutes",
    "delay_minutes",
    "courier_workload_2h",
    "workload_pressure_score",
    "event_sequence_abnormality_score",
    "time_window_violation_score",
    "route_execution_instability_score",
    "operational_exception_severity",
]

CATEGORICAL_FEATURES = [
    "city",
    "zone_id",
    "task_type",
    "delay_category",
    "severity_class",
]

TARGET_COLUMN = "root_cause_label"


@dataclass
class ModelTrainingResult:
    accuracy: float
    classification_report_text: str
    confusion_matrix_df: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def build_baseline_classifier() -> Pipeline:
    """
    Build a simple interpretable baseline classifier.

    RandomForest is used because it handles nonlinear patterns well and works
    reliably for an MVP without heavy tuning.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector.
    """
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing_cols = sorted(required_columns - set(df.columns))

    if missing_cols:
        raise ValueError(f"Missing required columns for model training: {missing_cols}")

    data = df.copy()

    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].fillna("Unknown").astype(str)

    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET_COLUMN].astype(str)

    return x, y


def train_evaluate_classifier(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[Pipeline, ModelTrainingResult]:
    """
    Train and evaluate the baseline root-cause classifier.
    """
    x, y = prepare_model_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_baseline_classifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    confusion_matrix_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    result = ModelTrainingResult(
        accuracy=accuracy,
        classification_report_text=report,
        confusion_matrix_df=confusion_matrix_df,
        feature_columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
    )

    return model, result


def save_model(model: Pipeline, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_root_cause(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    x, _ = prepare_model_data(df)
    return pd.Series(model.predict(x), index=df.index, name="ml_root_cause_prediction")