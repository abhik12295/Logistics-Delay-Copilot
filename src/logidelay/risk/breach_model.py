from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "service_window_breach"

# These columns must not be used for proactive prediction because they are only
# known after pickup/service completion.
LEAKAGE_COLUMNS = {
    "pickup_time",
    "completed_time",
    "delay_minutes",
    "delay_category",
    "root_cause_label",
    "severity_class",
    "operational_exception_severity",
    "delivery_duration_minutes",
    "execution_time_minutes",
    "distance_adjusted_execution_ratio",
    "route_execution_instability_score",
    "time_window_violation_score",
    "event_sequence_abnormality_score",
}

NUMERIC_FEATURE_CANDIDATES = [
    "time_to_window_start_minutes",
    "time_to_window_end_minutes",
    "service_window_length_minutes",
    "distance_km",
    "expected_travel_time_minutes",
    "feasibility_margin_minutes",
    "courier_workload_2h",
    "time_pressure_score",
    "workload_pressure_score",
    "distance_feasibility_pressure_score",
    "historical_courier_breach_rate",
    "historical_zone_breach_rate",
    "historical_city_breach_rate",
    "accept_hour",
    "accept_dayofweek",
    "accept_month",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "city",
    "zone_id",
    "task_type",
    "aoi_type",
]


@dataclass
class BreachModelResult:
    model_name: str
    pipeline: Pipeline
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def get_available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Select only proactive non-leakage features available in the dataframe.
    """
    numeric_features = [
        col
        for col in NUMERIC_FEATURE_CANDIDATES
        if col in df.columns and col not in LEAKAGE_COLUMNS
    ]

    categorical_features = [
        col
        for col in CATEGORICAL_FEATURE_CANDIDATES
        if col in df.columns and col not in LEAKAGE_COLUMNS
    ]

    return numeric_features, categorical_features


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split records chronologically by accepted_time.

    This is more realistic than random split because the model trains on earlier
    tasks and tests on later tasks.
    """
    data = df.copy()

    if "accepted_time" in data.columns:
        data["accepted_time"] = pd.to_datetime(data["accepted_time"], errors="coerce")
        data = data.sort_values("accepted_time")

    split_index = int(len(data) * (1 - test_size))

    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()

    return train_df, test_df


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_candidate_models(
    preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    """
    Build candidate ML models.

    We use class_weight='balanced' because service-window breaches are rare.
    """
    return {
        "logistic_regression_balanced": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Precision among the top-k highest-risk tasks.
    """
    if len(y_true) == 0:
        return 0.0

    k = min(k, len(y_true))
    top_indices = np.argsort(y_score)[::-1][:k]

    return float(np.mean(y_true[top_indices]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Recall captured among the top-k highest-risk tasks.
    """
    positives = np.sum(y_true)

    if positives == 0:
        return 0.0

    k = min(k, len(y_true))
    top_indices = np.argsort(y_score)[::-1][:k]
    captured_positives = np.sum(y_true[top_indices])

    return float(captured_positives / positives)


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Evaluate predicted breach probabilities.

    For imbalanced logistics data, PR-AUC and top-k metrics are more important
    than accuracy.
    """
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "positive_rate_actual": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
        "precision_at_25": precision_at_k(y_true, y_score, 25),
        "recall_at_25": recall_at_k(y_true, y_score, 25),
        "precision_at_50": precision_at_k(y_true, y_score, 50),
        "recall_at_50": recall_at_k(y_true, y_score, 50),
        "precision_at_100": precision_at_k(y_true, y_score, 100),
        "recall_at_100": recall_at_k(y_true, y_score, 100),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    return metrics


def train_and_evaluate_breach_models(
    df: pd.DataFrame,
) -> tuple[BreachModelResult, list[BreachModelResult], pd.DataFrame, pd.DataFrame]:
    """
    Train candidate breach prediction models and return the best model.

    Best model is selected using PR-AUC because service-window breach is rare.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COLUMN}'. "
            "Run scripts/prepare_breach_dataset.py first."
        )

    data = df.copy()
    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    ).fillna(0).astype(int)

    train_df, test_df = temporal_train_test_split(data)

    numeric_features, categorical_features = get_available_features(data)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable proactive features found for model training.")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    candidate_models = build_candidate_models(preprocessor)

    feature_columns = numeric_features + categorical_features

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].values

    results: list[BreachModelResult] = []

    for model_name, pipeline in candidate_models.items():
        pipeline.fit(X_train, y_train)

        y_score = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_score)

        predictions = test_df[
            [
                col
                for col in [
                    "package_id",
                    "city",
                    "zone_id",
                    "courier_id",
                    "accepted_time",
                    "promised_delivery_time",
                    TARGET_COLUMN,
                    "time_to_window_end_minutes",
                    "distance_km",
                    "expected_travel_time_minutes",
                    "feasibility_margin_minutes",
                    "courier_workload_2h",
                    "time_pressure_score",
                    "distance_feasibility_pressure_score",
                    "workload_pressure_score",
                ]
                if col in test_df.columns
            ]
        ].copy()

        predictions["predicted_breach_probability"] = y_score
        predictions["predicted_breach"] = (y_score >= 0.50).astype(int)
        predictions["model_name"] = model_name

        results.append(
            BreachModelResult(
                model_name=model_name,
                pipeline=pipeline,
                metrics=metrics,
                predictions=predictions,
            )
        )

    best_result = max(
        results,
        key=lambda result: result.metrics["average_precision_pr_auc"],
    )

    return best_result, results, train_df, test_df


def save_breach_model_artifacts(
    best_result: BreachModelResult,
    all_results: list[BreachModelResult],
    output_dir: str | Path = "models/breach_prediction",
) -> None:
    """
    Save best model, all model metrics, and best prediction output.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "best_breach_model.joblib"
    metrics_path = output_path / "breach_model_metrics.csv"
    predictions_path = output_path / "breach_model_predictions.csv"

    joblib.dump(best_result.pipeline, model_path)

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": result.model_name,
                **result.metrics,
            }
            for result in all_results
        ]
    )

    metrics_df.to_csv(metrics_path, index=False)
    best_result.predictions.to_csv(predictions_path, index=False)

    print(f"Saved best model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")