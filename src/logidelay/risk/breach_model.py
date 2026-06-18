from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "service_window_breach"

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
    pipeline: Pipeline | CalibratedClassifierCV
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    calibration: pd.DataFrame


def get_available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
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


def _make_pipeline(
    classifier: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("classifier", classifier),
        ]
    )


def _make_calibrated_model(
    base_pipeline: Pipeline,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv=cv,
    )


def build_candidate_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline | CalibratedClassifierCV]:
    logistic_pipeline = _make_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        numeric_features,
        categorical_features,
    )

    random_forest_pipeline = _make_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        numeric_features,
        categorical_features,
    )

    hist_gradient_pipeline = _make_pipeline(
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        ),
        numeric_features,
        categorical_features,
    )

    calibrated_logistic = _make_calibrated_model(
        _make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            numeric_features,
            categorical_features,
        )
    )

    calibrated_random_forest = _make_calibrated_model(
        _make_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            numeric_features,
            categorical_features,
        )
    )

    calibrated_hist_gradient = _make_calibrated_model(
        _make_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42,
            ),
            numeric_features,
            categorical_features,
        )
    )

    return {
        "logistic_regression_balanced": logistic_pipeline,
        "random_forest_balanced": random_forest_pipeline,
        "hist_gradient_boosting": hist_gradient_pipeline,
        "calibrated_logistic_regression": calibrated_logistic,
        "calibrated_random_forest": calibrated_random_forest,
        "calibrated_hist_gradient_boosting": calibrated_hist_gradient,
    }


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0

    k = min(k, len(y_true))
    top_indices = np.argsort(y_score)[::-1][:k]

    return float(np.mean(y_true[top_indices]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    positives = np.sum(y_true)

    if positives == 0:
        return 0.0

    k = min(k, len(y_true))
    top_indices = np.argsort(y_score)[::-1][:k]
    captured_positives = np.sum(y_true[top_indices])

    return float(captured_positives / positives)


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0

    base_rate = float(np.mean(y_true))

    if base_rate == 0:
        return 0.0

    return precision_at_k(y_true, y_score, k) / base_rate


def choose_threshold_by_fbeta(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 2.0,
) -> float:
    """
    Select a threshold that emphasizes recall while still considering precision.

    F2 is useful for dispatch-risk detection because missing a true breach is usually
    worse than reviewing a small number of false alerts.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) == 0:
        return 0.50

    precision = precision[:-1]
    recall = recall[:-1]

    beta_squared = beta**2

    fbeta = (
        (1 + beta_squared)
        * precision
        * recall
        / ((beta_squared * precision) + recall + 1e-12)
    )

    best_index = int(np.nanargmax(fbeta))

    return float(thresholds[best_index])


def build_calibration_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    calibration_df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted_probability": y_score,
        }
    )

    try:
        calibration_df["probability_bin"] = pd.qcut(
            calibration_df["predicted_probability"],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        calibration_df["probability_bin"] = pd.cut(
            calibration_df["predicted_probability"],
            bins=n_bins,
            duplicates="drop",
        )

    grouped = (
        calibration_df.groupby("probability_bin", observed=False)
        .agg(
            mean_predicted_probability=("predicted_probability", "mean"),
            actual_breach_rate=("actual", "mean"),
            record_count=("actual", "size"),
        )
        .reset_index()
    )

    grouped["probability_bin"] = grouped["probability_bin"].astype(str)

    return grouped


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    if threshold is None:
        threshold = choose_threshold_by_fbeta(y_true, y_score, beta=2.0)
        threshold_method = "f2_optimized"
    else:
        threshold_method = "manual"

    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "threshold_method": threshold_method,
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
        "precision_at_10": precision_at_k(y_true, y_score, 10),
        "recall_at_10": recall_at_k(y_true, y_score, 10),
        "lift_at_10": lift_at_k(y_true, y_score, 10),
        "precision_at_25": precision_at_k(y_true, y_score, 25),
        "recall_at_25": recall_at_k(y_true, y_score, 25),
        "lift_at_25": lift_at_k(y_true, y_score, 25),
        "precision_at_50": precision_at_k(y_true, y_score, 50),
        "recall_at_50": recall_at_k(y_true, y_score, 50),
        "lift_at_50": lift_at_k(y_true, y_score, 50),
        "precision_at_100": precision_at_k(y_true, y_score, 100),
        "recall_at_100": recall_at_k(y_true, y_score, 100),
        "lift_at_100": lift_at_k(y_true, y_score, 100),
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


def _build_prediction_output(
    test_df: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    output_cols = [
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
        "historical_courier_breach_rate",
        "historical_zone_breach_rate",
        "historical_city_breach_rate",
    ]

    available_cols = [col for col in output_cols if col in test_df.columns]

    predictions = test_df[available_cols].copy()
    predictions["predicted_breach_probability"] = y_score
    predictions["predicted_breach"] = (y_score >= threshold).astype(int)
    predictions["model_name"] = model_name

    return predictions


def train_and_evaluate_breach_models(
    df: pd.DataFrame,
) -> tuple[BreachModelResult, list[BreachModelResult], pd.DataFrame, pd.DataFrame]:
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

    candidate_models = build_candidate_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    feature_columns = numeric_features + categorical_features

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].values

    results: list[BreachModelResult] = []

    for model_name, model in candidate_models.items():
        print(f"Training model: {model_name}")

        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_score)
        threshold = float(metrics["threshold"])

        predictions = _build_prediction_output(
            test_df=test_df,
            y_score=y_score,
            threshold=threshold,
            model_name=model_name,
        )

        calibration = build_calibration_table(y_test, y_score)
        calibration["model_name"] = model_name

        results.append(
            BreachModelResult(
                model_name=model_name,
                pipeline=model,
                metrics=metrics,
                predictions=predictions,
                calibration=calibration,
            )
        )

    best_result = max(
        results,
        key=lambda result: (
            result.metrics["average_precision_pr_auc"],
            result.metrics["recall_at_50"],
            result.metrics["lift_at_50"],
        ),
    )

    return best_result, results, train_df, test_df


def save_breach_model_artifacts(
    best_result: BreachModelResult,
    all_results: list[BreachModelResult],
    output_dir: str | Path = "models/breach_prediction",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "best_breach_model.joblib"
    metrics_path = output_path / "breach_model_metrics.csv"
    predictions_path = output_path / "breach_model_predictions.csv"
    calibration_path = output_path / "breach_model_calibration.csv"

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

    calibration_df = pd.concat(
        [result.calibration for result in all_results],
        ignore_index=True,
    )

    metrics_df.to_csv(metrics_path, index=False)
    best_result.predictions.to_csv(predictions_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)

    print(f"Saved best model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved calibration table to: {calibration_path}")