from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.copilot.local_llm_engine import generate_dispatcher_recommendation


PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_predictions_with_urgency.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "llm_recommendation_evaluation.csv"
)

SUMMARY_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "llm_recommendation_evaluation_summary.csv"
)


EVIDENCE_COLUMNS = [
    "package_id",
    "city",
    "zone_id",
    "intervention_priority",
    "predicted_breach_probability",
    "model_risk_rank_score",
    "intervention_urgency_score",
    "time_to_window_end_minutes",
    "distance_km",
    "expected_travel_time_minutes",
    "feasibility_margin_minutes",
    "courier_workload_2h",
    "time_pressure_score",
    "distance_feasibility_pressure_score",
    "workload_pressure_score",
]


UNSUPPORTED_TERMS = [
    "weather",
    "rain",
    "snow",
    "traffic jam",
    "accident",
    "road closure",
    "vehicle breakdown",
    "customer called",
    "customer unavailable",
    "address incorrect",
    "warehouse delay",
    "driver illness",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_task_evidence(row: pd.Series) -> dict[str, Any]:
    return {
        col: row.get(col)
        for col in EVIDENCE_COLUMNS
        if col in row.index
    }


def combine_recommendation_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("risk_summary", "")),
            str(row.get("evidence_summary", "")),
            str(row.get("recommended_action", "")),
            str(row.get("dispatcher_note", "")),
        ]
    ).lower()


def evaluate_hallucination(text: str) -> int:
    """
    Returns 1 if unsupported operational claims are detected, else 0.
    """
    text_lower = text.lower()

    for term in UNSUPPORTED_TERMS:
        if term in text_lower:
            return 1

    return 0


def evaluate_evidence_coverage(text: str) -> float:
    """
    Measures whether the recommendation references key evidence categories.
    """

    evidence_groups = {
        "model_risk": ["model", "risk", "probability", "predicted"],
        "urgency": ["urgency", "priority", "rank"],
        "time": ["time", "window", "minutes", "deadline"],
        "feasibility": ["feasibility", "margin", "distance", "travel"],
        "workload": ["workload", "courier", "tasks"],
    }

    covered = 0

    for keywords in evidence_groups.values():
        if any(keyword in text for keyword in keywords):
            covered += 1

    return covered / len(evidence_groups)


def evaluate_action_alignment(text: str, priority: str) -> float:
    """
    Checks whether the recommendation action aligns with the intervention priority.
    """

    priority = str(priority).strip().lower()

    high_action_terms = [
        "prioritize",
        "review",
        "confirm",
        "reassign",
        "escalate",
        "backup",
        "monitor",
    ]

    medium_action_terms = [
        "monitor",
        "review",
        "escalate",
        "track",
    ]

    low_action_terms = [
        "normal",
        "monitoring",
        "no immediate",
        "continue",
    ]

    if priority in {"critical", "high"}:
        return 1.0 if any(term in text for term in high_action_terms) else 0.0

    if priority == "medium":
        return 1.0 if any(term in text for term in medium_action_terms) else 0.0

    if priority == "low":
        return 1.0 if any(term in text for term in low_action_terms) else 0.5

    return 0.5


def evaluate_completeness(row: dict[str, Any]) -> float:
    fields = [
        "risk_summary",
        "evidence_summary",
        "recommended_action",
        "dispatcher_note",
        "confidence",
    ]

    completed = 0

    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            completed += 1

    return completed / len(fields)


def evaluate_readability(text: str) -> float:
    """
    Simple readability proxy based on length.

    A practical dispatcher recommendation should be concise but not empty.
    """

    word_count = len(text.split())

    if 35 <= word_count <= 120:
        return 1.0

    if 20 <= word_count < 35:
        return 0.8

    if 120 < word_count <= 170:
        return 0.7

    if word_count > 170:
        return 0.5

    return 0.3


def calculate_overall_score(
    evidence_coverage_score: float,
    action_alignment_score: float,
    completeness_score: float,
    readability_score: float,
    hallucination_flag: int,
) -> float:
    hallucination_score = 0.0 if hallucination_flag else 1.0

    return (
        0.30 * evidence_coverage_score
        + 0.25 * action_alignment_score
        + 0.20 * completeness_score
        + 0.15 * hallucination_score
        + 0.10 * readability_score
    )


def evaluate_recommendations(
    top_k: int,
    use_ollama: bool,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}. "
            "Run scripts/train_breach_model.py first."
        )

    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    required_cols = [
        "intervention_urgency_score",
        "service_window_breach",
        "package_id",
    ]

    missing_cols = [
        col for col in required_cols if col not in predictions_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    predictions_df["intervention_urgency_score"] = pd.to_numeric(
        predictions_df["intervention_urgency_score"],
        errors="coerce",
    ).fillna(0)

    top_tasks_df = (
        predictions_df.sort_values(
            "intervention_urgency_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    records: list[dict[str, Any]] = []

    for index, row in top_tasks_df.iterrows():
        dispatch_rank = index + 1
        task_evidence = build_task_evidence(row)

        recommendation = generate_dispatcher_recommendation(
            task_evidence=task_evidence,
            use_ollama=use_ollama,
            model_name=model_name,
        )

        result = {
            "dispatch_rank": dispatch_rank,
            "package_id": row.get("package_id"),
            "city": row.get("city"),
            "zone_id": row.get("zone_id"),
            "service_window_breach": row.get("service_window_breach"),
            "intervention_priority": row.get("intervention_priority"),
            "predicted_breach_probability": row.get(
                "predicted_breach_probability"
            ),
            "model_risk_rank_score": row.get("model_risk_rank_score"),
            "intervention_urgency_score": row.get(
                "intervention_urgency_score"
            ),
            "recommendation_source": recommendation.source,
            "risk_summary": recommendation.risk_summary,
            "evidence_summary": recommendation.evidence_summary,
            "recommended_action": recommendation.recommended_action,
            "dispatcher_note": recommendation.dispatcher_note,
            "confidence": recommendation.confidence,
        }

        combined_text = combine_recommendation_text(result)

        hallucination_flag = evaluate_hallucination(combined_text)
        evidence_coverage_score = evaluate_evidence_coverage(combined_text)
        action_alignment_score = evaluate_action_alignment(
            combined_text,
            str(row.get("intervention_priority", "")),
        )
        completeness_score = evaluate_completeness(result)
        readability_score = evaluate_readability(combined_text)

        overall_score = calculate_overall_score(
            evidence_coverage_score=evidence_coverage_score,
            action_alignment_score=action_alignment_score,
            completeness_score=completeness_score,
            readability_score=readability_score,
            hallucination_flag=hallucination_flag,
        )

        result.update(
            {
                "hallucination_flag": hallucination_flag,
                "evidence_coverage_score": evidence_coverage_score,
                "action_alignment_score": action_alignment_score,
                "completeness_score": completeness_score,
                "readability_score": readability_score,
                "overall_recommendation_score": overall_score,
            }
        )

        records.append(result)

    evaluation_df = pd.DataFrame(records)

    summary_df = pd.DataFrame(
        [
            {
                "top_k": top_k,
                "use_ollama_requested": use_ollama,
                "model_name": model_name,
                "records_evaluated": len(evaluation_df),
                "recommendation_source_values": ", ".join(
                    sorted(evaluation_df["recommendation_source"].unique())
                ),
                "actual_breaches_in_top_k": int(
                    evaluation_df["service_window_breach"].sum()
                ),
                "hallucination_rate": evaluation_df[
                    "hallucination_flag"
                ].mean(),
                "avg_evidence_coverage_score": evaluation_df[
                    "evidence_coverage_score"
                ].mean(),
                "avg_action_alignment_score": evaluation_df[
                    "action_alignment_score"
                ].mean(),
                "avg_completeness_score": evaluation_df[
                    "completeness_score"
                ].mean(),
                "avg_readability_score": evaluation_df[
                    "readability_score"
                ].mean(),
                "avg_overall_recommendation_score": evaluation_df[
                    "overall_recommendation_score"
                ].mean(),
            }
        ]
    )

    return evaluation_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GenAI dispatcher recommendations."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top urgency-ranked tasks to evaluate.",
    )

    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama model. Falls back to rule-based recommendations if unavailable.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="llama3.2:3b",
        help="Ollama model name.",
    )

    args = parser.parse_args()

    evaluation_df, summary_df = evaluate_recommendations(
        top_k=args.top_k,
        use_ollama=args.use_ollama,
        model_name=args.model_name,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    evaluation_df.to_csv(OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved recommendation evaluation to: {OUTPUT_PATH}")
    print(f"Saved recommendation summary to: {SUMMARY_PATH}")

    print("\nRecommendation evaluation summary:")
    print(summary_df.to_string(index=False))

    print("\nPreview:")
    preview_cols = [
        "dispatch_rank",
        "package_id",
        "service_window_breach",
        "intervention_priority",
        "recommendation_source",
        "hallucination_flag",
        "evidence_coverage_score",
        "action_alignment_score",
        "completeness_score",
        "overall_recommendation_score",
    ]

    available_preview_cols = [
        col for col in preview_cols if col in evaluation_df.columns
    ]

    print(evaluation_df[available_preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()