# from __future__ import annotations

# import sys
# from pathlib import Path

# import pandas as pd
# import plotly.express as px
# import streamlit as st

# ROOT_DIR = Path(__file__).resolve().parents[1]
# SRC_DIR = ROOT_DIR / "src"

# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# from logidelay.copilot.explanation_engine import generate_explanation
# from logidelay.utils.app_helpers import load_data_from_sidebar
# from logidelay.copilot.explanation_engine import generate_explanation
# from logidelay.diagnosis.weak_labeler import add_root_cause_labels
# from logidelay.features.event_features import add_event_features
# from logidelay.severity.scoring import add_operational_exception_severity


# st.set_page_config(
#     page_title="LogiDelay Copilot",
#     page_icon="🚚",
#     layout="wide",
# )

# st.title("🚚 LogiDelay Copilot")
# st.subheader("Explainable AI for Logistics Delay Diagnosis")

# st.markdown(
# """
# LogiDelay Copilot is a research prototype for diagnosing logistics service delays
# using event logs and **Operational Exception Severity**.

# The system identifies delayed logistics tasks, diagnoses the likely operational cause,
# assigns a severity score, and generates a planner-facing explanation with a
# recommended action.

# The app supports both synthetic MVP data and a standardized public LaDe-P sample.
# No paid AI API is required.
# """
# )

# df = load_data_from_sidebar(str(ROOT_DIR))

# st.sidebar.download_button(
#     label="Download current dataset",
#     data=df.to_csv(index=False).encode("utf-8"),
#     file_name="logidelay_current_dataset.csv",
#     mime="text/csv",
# )

# total_deliveries = len(df)
# delayed_count = int(df["is_delayed"].sum())
# avg_delay = df["delay_minutes"].mean()
# critical_count = int((df["severity_class"] == "Critical").sum())

# col1, col2, col3, col4 = st.columns(4)

# col1.metric("Total Records", f"{total_deliveries:,}")
# col2.metric("Delayed Records", f"{delayed_count:,}")
# col3.metric("Avg Delay Minutes", f"{avg_delay:.1f}")
# col4.metric("Critical Exceptions", f"{critical_count:,}")

# st.divider()

# left, right = st.columns(2)

# with left:
#     st.markdown("### Delay Category Distribution")
#     delay_chart = (
#         df["delay_category"]
#         .value_counts()
#         .rename_axis("delay_category")
#         .reset_index(name="count")
#     )
#     fig = px.bar(
#         delay_chart,
#         x="delay_category",
#         y="count",
#         text="count",
#         title="Deliveries by Delay Category",
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with right:
#     st.markdown("### Operational Exception Severity")
#     severity_chart = (
#         df["severity_class"]
#         .value_counts()
#         .rename_axis("severity_class")
#         .reset_index(name="count")
#     )
#     fig = px.bar(
#         severity_chart,
#         x="severity_class",
#         y="count",
#         text="count",
#         title="Deliveries by Severity Class",
#     )
#     st.plotly_chart(fig, use_container_width=True)

# st.divider()

# st.markdown("### Delivery Diagnosis Explorer")

# package_options = df["package_id"].tolist()
# selected_package = st.selectbox("Select a package", package_options)

# record = df[df["package_id"] == selected_package].iloc[0]

# c1, c2, c3 = st.columns(3)
# c1.metric("Root Cause", record["root_cause_label"])
# c2.metric("Severity", record["severity_class"])
# c3.metric("OES Score", f"{record['operational_exception_severity']:.2f}")

# st.markdown("#### Event Evidence")

# evidence_cols = [
#     "package_id",
#     "courier_id",
#     "city",
#     "zone_id",
#     "origin_lat",
#     "origin_lng",
#     "destination_lat",
#     "destination_lng",
#     "distance_km",
#     "assigned_time",
#     "accepted_time",
#     "pickup_time",
#     "completed_time",
#     "promised_delivery_time",
#     "expected_execution_time_minutes",
#     "distance_adjusted_execution_ratio",
#     "acceptance_gap_minutes",
#     "pickup_gap_minutes",
#     "execution_time_minutes",
#     "delay_minutes",
#     "courier_workload_2h",
#     "root_cause_label",
#     "severity_class",
# ]

# available_cols = [col for col in evidence_cols if col in df.columns]
# st.dataframe(pd.DataFrame([record[available_cols]]), use_container_width=True)

# st.markdown("#### Copilot Explanation")
# st.info(generate_explanation(record))

# st.divider()

# st.markdown("### Research Pipeline")

# st.code(
#     """
# Logistics Event Logs
# → Event Feature Engineering
# → Delay Detection
# → Root-Cause Weak Labeling
# → Operational Exception Severity Scoring
# → Free GenAI-style Explanation
# → Planner Recommendation
# """,
#     language="text",
# )

# st.markdown(
#     """
# ### Next Research Output

# Use the **Research Results** page to generate paper-ready summary tables for:

# - delay distribution
# - root-cause distribution
# - severity distribution
# - average OES by root cause
# - top exception examples
# """
# )

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="LogiDelay Copilot",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 LogiDelay Copilot")

st.markdown(
    """
## Proactive AI for Logistics Service-Window Breach Prevention

LogiDelay Copilot is a research prototype for predicting and preventing
**service-window breaches** in last-mile logistics.

Instead of only explaining delays after they happen, the system predicts which pickup
tasks are likely to miss their service window and ranks them for dispatcher intervention.

The project combines:

- courier event-log processing,
- proactive service-window breach prediction,
- distance-aware feasibility features,
- workload-aware risk indicators,
- calibrated machine learning models,
- intervention urgency scoring,
- top-K dispatch prioritization,
- and grounded AI/GenAI-style dispatcher recommendations.
"""
)

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_predictions_with_urgency.csv"
)

METRICS_PATH = (
    ROOT_DIR
    / "models"
    / "breach_prediction"
    / "breach_model_metrics.csv"
)

LLM_SUMMARY_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "llm_recommendation_evaluation_summary.csv"
)

@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "service_window_breach",
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

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "average_precision_pr_auc",
        "roc_auc",
        "precision_at_50",
        "recall_at_50",
        "lift_at_50",
        "precision",
        "recall",
        "threshold",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

@st.cache_data
def load_llm_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "top_k",
        "records_evaluated",
        "actual_breaches_in_top_k",
        "hallucination_rate",
        "avg_evidence_coverage_score",
        "avg_action_alignment_score",
        "avg_completeness_score",
        "avg_readability_score",
        "avg_overall_recommendation_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def build_topk_summary(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "service_window_breach",
    k: int = 50,
) -> dict:
    sorted_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    topk = sorted_df.head(k)
    total_breaches = int(sorted_df[label_col].sum())
    captured_breaches = int(topk[label_col].sum())

    precision_k = captured_breaches / k if k else 0
    recall_k = captured_breaches / total_breaches if total_breaches else 0

    random_expected = (k / len(sorted_df)) * total_breaches if len(sorted_df) else 0
    lift = captured_breaches / random_expected if random_expected else 0

    return {
        "k": k,
        "captured_breaches": captured_breaches,
        "precision_k": precision_k,
        "recall_k": recall_k,
        "lift": lift,
    }


def build_topk_table(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "service_window_breach",
) -> pd.DataFrame:
    sorted_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    total_breaches = sorted_df[label_col].sum()
    total_tasks = len(sorted_df)

    rows = []

    for k in [10, 25, 50, 100, 200, 500]:
        k = min(k, total_tasks)
        topk = sorted_df.head(k)

        captured_breaches = int(topk[label_col].sum())
        precision_k = captured_breaches / k if k else 0
        recall_k = captured_breaches / total_breaches if total_breaches else 0
        random_expected = (k / total_tasks) * total_breaches if total_tasks else 0
        lift = captured_breaches / random_expected if random_expected else 0

        rows.append(
            {
                "K": k,
                "Captured Breaches": captured_breaches,
                "Precision@K": precision_k,
                "Recall@K": recall_k,
                "Lift over Random": lift,
            }
        )

    return pd.DataFrame(rows)


st.divider()

if not PREDICTIONS_PATH.exists() or not METRICS_PATH.exists():
    st.warning(
        """
The proactive breach prediction outputs were not found yet.

Run these commands locally:

```bash
uv run python scripts/prepare_breach_dataset.py
uv run python scripts/train_breach_model.py
```

Then refresh the app.
"""
    )

    st.subheader("Research Pipeline")

    st.code(
        """
LaDe-P public pickup event logs
→ Standardized logistics schema
→ Proactive breach feature engineering
→ Calibrated ML breach prediction
→ Intervention urgency ranking
→ Dispatcher recommendation
→ Top-K operational evaluation
""",
        language="text",
    )

    st.stop()

predictions_df = load_predictions(PREDICTIONS_PATH)
metrics_df = load_metrics(METRICS_PATH)

best_metrics = metrics_df.sort_values(
    "average_precision_pr_auc",
    ascending=False,
).iloc[0]

best_model = best_metrics["model_name"]

urgency_top50 = build_topk_summary(
    predictions_df,
    score_col="intervention_urgency_score",
    k=50,
)

total_tasks = len(predictions_df)
actual_breaches = int(predictions_df["service_window_breach"].sum())
actual_breach_rate = predictions_df["service_window_breach"].mean()

st.subheader("Current Research Result Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Tasks", f"{total_tasks:,}")
col2.metric("Actual Breaches", f"{actual_breaches:,}", f"{actual_breach_rate:.2%}")
col3.metric("Best Model", str(best_model))
col4.metric("PR-AUC", f"{best_metrics['average_precision_pr_auc']:.4f}")

col5, col6, col7, col8 = st.columns(4)

col5.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")
col6.metric("Model Lift@50", f"{best_metrics['lift_at_50']:.2f}x")
col7.metric("Urgency Top-50 Breaches", urgency_top50["captured_breaches"])
col8.metric("Urgency Lift@50", f"{urgency_top50['lift']:.2f}x")

st.success(
    f"""
Using the proposed intervention urgency ranking, the top 50 dispatch queue captures
**{urgency_top50['captured_breaches']} actual service-window breaches**, with
**Precision@50 = {urgency_top50['precision_k']:.2%}**,
**Recall@50 = {urgency_top50['recall_k']:.2%}**, and
**Lift@50 = {urgency_top50['lift']:.2f}x** over random task selection.
"""
)


st.divider()

st.subheader("GenAI Recommendation Evaluation Snapshot")

if LLM_SUMMARY_PATH.exists():
    llm_summary_df = load_llm_summary(LLM_SUMMARY_PATH)
    llm_summary = llm_summary_df.iloc[0]

    genai_col1, genai_col2, genai_col3, genai_col4 = st.columns(4)

    genai_col1.metric(
        "Recommendation Source",
        str(llm_summary["recommendation_source_values"]),
    )
    genai_col2.metric(
        "Records Evaluated",
        f"{int(llm_summary['records_evaluated']):,}",
    )
    genai_col3.metric(
        "Hallucination Rate",
        f"{llm_summary['hallucination_rate']:.2%}",
    )
    genai_col4.metric(
        "Overall GenAI Score",
        f"{llm_summary['avg_overall_recommendation_score']:.3f}",
    )

    genai_col5, genai_col6, genai_col7, genai_col8 = st.columns(4)

    genai_col5.metric(
        "Evidence Coverage",
        f"{llm_summary['avg_evidence_coverage_score']:.3f}",
    )
    genai_col6.metric(
        "Action Alignment",
        f"{llm_summary['avg_action_alignment_score']:.3f}",
    )
    genai_col7.metric(
        "Completeness",
        f"{llm_summary['avg_completeness_score']:.3f}",
    )
    genai_col8.metric(
        "Readability",
        f"{llm_summary['avg_readability_score']:.3f}",
    )

    st.success(
        f"""
The GenAI recommendation module was evaluated on
**{int(llm_summary['records_evaluated'])} top-ranked dispatch tasks** using
**{llm_summary['recommendation_source_values']}**. It achieved
**hallucination rate = {llm_summary['hallucination_rate']:.2%}** and
**overall recommendation score = {llm_summary['avg_overall_recommendation_score']:.3f}**.
"""
    )
else:
    st.info(
        """
        GenAI recommendation evaluation outputs are not available yet.

        Run:

        ```bash
        uv run python scripts/evaluate_llm_recommendations.py --top-k 50 --use-ollama --model-name qwen2:7b
        """
    )

st.divider()

st.subheader("Why This Problem Matters")

st.markdown(
    """
In last-mile logistics, dispatchers often cannot manually inspect every active pickup task.
The operational challenge is to identify the small number of tasks that are most likely
to miss their service window.

This project treats service-window breach prevention as a rare-event prediction and
ranking problem. The model does not simply classify all tasks. It helps answer:

> Which pickup tasks should dispatch review first?

This makes the system useful for real logistics operations because it converts event-log
data into a prioritized intervention queue.
"""
)

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Intervention Priority Distribution")

    priority_counts = (
        predictions_df["intervention_priority"]
        .value_counts()
        .reset_index()
    )
    priority_counts.columns = ["intervention_priority", "count"]

    fig = px.bar(
        priority_counts,
        x="intervention_priority",
        y="count",
        title="Intervention Priority Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Predicted Breach Probability")

    fig = px.histogram(
        predictions_df,
        x="predicted_breach_probability",
        nbins=30,
        title="Predicted Service-Window Breach Probability",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Top-K Dispatch Ranking Comparison")

probability_topk = build_topk_table(
    predictions_df,
    score_col="predicted_breach_probability",
)

urgency_topk = build_topk_table(
    predictions_df,
    score_col="intervention_urgency_score",
)

probability_topk["Ranking Strategy"] = "Model probability"
urgency_topk["Ranking Strategy"] = "Intervention urgency"

combined_topk = pd.concat(
    [probability_topk, urgency_topk],
    ignore_index=True,
)

st.dataframe(combined_topk, use_container_width=True)

fig = px.line(
    combined_topk,
    x="K",
    y="Recall@K",
    color="Ranking Strategy",
    markers=True,
    title="Recall@K by Ranking Strategy",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Top 20 Intervention Queue")

top_queue_cols = [
    "package_id",
    "city",
    "zone_id",
    "service_window_breach",
    "predicted_breach_probability",
    "model_risk_rank_score",
    "intervention_urgency_score",
    "intervention_priority",
    "time_to_window_end_minutes",
    "distance_km",
    "expected_travel_time_minutes",
    "feasibility_margin_minutes",
    "courier_workload_2h",
]

available_top_queue_cols = [
    col for col in top_queue_cols if col in predictions_df.columns
]

top_queue = (
    predictions_df.sort_values(
        "intervention_urgency_score",
        ascending=False,
    )
    .head(20)
    [available_top_queue_cols]
)

st.dataframe(top_queue, use_container_width=True)

st.caption(
    """
The actual breach label is shown for research evaluation. In a live logistics system,
the actual outcome would not be known at prediction time.
"""
)

st.divider()

st.subheader("Research Contribution")

st.markdown(
    """
The current version of LogiDelay Copilot contributes:

1. A proactive service-window breach prediction framework using public logistics event logs.
2. Distance-aware and workload-aware feature engineering for last-mile pickup tasks.
3. Calibrated ML models for rare-event breach prediction.
4. An intervention urgency score that combines ML risk ranking with operational pressure.
5. Top-K dispatch evaluation using Precision@K, Recall@K, and Lift@K.
6. A foundation for grounded GenAI dispatcher recommendations in the next stage.
"""
)

st.divider()

st.subheader("Next Research Stage")

st.markdown(
    """
The next stage will add a local open-source LLM, such as Ollama, to generate grounded
dispatcher recommendations from structured model evidence.

The LLM will not replace the ML model. Instead, it will translate the model output into
clear operational guidance for dispatchers.
"""
)