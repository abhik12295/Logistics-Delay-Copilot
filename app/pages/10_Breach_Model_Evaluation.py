from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Breach Model Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Breach Model Evaluation")

st.markdown(
    """
This page evaluates the proactive machine learning model for predicting
**service-window breach risk** in last-mile pickup logistics.

The key research question is:

> **Can machine learning identify the small number of pickup tasks likely to miss
> their service window early enough for dispatchers to intervene?**

Because service-window breaches are rare, this page focuses on imbalance-aware and
operations-oriented metrics:

- PR-AUC
- ROC-AUC
- Precision@K
- Recall@K
- Lift@K
- top-risk capture rate
- probability calibration
- intervention queue effectiveness
"""
)

METRICS_PATH = ROOT_DIR / "models" / "breach_prediction" / "breach_model_metrics.csv"

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_predictions_with_urgency.csv"
)

CALIBRATION_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_calibration.csv"
)

if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists():
    st.error(
        "Breach model evaluation files were not found.\n\n"
        "Run these commands first:\n\n"
        "`uv run python scripts/prepare_breach_dataset.py`\n\n"
        "`uv run python scripts/train_breach_model.py`"
    )
    st.stop()


@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "threshold",
        "positive_rate_actual",
        "positive_rate_predicted",
        "precision",
        "recall",
        "f1_score",
        "average_precision_pr_auc",
        "precision_at_10",
        "recall_at_10",
        "lift_at_10",
        "precision_at_25",
        "recall_at_25",
        "lift_at_25",
        "precision_at_50",
        "recall_at_50",
        "lift_at_50",
        "precision_at_100",
        "recall_at_100",
        "lift_at_100",
        "roc_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "service_window_breach",
        "predicted_breach_probability",
        "predicted_breach",
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

    if "accepted_time" in df.columns:
        df["accepted_time"] = pd.to_datetime(df["accepted_time"], errors="coerce")

    return df


@st.cache_data
def load_calibration(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    numeric_cols = [
        "mean_predicted_probability",
        "actual_breach_rate",
        "record_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_topk_table(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "service_window_breach",
) -> pd.DataFrame:
    sorted_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    total_breaches = sorted_df[label_col].sum()
    total_tasks = len(sorted_df)

    k_values = [10, 25, 50, 100, 200, 500]
    rows = []

    for k in k_values:
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
                "Expected Breaches by Random Selection": random_expected,
                "Lift over Random": lift,
            }
        )

    return pd.DataFrame(rows)


metrics_df = load_metrics(METRICS_PATH)
predictions_df = load_predictions(PREDICTIONS_PATH)
calibration_df = load_calibration(CALIBRATION_PATH)

st.divider()

# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------
st.subheader("Model Comparison")

display_metric_cols = [
    "model_name",
    "average_precision_pr_auc",
    "roc_auc",
    "threshold",
    "threshold_method",
    "precision",
    "recall",
    "f1_score",
    "precision_at_50",
    "recall_at_50",
    "lift_at_50",
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
]

available_metric_cols = [
    col for col in display_metric_cols if col in metrics_df.columns
]

model_comparison_df = metrics_df[available_metric_cols].copy()

if "average_precision_pr_auc" in model_comparison_df.columns:
    model_comparison_df = model_comparison_df.sort_values(
        "average_precision_pr_auc",
        ascending=False,
    )

st.dataframe(
    model_comparison_df,
    use_container_width=True,
)

best_model = model_comparison_df.iloc[0]["model_name"]

st.success(f"Best model by PR-AUC: **{best_model}**")

st.markdown(
    """
**Interpretation:**  
PR-AUC is the most important model-selection metric here because service-window
breaches are rare. Lift@K and Recall@K are the most important operational metrics
because dispatchers usually review only a limited number of high-risk tasks.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Best model metrics
# ---------------------------------------------------------------------
st.subheader("Best Model Metrics")

best_metrics = metrics_df.sort_values(
    "average_precision_pr_auc",
    ascending=False,
).iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "PR-AUC",
    f"{best_metrics.get('average_precision_pr_auc', 0):.4f}",
)
col2.metric(
    "ROC-AUC",
    (
        f"{best_metrics.get('roc_auc'):.4f}"
        if pd.notna(best_metrics.get("roc_auc"))
        else "N/A"
    ),
)
col3.metric(
    "Recall",
    f"{best_metrics.get('recall', 0):.2%}",
)
col4.metric(
    "Precision",
    f"{best_metrics.get('precision', 0):.2%}",
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Precision@50",
    f"{best_metrics.get('precision_at_50', 0):.2%}",
)
col6.metric(
    "Recall@50",
    f"{best_metrics.get('recall_at_50', 0):.2%}",
)
col7.metric(
    "Lift@50",
    f"{best_metrics.get('lift_at_50', 0):.2f}x",
)
col8.metric(
    "Model Threshold",
    f"{best_metrics.get('threshold', 0):.4f}",
)

st.divider()

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("PR-AUC by Model")

    fig = px.bar(
        metrics_df.sort_values(
            "average_precision_pr_auc",
            ascending=False,
        ),
        x="model_name",
        y="average_precision_pr_auc",
        title="Average Precision / PR-AUC",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Lift@50 by Model")

    if "lift_at_50" in metrics_df.columns:
        fig = px.bar(
            metrics_df.sort_values(
                "lift_at_50",
                ascending=False,
            ),
            x="model_name",
            y="lift_at_50",
            title="Lift@50 over Random Selection",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Lift@50 metric is missing.")

st.divider()

# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------
st.subheader("Probability Calibration")

if calibration_df.empty:
    st.warning(
        "Calibration file was not found. Run `uv run python scripts/train_breach_model.py` again."
    )
else:
    selected_calibration_model = st.selectbox(
        "Select model for calibration view",
        sorted(calibration_df["model_name"].dropna().unique().tolist()),
        index=0,
    )

    selected_calibration_df = calibration_df[
        calibration_df["model_name"] == selected_calibration_model
    ].copy()

    st.dataframe(selected_calibration_df, use_container_width=True)

    fig = px.line(
        selected_calibration_df,
        x="mean_predicted_probability",
        y="actual_breach_rate",
        markers=True,
        title="Calibration Curve: Predicted Probability vs Actual Breach Rate",
        hover_data=["record_count", "probability_bin"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
Calibration checks whether predicted probabilities are realistic. In rare-event
prediction, the absolute probabilities may be low, but the highest probability bins
should still have higher actual breach rates than lower bins.
"""
    )

st.divider()

# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------
st.subheader("Confusion Matrix at Selected Model Threshold")

tn = int(best_metrics.get("true_negative", 0))
fp = int(best_metrics.get("false_positive", 0))
fn = int(best_metrics.get("false_negative", 0))
tp = int(best_metrics.get("true_positive", 0))

confusion_df = pd.DataFrame(
    [
        {
            "Actual": "No Breach",
            "Predicted No Breach": tn,
            "Predicted Breach": fp,
        },
        {
            "Actual": "Breach",
            "Predicted No Breach": fn,
            "Predicted Breach": tp,
        },
    ]
)

st.dataframe(confusion_df, use_container_width=True)

st.markdown(
    """
The threshold-based confusion matrix is useful, but the main operational scenario is
a **ranked dispatch queue**. Dispatchers usually act on the highest-risk tasks first,
so top-K evaluation is more relevant than a fixed 0.50 threshold.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Top-K operational evaluation
# ---------------------------------------------------------------------
st.subheader("Top-K Dispatch Prioritization")

st.markdown(
    """
This section compares two ranking strategies:

1. **Model probability ranking** — ranks tasks by predicted breach probability.
2. **Intervention urgency ranking** — ranks tasks by model risk percentile plus
   operational pressure signals.
"""
)

probability_topk_df = build_topk_table(
    predictions_df,
    score_col="predicted_breach_probability",
)

urgency_topk_df = build_topk_table(
    predictions_df,
    score_col="intervention_urgency_score",
)

probability_topk_df["Ranking Strategy"] = "Model probability"
urgency_topk_df["Ranking Strategy"] = "Intervention urgency"

combined_topk_df = pd.concat(
    [probability_topk_df, urgency_topk_df],
    ignore_index=True,
)

st.dataframe(combined_topk_df, use_container_width=True)

fig = px.line(
    combined_topk_df,
    x="K",
    y="Recall@K",
    color="Ranking Strategy",
    markers=True,
    title="Recall@K by Ranking Strategy",
)
st.plotly_chart(fig, use_container_width=True)

fig = px.line(
    combined_topk_df,
    x="K",
    y="Lift over Random",
    color="Ranking Strategy",
    markers=True,
    title="Lift@K over Random Selection",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**Research interpretation:**  
Top-K evaluation directly measures dispatch usefulness. A good model should capture
more actual service-window breaches in the first 50 or 100 reviewed tasks than random
selection. The intervention urgency ranking is especially important because it combines
ML risk with operational feasibility signals.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Top risky tasks
# ---------------------------------------------------------------------
st.subheader("Top-Ranked Intervention Queue Records")

ranking_choice = st.radio(
    "Sort top records by",
    options=[
        "intervention_urgency_score",
        "predicted_breach_probability",
    ],
    horizontal=True,
)

records_sorted = predictions_df.sort_values(
    ranking_choice,
    ascending=False,
).reset_index(drop=True)

top_records_cols = [
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

available_top_records_cols = [
    col for col in top_records_cols if col in records_sorted.columns
]

top_records = records_sorted[available_top_records_cols].head(50)

st.dataframe(top_records, use_container_width=True)

top50_breaches = int(records_sorted.head(50)["service_window_breach"].sum())

st.info(
    f"Top 50 records by `{ranking_choice}` capture **{top50_breaches}** actual service-window breaches."
)

st.divider()

# ---------------------------------------------------------------------
# Download section
# ---------------------------------------------------------------------
st.subheader("Download Evaluation Outputs")

csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
csv_predictions = predictions_df.to_csv(index=False).encode("utf-8")
csv_topk = combined_topk_df.to_csv(index=False).encode("utf-8")

download_col1, download_col2, download_col3 = st.columns(3)

download_col1.download_button(
    label="Download model metrics",
    data=csv_metrics,
    file_name="breach_model_metrics.csv",
    mime="text/csv",
)

download_col2.download_button(
    label="Download predictions",
    data=csv_predictions,
    file_name="breach_model_predictions.csv",
    mime="text/csv",
)

download_col3.download_button(
    label="Download top-K evaluation",
    data=csv_topk,
    file_name="topk_dispatch_evaluation.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------------------
# Paper-ready summary
# ---------------------------------------------------------------------
st.subheader("Paper-Ready Summary")

actual_breach_rate = predictions_df["service_window_breach"].mean()

best_pr_auc = best_metrics.get("average_precision_pr_auc", 0)
best_roc_auc = best_metrics.get("roc_auc", 0)
best_lift_50 = best_metrics.get("lift_at_50", 0)

urgency_top50 = urgency_topk_df[urgency_topk_df["K"] == 50].iloc[0]
urgency_top50_captured = int(urgency_top50["Captured Breaches"])
urgency_top50_recall = urgency_top50["Recall@K"]
urgency_top50_lift = urgency_top50["Lift over Random"]

st.markdown(
    f"""
The proactive breach prediction experiment evaluates whether machine learning can
identify pickup tasks that are likely to miss their service window using only information
available at task acceptance time.

In the test split, the actual service-window breach rate is approximately
**{actual_breach_rate:.2%}**, confirming that the task is a rare-event and imbalanced
classification problem.

The best-performing model by PR-AUC is **{best_model}**, achieving:

- **PR-AUC:** {best_pr_auc:.4f}
- **ROC-AUC:** {best_roc_auc:.4f}
- **Lift@50:** {best_lift_50:.2f}x

Using the intervention urgency ranking, the top 50 dispatch queue records capture
**{urgency_top50_captured}** actual breaches, corresponding to **Recall@50 =
{urgency_top50_recall:.2%}** and **Lift@50 = {urgency_top50_lift:.2f}x** over random
task selection.

This supports the central research claim that proactive ML-based ranking can help
dispatchers focus on a small subset of pickup tasks with disproportionately high
service-window breach risk.
"""
)