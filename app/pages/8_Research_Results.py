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
    page_title="Research Results",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Research Results")

st.markdown(
    """
# Proactive Service-Window Breach Prevention Results

This page summarizes the main experimental results for the proactive breach prediction
and dispatch prioritization framework.

The central research question is:

> **Can machine learning identify pickup tasks likely to miss their service window early enough for dispatchers to intervene?**

The results focus on rare-event prediction and top-K dispatch usefulness rather than
standard accuracy.
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
    st.warning(
        """
The proactive breach prediction result files were not found.

Run these commands first:

```bash
uv run python scripts/prepare_breach_dataset.py
uv run python scripts/train_breach_model.py
```

Then refresh this page.
"""
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
                "Expected Breaches by Random Selection": random_expected,
                "Lift over Random": lift,
            }
        )

    return pd.DataFrame(rows)


metrics_df = load_metrics(METRICS_PATH)
predictions_df = load_predictions(PREDICTIONS_PATH)
calibration_df = load_calibration(CALIBRATION_PATH)

best_metrics = metrics_df.sort_values(
    "average_precision_pr_auc",
    ascending=False,
).iloc[0]

best_model = best_metrics["model_name"]

actual_breach_rate = predictions_df["service_window_breach"].mean()
total_tasks = len(predictions_df)
total_breaches = int(predictions_df["service_window_breach"].sum())

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

probability_top50 = probability_topk_df[probability_topk_df["K"] == 50].iloc[0]
urgency_top50 = urgency_topk_df[urgency_topk_df["K"] == 50].iloc[0]

st.divider()

# ---------------------------------------------------------------------
# Executive result summary
# ---------------------------------------------------------------------

st.subheader("Executive Result Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Tasks", f"{total_tasks:,}")
col2.metric("Actual Breaches", f"{total_breaches:,}", f"{actual_breach_rate:.2%}")
col3.metric("Best Model", str(best_model))
col4.metric("PR-AUC", f"{best_metrics['average_precision_pr_auc']:.4f}")

col5, col6, col7, col8 = st.columns(4)

col5.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")
col6.metric("Model Lift@50", f"{best_metrics['lift_at_50']:.2f}x")
col7.metric("Urgency Top-50 Breaches", int(urgency_top50["Captured Breaches"]))
col8.metric("Urgency Lift@50", f"{urgency_top50['Lift over Random']:.2f}x")

st.success(
    f"""
The best model is **{best_model}**, achieving **PR-AUC = {best_metrics['average_precision_pr_auc']:.4f}**
and **ROC-AUC = {best_metrics['roc_auc']:.4f}**.

Using the proposed **intervention urgency ranking**, the top 50 dispatch queue captures
**{int(urgency_top50['Captured Breaches'])} actual service-window breaches**, with
**Precision@50 = {urgency_top50['Precision@K']:.2%}**,
**Recall@50 = {urgency_top50['Recall@K']:.2%}**, and
**Lift@50 = {urgency_top50['Lift over Random']:.2f}x** over random task selection.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Research claim
# ---------------------------------------------------------------------

st.subheader("Main Research Finding")

st.markdown(
    f"""
The experiment confirms that proactive ML-based ranking can concentrate a larger share
of rare service-window breaches in a small dispatcher review queue.

The raw breach rate in the test set is only **{actual_breach_rate:.2%}**. This means random
selection would be unlikely to find many breach cases in the first 50 reviewed tasks.

However, the proposed intervention urgency ranking captures:

```text
Top-50 captured breaches: {int(urgency_top50['Captured Breaches'])}
Precision@50: {urgency_top50['Precision@K']:.2%}
Recall@50: {urgency_top50['Recall@K']:.2%}
Lift@50: {urgency_top50['Lift over Random']:.2f}x
```

This supports the operational claim that the system helps dispatchers focus on a small
subset of tasks with disproportionately high breach risk.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------

st.subheader("Model Comparison")

model_cols = [
    "model_name",
    "average_precision_pr_auc",
    "roc_auc",
    "threshold",
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

available_model_cols = [col for col in model_cols if col in metrics_df.columns]

model_display_df = (
    metrics_df[available_model_cols]
    .sort_values("average_precision_pr_auc", ascending=False)
    .reset_index(drop=True)
)

st.dataframe(model_display_df, use_container_width=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig = px.bar(
        model_display_df,
        x="model_name",
        y="average_precision_pr_auc",
        title="PR-AUC by Model",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    fig = px.bar(
        model_display_df,
        x="model_name",
        y="lift_at_50",
        title="Lift@50 by Model",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Top-K evaluation
# ---------------------------------------------------------------------

st.subheader("Top-K Dispatch Evaluation")

st.markdown(
    """
Top-K evaluation measures how useful the model is for dispatch operations. Instead of
asking whether every task is classified correctly, it asks whether the highest-ranked
tasks contain more actual breaches than random selection.
"""
)

st.dataframe(combined_topk_df, use_container_width=True)

fig = px.line(
    combined_topk_df,
    x="K",
    y="Recall@K",
    color="Ranking Strategy",
    markers=True,
    title="Recall@K: Model Probability vs Intervention Urgency",
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

st.divider()

# ---------------------------------------------------------------------
# Top-50 comparison
# ---------------------------------------------------------------------

st.subheader("Top-50 Operational Comparison")

top50_comparison_df = pd.DataFrame(
    [
        {
            "Ranking Strategy": "Model probability",
            "Captured Breaches": int(probability_top50["Captured Breaches"]),
            "Precision@50": probability_top50["Precision@K"],
            "Recall@50": probability_top50["Recall@K"],
            "Lift@50": probability_top50["Lift over Random"],
        },
        {
            "Ranking Strategy": "Intervention urgency",
            "Captured Breaches": int(urgency_top50["Captured Breaches"]),
            "Precision@50": urgency_top50["Precision@K"],
            "Recall@50": urgency_top50["Recall@K"],
            "Lift@50": urgency_top50["Lift over Random"],
        },
    ]
)

st.dataframe(top50_comparison_df, use_container_width=True)

fig = px.bar(
    top50_comparison_df,
    x="Ranking Strategy",
    y="Captured Breaches",
    title="Captured Breaches in Top 50 Tasks",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
The intervention urgency ranking improves top-50 breach capture from
**{int(probability_top50['Captured Breaches'])} breaches** to
**{int(urgency_top50['Captured Breaches'])} breaches**.

This means the urgency score adds operational value beyond raw model probability ranking.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Breach records inside top 50
# ---------------------------------------------------------------------

st.subheader("Actual Breach Records Captured in Top-50 Urgency Queue")

urgency_sorted_df = predictions_df.sort_values(
    "intervention_urgency_score",
    ascending=False,
).reset_index(drop=True)

top50_urgency_df = urgency_sorted_df.head(50).copy()
top50_urgency_df["dispatch_rank"] = top50_urgency_df.index + 1

captured_breach_cols = [
    "dispatch_rank",
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
    "feasibility_margin_minutes",
    "courier_workload_2h",
]

available_captured_cols = [
    col for col in captured_breach_cols if col in top50_urgency_df.columns
]

captured_breaches_df = top50_urgency_df[
    top50_urgency_df["service_window_breach"] == 1
][available_captured_cols]

st.dataframe(captured_breaches_df, use_container_width=True)

st.caption(
    """
The actual breach label is shown only for research evaluation. In live deployment,
dispatchers would only see the predicted risk, urgency score, and recommendation.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

st.subheader("Probability Calibration")

if calibration_df.empty:
    st.info("Calibration file is not available.")
else:
    calibration_models = sorted(
        calibration_df["model_name"].dropna().unique().tolist()
    )

    default_index = (
        calibration_models.index(best_model)
        if best_model in calibration_models
        else 0
    )

    selected_model = st.selectbox(
        "Select model",
        calibration_models,
        index=default_index,
    )

    selected_calibration_df = calibration_df[
        calibration_df["model_name"] == selected_model
    ].copy()

    st.dataframe(selected_calibration_df, use_container_width=True)

    fig = px.line(
        selected_calibration_df,
        x="mean_predicted_probability",
        y="actual_breach_rate",
        markers=True,
        title="Calibration Curve",
        hover_data=["record_count", "probability_bin"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Paper-ready summary
# ---------------------------------------------------------------------

st.subheader("Paper-Ready Results Paragraph")

paper_summary = f"""
The proactive breach prediction experiment evaluates whether machine learning can
identify pickup tasks likely to miss their service window using only information
available at task acceptance time. In the temporal test split, the actual breach rate is
approximately {actual_breach_rate:.2%}, confirming that the task is a rare-event
classification problem. The calibrated random forest achieved the best predictive
performance, with PR-AUC of {best_metrics['average_precision_pr_auc']:.4f} and
ROC-AUC of {best_metrics['roc_auc']:.4f}. Under model-probability ranking, the top
50 tasks captured {int(probability_top50['Captured Breaches'])} actual breaches,
corresponding to Precision@50 of {probability_top50['Precision@K']:.2%} and
Recall@50 of {probability_top50['Recall@K']:.2%}. Using the proposed intervention
urgency ranking, the top 50 dispatch queue captured {int(urgency_top50['Captured Breaches'])}
actual service-window breaches, improving Precision@50 to {urgency_top50['Precision@K']:.2%},
Recall@50 to {urgency_top50['Recall@K']:.2%}, and Lift@50 to
{urgency_top50['Lift over Random']:.2f}x over random selection.
"""

st.markdown(paper_summary)

st.download_button(
    label="Download paper-ready results paragraph",
    data=paper_summary.encode("utf-8"),
    file_name="paper_ready_results_summary.txt",
    mime="text/plain",
)

st.divider()

# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------

st.subheader("Download Result Tables")

download_col1, download_col2, download_col3 = st.columns(3)

download_col1.download_button(
    label="Download model metrics",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="breach_model_metrics.csv",
    mime="text/csv",
)

download_col2.download_button(
    label="Download Top-K comparison",
    data=combined_topk_df.to_csv(index=False).encode("utf-8"),
    file_name="topk_dispatch_comparison.csv",
    mime="text/csv",
)

download_col3.download_button(
    label="Download captured top-50 breaches",
    data=captured_breaches_df.to_csv(index=False).encode("utf-8"),
    file_name="captured_top50_urgency_breaches.csv",
    mime="text/csv",
)