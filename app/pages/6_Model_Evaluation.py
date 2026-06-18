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
    page_title="Model Evaluation",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Model Evaluation Overview")

st.markdown(
    """
# Proactive Breach Prediction Model Evaluation

This page provides a high-level overview of the machine learning evaluation strategy for
the proactive service-window breach prediction framework.

For detailed charts, calibration results, confusion matrix, and top-K dispatch analysis,
open the dedicated page:

> **Breach Model Evaluation**
"""
)

METRICS_PATH = ROOT_DIR / "models" / "breach_prediction" / "breach_model_metrics.csv"

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "lade_p_breach_model_predictions_with_urgency.csv"
)

if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists():
    st.warning(
        """
The breach model evaluation files were not found.

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
        "model_risk_rank_score",
        "intervention_urgency_score",
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
                "Lift over Random": lift,
            }
        )

    return pd.DataFrame(rows)


metrics_df = load_metrics(METRICS_PATH)
predictions_df = load_predictions(PREDICTIONS_PATH)

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

probability_top50 = probability_topk_df[probability_topk_df["K"] == 50].iloc[0]
urgency_top50 = urgency_topk_df[urgency_topk_df["K"] == 50].iloc[0]

st.divider()

st.subheader("Evaluation Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Tasks", f"{total_tasks:,}")
col2.metric("Actual Breaches", f"{total_breaches:,}", f"{actual_breach_rate:.2%}")
col3.metric("Best Model", str(best_model))
col4.metric("PR-AUC", f"{best_metrics['average_precision_pr_auc']:.4f}")

col5, col6, col7, col8 = st.columns(4)

col5.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")
col6.metric("Model Precision@50", f"{best_metrics['precision_at_50']:.2%}")
col7.metric("Model Recall@50", f"{best_metrics['recall_at_50']:.2%}")
col8.metric("Model Lift@50", f"{best_metrics['lift_at_50']:.2f}x")

st.success(
    f"""
The best-performing model is **{best_model}**, with **PR-AUC =
{best_metrics['average_precision_pr_auc']:.4f}** and **ROC-AUC =
{best_metrics['roc_auc']:.4f}**.

Using raw model probability ranking, the top 50 tasks capture
**{int(probability_top50['Captured Breaches'])} actual breaches**.

Using intervention urgency ranking, the top 50 dispatch queue captures
**{int(urgency_top50['Captured Breaches'])} actual breaches**, improving Recall@50 to
**{urgency_top50['Recall@K']:.2%}** and Lift@50 to
**{urgency_top50['Lift over Random']:.2f}x**.
"""
)

st.divider()

st.subheader("Why Accuracy Is Not the Main Metric")

st.markdown(
    f"""
Service-window breaches are rare in the temporal test split. The actual breach rate is:

```text
{actual_breach_rate:.2%}
```

Because of this imbalance, a model could achieve high accuracy by predicting almost all
tasks as non-breaches. That would not be operationally useful.

Therefore, this research focuses on:

```text
PR-AUC
ROC-AUC
Precision@K
Recall@K
Lift@K
Top-K breach capture
Probability calibration
Intervention queue effectiveness
```

These metrics better reflect whether the model can help dispatchers identify the small
number of risky tasks.
"""
)

st.divider()

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

st.subheader("Model Probability vs Intervention Urgency")

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
    """
The model probability ranking evaluates pure predictive risk. The intervention urgency
ranking adds operational context, including time pressure, feasibility pressure, and
workload pressure.

This makes the urgency ranking more suitable for dispatcher prioritization.
"""
)

st.divider()

st.subheader("Paper-Ready Evaluation Statement")

st.markdown(
    f"""
The calibrated random forest achieved the best model performance with **PR-AUC =
{best_metrics['average_precision_pr_auc']:.4f}** and **ROC-AUC =
{best_metrics['roc_auc']:.4f}**. Under model-probability ranking, the top 50 tasks
captured **{int(probability_top50['Captured Breaches'])}** actual breaches, corresponding
to **Precision@50 = {probability_top50['Precision@K']:.2%}** and **Recall@50 =
{probability_top50['Recall@K']:.2%}**. Using the intervention urgency ranking, the
top 50 dispatch queue captured **{int(urgency_top50['Captured Breaches'])}** actual
service-window breaches, improving **Precision@50 to {urgency_top50['Precision@K']:.2%}**,
**Recall@50 to {urgency_top50['Recall@K']:.2%}**, and **Lift@50 to
{urgency_top50['Lift over Random']:.2f}x** over random selection.
"""
)

st.divider()

st.info(
    """
For the complete evaluation dashboard, including probability calibration and confusion
matrix, open the **Breach Model Evaluation** page from the sidebar.
"""
)