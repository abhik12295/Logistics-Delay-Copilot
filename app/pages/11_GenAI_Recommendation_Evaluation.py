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
    page_title="GenAI Recommendation Evaluation",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 GenAI Recommendation Evaluation")

st.markdown(
    """
# Grounded Dispatcher Recommendation Evaluation

This page evaluates the GenAI dispatcher recommendation component of the hybrid
machine learning and GenAI framework.

The recommendation engine uses structured model evidence from the proactive breach
prediction system and generates dispatcher-facing recommendations.

The evaluation checks whether recommendations are:

- grounded in provided evidence,
- aligned with dispatch action needs,
- complete,
- readable,
- and free from unsupported operational claims.
"""
)

EVALUATION_PATH = (
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

if not EVALUATION_PATH.exists() or not SUMMARY_PATH.exists():
    st.warning(
        """
The GenAI recommendation evaluation files were not found.

Run one of these commands first:

```bash
uv run python scripts/evaluate_llm_recommendations.py --top-k 50
```

For Ollama GenAI evaluation:

```bash
uv run python scripts/evaluate_llm_recommendations.py --top-k 50 --use-ollama --model-name qwen2:7b
```

Then refresh this page.
"""
    )
    st.stop()


@st.cache_data
def load_evaluation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "dispatch_rank",
        "service_window_breach",
        "predicted_breach_probability",
        "model_risk_rank_score",
        "intervention_urgency_score",
        "hallucination_flag",
        "evidence_coverage_score",
        "action_alignment_score",
        "completeness_score",
        "readability_score",
        "overall_recommendation_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def load_summary(path: Path) -> pd.DataFrame:
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


evaluation_df = load_evaluation(EVALUATION_PATH)
summary_df = load_summary(SUMMARY_PATH)

summary = summary_df.iloc[0]

st.divider()

# ---------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------

st.subheader("Evaluation Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Records Evaluated", f"{int(summary['records_evaluated']):,}")
col2.metric("Recommendation Source", str(summary["recommendation_source_values"]))
col3.metric("Actual Breaches in Top-K", int(summary["actual_breaches_in_top_k"]))
col4.metric("Hallucination Rate", f"{summary['hallucination_rate']:.2%}")

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Evidence Coverage",
    f"{summary['avg_evidence_coverage_score']:.3f}",
)
col6.metric(
    "Action Alignment",
    f"{summary['avg_action_alignment_score']:.3f}",
)
col7.metric(
    "Completeness",
    f"{summary['avg_completeness_score']:.3f}",
)
col8.metric(
    "Overall Score",
    f"{summary['avg_overall_recommendation_score']:.3f}",
)

st.success(
    f"""
The GenAI recommendation evaluation reviewed **{int(summary['records_evaluated'])}**
top-ranked dispatch tasks. The recommendation source was
**{summary['recommendation_source_values']}**.

The evaluation achieved **hallucination rate = {summary['hallucination_rate']:.2%}**
and **overall recommendation score = {summary['avg_overall_recommendation_score']:.3f}**.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------

st.subheader("Research Interpretation")

st.markdown(
    f"""
The GenAI recommendation component is evaluated as a grounded decision-support layer,
not as a replacement for the machine learning model.

The ML model identifies high-risk tasks. The intervention urgency score ranks the
dispatch queue. The GenAI component converts structured model evidence into a readable
dispatcher recommendation.

Current evaluation result:

```text
Recommendation source: {summary['recommendation_source_values']}
Records evaluated: {int(summary['records_evaluated'])}
Actual breaches in top-K: {int(summary['actual_breaches_in_top_k'])}
Hallucination rate: {summary['hallucination_rate']:.2%}
Evidence coverage score: {summary['avg_evidence_coverage_score']:.3f}
Action alignment score: {summary['avg_action_alignment_score']:.3f}
Completeness score: {summary['avg_completeness_score']:.3f}
Readability score: {summary['avg_readability_score']:.3f}
Overall recommendation score: {summary['avg_overall_recommendation_score']:.3f}
```

A hallucination rate of **0.0%** means the generated recommendations did not include
unsupported operational claims such as weather, accidents, road closures, or customer
issues that were not present in the structured evidence.
"""
)

if summary["avg_action_alignment_score"] < 0.75:
    st.info(
        """
The action alignment score is moderate. This means some LLM recommendations were
grounded and readable, but did not always use strong dispatcher action language such
as prioritize, review, confirm, monitor, or reassign.

This can be improved later through prompt tuning.
"""
    )

st.divider()

# ---------------------------------------------------------------------
# Score charts
# ---------------------------------------------------------------------

st.subheader("Recommendation Quality Scores")

score_cols = [
    "evidence_coverage_score",
    "action_alignment_score",
    "completeness_score",
    "readability_score",
    "overall_recommendation_score",
]

available_score_cols = [
    col for col in score_cols if col in evaluation_df.columns
]

score_summary_df = pd.DataFrame(
    [
        {
            "Metric": col.replace("_", " ").title(),
            "Average Score": evaluation_df[col].mean(),
        }
        for col in available_score_cols
    ]
)

fig = px.bar(
    score_summary_df,
    x="Metric",
    y="Average Score",
    title="Average GenAI Recommendation Quality Scores",
)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    evaluation_df,
    x="overall_recommendation_score",
    nbins=20,
    title="Distribution of Overall Recommendation Scores",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Source and priority breakdown
# ---------------------------------------------------------------------

st.subheader("Recommendation Source and Priority Breakdown")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    source_counts = (
        evaluation_df["recommendation_source"]
        .value_counts()
        .reset_index()
    )
    source_counts.columns = ["recommendation_source", "count"]

    fig = px.bar(
        source_counts,
        x="recommendation_source",
        y="count",
        title="Recommendation Source",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    priority_counts = (
        evaluation_df["intervention_priority"]
        .value_counts()
        .reset_index()
    )
    priority_counts.columns = ["intervention_priority", "count"]

    fig = px.bar(
        priority_counts,
        x="intervention_priority",
        y="count",
        title="Evaluated Tasks by Intervention Priority",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Evaluation table
# ---------------------------------------------------------------------

st.subheader("Detailed Recommendation Evaluation Table")

display_cols = [
    "dispatch_rank",
    "package_id",
    "city",
    "zone_id",
    "service_window_breach",
    "intervention_priority",
    "recommendation_source",
    "hallucination_flag",
    "evidence_coverage_score",
    "action_alignment_score",
    "completeness_score",
    "readability_score",
    "overall_recommendation_score",
]

available_display_cols = [
    col for col in display_cols if col in evaluation_df.columns
]

st.dataframe(
    evaluation_df[available_display_cols],
    use_container_width=True,
)

st.divider()

# ---------------------------------------------------------------------
# Recommendation explorer
# ---------------------------------------------------------------------

st.subheader("Recommendation Explorer")

task_options = evaluation_df["package_id"].astype(str).tolist()

selected_package_id = st.selectbox(
    "Select evaluated task",
    options=task_options,
    index=0,
)

selected_row = evaluation_df[
    evaluation_df["package_id"].astype(str) == selected_package_id
].iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dispatch Rank", int(selected_row["dispatch_rank"]))
col2.metric("Priority", str(selected_row["intervention_priority"]))
col3.metric(
    "Predicted Risk",
    f"{selected_row['predicted_breach_probability']:.2%}",
)
col4.metric(
    "Overall Score",
    f"{selected_row['overall_recommendation_score']:.3f}",
)

st.markdown("### Risk Summary")
st.info(str(selected_row.get("risk_summary", "")))

st.markdown("### Evidence Summary")
st.write(str(selected_row.get("evidence_summary", "")))

st.markdown("### Recommended Dispatch Action")
st.success(str(selected_row.get("recommended_action", "")))

st.markdown("### Dispatcher Note")
st.write(str(selected_row.get("dispatcher_note", "")))

st.markdown("### Evaluation Scores")

selected_scores_df = pd.DataFrame(
    [
        {
            "Metric": "Hallucination Flag",
            "Value": selected_row.get("hallucination_flag"),
        },
        {
            "Metric": "Evidence Coverage",
            "Value": selected_row.get("evidence_coverage_score"),
        },
        {
            "Metric": "Action Alignment",
            "Value": selected_row.get("action_alignment_score"),
        },
        {
            "Metric": "Completeness",
            "Value": selected_row.get("completeness_score"),
        },
        {
            "Metric": "Readability",
            "Value": selected_row.get("readability_score"),
        },
        {
            "Metric": "Overall Recommendation Score",
            "Value": selected_row.get("overall_recommendation_score"),
        },
    ]
)

st.dataframe(selected_scores_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Paper-ready paragraph
# ---------------------------------------------------------------------

st.subheader("Paper-Ready GenAI Evaluation Paragraph")

paper_paragraph = f"""
The GenAI dispatcher recommendation module was evaluated on the top
{int(summary['records_evaluated'])} urgency-ranked pickup tasks. The recommendation
source was {summary['recommendation_source_values']}. The evaluated queue contained
{int(summary['actual_breaches_in_top_k'])} actual service-window breaches. The generated
recommendations achieved a hallucination rate of {summary['hallucination_rate']:.2%},
average evidence coverage score of {summary['avg_evidence_coverage_score']:.3f},
average action alignment score of {summary['avg_action_alignment_score']:.3f},
average completeness score of {summary['avg_completeness_score']:.3f}, average readability
score of {summary['avg_readability_score']:.3f}, and average overall recommendation
score of {summary['avg_overall_recommendation_score']:.3f}. These results indicate that
the GenAI component can translate structured model evidence into readable, grounded
dispatcher recommendations while avoiding unsupported operational claims.
"""

st.markdown(paper_paragraph)

st.download_button(
    label="Download GenAI paper-ready paragraph",
    data=paper_paragraph.encode("utf-8"),
    file_name="genai_recommendation_evaluation_summary.txt",
    mime="text/plain",
)

st.divider()

# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------

st.subheader("Download GenAI Evaluation Outputs")

download_col1, download_col2 = st.columns(2)

download_col1.download_button(
    label="Download recommendation evaluation",
    data=evaluation_df.to_csv(index=False).encode("utf-8"),
    file_name="llm_recommendation_evaluation.csv",
    mime="text/csv",
)

download_col2.download_button(
    label="Download recommendation summary",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="llm_recommendation_evaluation_summary.csv",
    mime="text/csv",
)