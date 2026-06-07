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

from logidelay.diagnosis.classifier import train_evaluate_classifier
from logidelay.utils.app_helpers import load_data_from_sidebar

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation")
st.markdown(
    """
This page trains a baseline machine learning classifier to predict the weak-labeled
root cause of each logistics delay.

The goal is not to claim final production accuracy. The goal is to compare a
transparent rule-based diagnosis layer with a learnable ML baseline.
"""
)

df = load_data_from_sidebar(str(ROOT_DIR))

st.subheader("Root-Cause Label Distribution")

label_counts = (
    df["root_cause_label"]
    .value_counts()
    .rename_axis("root_cause_label")
    .reset_index(name="count")
)

fig = px.bar(
    label_counts,
    x="root_cause_label",
    y="count",
    text="count",
    title="Weak Root-Cause Label Distribution",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Train Baseline Classifier")

st.markdown(
    """
The baseline model uses event-gap features, distance-aware execution features,
workload features, severity score, and categorical context such as city and zone.
"""
)

if st.button("Train and Evaluate Model"):
    with st.spinner("Training baseline classifier..."):
        model, result = train_evaluate_classifier(df)

    st.success("Model training complete.")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{result.accuracy:.3f}")
    col2.metric("Target", result.target_column)

    st.subheader("Classification Report")
    st.code(result.classification_report_text, language="text")

    st.subheader("Confusion Matrix")
    st.dataframe(result.confusion_matrix_df, use_container_width=True)

    cm_long = (
        result.confusion_matrix_df
        .reset_index()
        .melt(id_vars="index", var_name="Predicted", value_name="Count")
        .rename(columns={"index": "Actual"})
    )

    fig = px.imshow(
        result.confusion_matrix_df,
        text_auto=True,
        title="Confusion Matrix Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click the button above to train and evaluate the baseline model.")