from __future__ import annotations

from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logidelay.utils.app_helpers import load_data_from_sidebar

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Data Overview")
st.markdown(
    """
This page summarizes the sample logistics event dataset used by the MVP.
The final research version will replace or extend this with public benchmark data.
"""
)

df = load_data_from_sidebar(str(ROOT_DIR))

st.subheader("Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{len(df):,}")
col2.metric("Couriers", f"{df['courier_id'].nunique():,}")
col3.metric("Cities", f"{df['city'].nunique():,}")
col4.metric("Zones", f"{df['zone_id'].nunique():,}")

st.divider()

left, right = st.columns(2)

with left:
    city_counts = df["city"].value_counts().reset_index()
    city_counts.columns = ["city", "count"]

    fig = px.bar(
        city_counts,
        x="city",
        y="count",
        text="count",
        title="Records by City",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    zone_counts = df["zone_id"].value_counts().reset_index()
    zone_counts.columns = ["zone_id", "count"]

    fig = px.bar(
        zone_counts,
        x="zone_id",
        y="count",
        text="count",
        title="Records by Zone",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Delay Distribution")

fig = px.histogram(
    df,
    x="delay_minutes",
    nbins=40,
    title="Distribution of Delay Minutes",
)
st.plotly_chart(fig, use_container_width=True)