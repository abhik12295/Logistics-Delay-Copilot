from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Research Methodology", page_icon="📘", layout="wide")

st.title("📘 Research Methodology")

st.markdown(
    """
# Research Objective

This project proposes a country-neutral explainable AI framework for logistics delay diagnosis.
The goal is not only to detect that a delivery is late, but to diagnose why it is late,
estimate how serious the exception is, and generate a planner-facing explanation.

# Core Pipeline

```text
Event Logs
→ Feature Engineering
→ Delay Detection
→ Weak Root-Cause Labeling
→ Operational Exception Severity
→ Explanation Generation
→ Corrective Action Recommendation
"""
)