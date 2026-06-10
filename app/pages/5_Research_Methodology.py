from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Research Methodology", page_icon="📘", layout="wide")

st.title("📘 Research Methodology")

st.markdown(
    """
# Research Objective

This project proposes a country-neutral explainable AI framework for logistics delay diagnosis.
The goal is not only to detect that a logistics task is late, but to diagnose why it is late,
estimate how serious the exception is, and generate a planner-facing explanation with a recommended action.

The current app supports logistics service-task delay diagnosis, with pickup service delay used as the
first real public-data validation case.

---

# Core Pipeline

```text
Event Logs
→ Feature Engineering
→ Delay Detection
→ Weak Root-Cause Labeling
→ Operational Exception Severity
→ Explanation Generation
→ Corrective Action Recommendation

# Baseline Machine Learning Evaluation

The MVP includes a baseline machine learning classifier trained on weak-labeled
root-cause categories. The classifier uses event-gap features, distance-aware
execution features, workload pressure, event abnormality, delay category, and
Operational Exception Severity.

This allows the research to compare:

```text
Rule-based diagnosis
vs.
Machine learning root-cause classification
vs.
GenAI-style grounded explanation


# Dataset Standardization Layer

The project uses a standard event-log schema so the framework can remain
country-neutral and dataset-flexible. Raw datasets are first mapped into common
columns such as package ID, courier/carrier ID, assigned time, accepted time,
pickup/task start time, completion time, promised delivery time, workload, and
origin/destination coordinates.

This allows the same diagnosis and severity pipeline to work across public
last-mile datasets, enterprise TMS-style datasets, or future benchmark datasets.
"""


)