from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Research Methodology",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Research Methodology")

st.markdown(
    """
# Proactive Service-Window Breach Prevention Methodology

This research develops a hybrid machine learning and GenAI framework for
**proactive service-window breach prevention** in last-mile pickup logistics.

The goal is to predict, at the time a courier accepts a pickup task, whether the task
is likely to miss its service window and whether dispatch should intervene.
"""
)

st.divider()

st.subheader("Research Problem")

st.markdown(
    """
Traditional delay analytics often explain failures after they occur. However, in real
dispatch operations, the more valuable question is:

> **Which active pickup tasks are likely to violate their service window, and what should dispatch do before the failure happens?**

This project treats service-window breach prevention as a rare-event prediction and
dispatch prioritization problem.
"""
)

st.divider()

st.subheader("Research Questions")

st.markdown(
    """
The study is guided by the following research questions:

1. **RQ1:** Can machine learning predict pickup service-window breaches using only information available at task acceptance time?
2. **RQ2:** Do distance-aware feasibility features improve breach prediction compared with time-only features?
3. **RQ3:** Can workload and historical courier/zone risk improve detection of rare service-window failures?
4. **RQ4:** Can an intervention urgency score improve top-K prioritization of high-risk logistics tasks for dispatch teams?
5. **RQ5:** Can a grounded GenAI copilot generate dispatcher-ready recommendations from structured model evidence?
"""
)

st.divider()

st.subheader("Dataset")

st.markdown(
    """
This project uses the public **Cainiao LaDe-P pickup logistics dataset**.

The dataset contains pickup task event records with:

- task/order identifiers,
- courier identifiers,
- city and region information,
- task acceptance time,
- pickup service-window start and end time,
- actual pickup completion time,
- geographic coordinates.

The raw public dataset is mapped into a standardized logistics event schema to support
feature engineering, model training, dashboard evaluation, and paper reproducibility.
"""
)

st.code(
    """
LaDe-P field mapping:

order_id           → package_id
courier_id         → courier_id
city               → city
region_id          → zone_id
accept_time        → accepted_time
time_window_start  → service_window_start_time
time_window_end    → promised_delivery_time
pickup_time        → completed_time
accept_gps_lat     → origin_lat
accept_gps_lng     → origin_lng
lat                → destination_lat
lng                → destination_lng
""",
    language="text",
)

st.divider()

st.subheader("Target Variable")

st.markdown(
    """
The prediction target is a binary service-window breach label.

A pickup task is labeled as a breach if the actual completion time occurs after the
promised service-window end time.
"""
)

st.code(
    """
service_window_breach = 1 if completed_time > promised_delivery_time else 0
""",
    language="text",
)

st.markdown(
    """
This creates a rare-event classification problem because only a small percentage of
pickup tasks violate their service window.
"""
)

st.divider()

st.subheader("Leakage Prevention")

st.markdown(
    """
The model is designed for proactive prediction. Therefore, it only uses information
available at the time the pickup task is accepted.

The following post-event fields are excluded from model features:

- pickup completion time,
- delay duration,
- delay category,
- actual breach label,
- root-cause label,
- severity score,
- post-event operational exception metrics.

This prevents the model from learning information that would not be available during
real-time dispatch.
"""
)

st.divider()

st.subheader("Proactive Feature Engineering")

st.markdown(
    """
The feature engineering step creates variables that describe time pressure, travel
feasibility, courier workload, and historical operational risk.
"""
)

st.code(
    """
Key proactive features:

time_to_window_start_minutes
time_to_window_end_minutes
service_window_length_minutes
accept_hour
accept_dayofweek
accept_month
distance_km
expected_travel_time_minutes
feasibility_margin_minutes
courier_workload_2h
time_pressure_score
workload_pressure_score
distance_feasibility_pressure_score
historical_courier_breach_rate
historical_zone_breach_rate
historical_city_breach_rate
""",
    language="text",
)

st.markdown(
    """
The central feasibility feature compares the remaining time before the service-window
deadline with the expected travel time.
"""
)

st.code(
    """
expected_travel_time_minutes = distance_km / expected_speed_kmph * 60

feasibility_margin_minutes =
    time_to_window_end_minutes - expected_travel_time_minutes
""",
    language="text",
)

st.divider()

st.subheader("Model Training")

st.markdown(
    """
The project evaluates multiple machine learning models for rare-event breach prediction.
A temporal train/test split is used so that earlier tasks are used for training and later
tasks are used for evaluation.
"""
)

st.code(
    """
Models evaluated:

1. Balanced logistic regression
2. Balanced random forest
3. Histogram gradient boosting
4. Calibrated logistic regression
5. Calibrated random forest
6. Calibrated histogram gradient boosting
""",
    language="text",
)

st.markdown(
    """
Calibration is included because dispatchers need risk estimates that are operationally
meaningful. In rare-event settings, even small calibrated probabilities may represent
high relative risk when the base breach rate is below 1%.
"""
)

st.divider()

st.subheader("Intervention Urgency Score")

st.markdown(
    """
The intervention urgency score is designed to rank tasks for dispatch review. It combines
the model's relative risk ranking with operational pressure signals.
"""
)

st.code(
    """
Intervention Urgency Score =
    0.65 × model_risk_rank_score
  + 0.20 × time_pressure_score
  + 0.10 × distance_feasibility_pressure_score
  + 0.05 × workload_pressure_score
""",
    language="text",
)

st.markdown(
    """
This ranking is intended to answer the operational question:

> Which pickup tasks should dispatch review first?
"""
)

st.divider()

st.subheader("Evaluation Strategy")

st.markdown(
    """
Because service-window breaches are rare, standard accuracy is not used as the main
evaluation metric. Instead, the project focuses on imbalance-aware and operationally
meaningful metrics.
"""
)

st.code(
    """
Primary evaluation metrics:

PR-AUC
ROC-AUC
Precision@K
Recall@K
Lift@K
Top-K breach capture
Probability calibration
Intervention queue effectiveness
""",
    language="text",
)

st.markdown(
    """
Top-K metrics are especially important because dispatchers usually cannot inspect every
active task. A useful system should concentrate more actual breaches in the first 50 or
100 ranked tasks than random selection.
"""
)

st.divider()

st.subheader("Current Experimental Result")

st.markdown(
    """
The current best model is the **calibrated random forest**.

Current validated result:

- **PR-AUC:** 0.1353
- **ROC-AUC:** 0.7819
- **Model Precision@50:** 6.00%
- **Model Recall@50:** 27.27%
- **Model Lift@50:** 10.91x

Using the proposed intervention urgency ranking:

- **Top-50 captured breaches:** 4
- **Urgency Precision@50:** 8.00%
- **Urgency Recall@50:** 36.36%
- **Urgency Lift@50:** 14.55x
"""
)

st.divider()

st.subheader("Hybrid GenAI Extension")

st.markdown(
    """
The next stage adds a grounded GenAI recommendation module.

The GenAI component will not replace the predictive model. Instead, it will translate
structured model evidence into dispatcher-friendly recommendations.

For example, the GenAI copilot may summarize:

- why a task is high risk,
- which operational factors contributed to the risk,
- what dispatch should do next,
- and whether the recommendation is supported by model evidence.

A local open-source model such as Ollama can be used for reproducible offline research
experiments.
"""
)

st.divider()

st.subheader("Methodology Summary")

st.code(
    """
Cainiao LaDe-P pickup event logs
        ↓
Standardized logistics event schema
        ↓
Proactive feature engineering
        ↓
Temporal train/test split
        ↓
Calibrated ML breach prediction
        ↓
Predicted breach probability
        ↓
Intervention urgency ranking
        ↓
Top-K dispatch evaluation
        ↓
Grounded GenAI dispatcher recommendation
""",
    language="text",
)