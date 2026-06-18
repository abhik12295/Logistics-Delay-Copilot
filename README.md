# A Hybrid Machine Learning and Generative AI Framework for Proactive Service-Window Breach Prevention in Last-Mile Logistics

## Proactive Dispatch Risk Copilot

This repository contains a research prototype for **proactive service-window breach prevention** in last-mile pickup logistics.

Instead of only explaining delivery or pickup delays after they occur, the system predicts which active pickup tasks are likely to miss their service window and ranks them for dispatcher intervention.

The project is designed as an IEEE BigData 2026 research artifact combining:

* public courier event-log data,
* proactive machine learning,
* rare-event classification,
* calibrated risk prediction,
* operational dispatch prioritization,
* and grounded GenAI-style dispatcher recommendations.

---

## Research Problem

Dispatch teams often manage many active pickup tasks at the same time. Because service-window breaches are rare, manually identifying the few high-risk tasks is difficult.

This project addresses the following core research question:

> Can machine learning identify the small number of pickup tasks likely to miss their service window early enough for dispatchers to intervene?

The goal is not only to predict risk, but also to convert model output into a practical **ranked intervention queue**.

---

## Research Goal

The main goal is to build a hybrid machine learning and GenAI framework that can:

1. Predict pickup service-window breach risk using real public courier event logs.
2. Engineer proactive features available at task acceptance time.
3. Use distance-aware, workload-aware, and time-pressure features.
4. Train calibrated machine learning models for rare-event breach prediction.
5. Create an intervention urgency score for dispatch prioritization.
6. Evaluate performance using imbalanced classification and top-K operational metrics.
7. Generate grounded dispatcher recommendations using structured model evidence.

---

## Dataset

This project uses the public **Cainiao LaDe-P pickup logistics dataset**.

The LaDe-P dataset provides pickup-task event records, including:

* order/task identifiers,
* courier identifiers,
* city and region information,
* task acceptance time,
* pickup service-window start and end time,
* actual pickup completion time,
* pickup and courier GPS coordinates.

The raw public dataset is mapped into a standard logistics event schema for reproducible experimentation.

---

## Standardized Input Schema

The standardized schema used by the project includes:

```text
package_id
courier_id
city
zone_id
assigned_time
accepted_time
service_window_start_time
promised_delivery_time
pickup_time
completed_time
origin_lat
origin_lng
destination_lat
destination_lng
courier_workload_2h
```

For LaDe-P pickup data:

```text
accept_time        → accepted_time
time_window_start  → service_window_start_time
time_window_end    → promised_delivery_time
pickup_time        → completed_time
```

The service-window breach label is defined as:

```text
service_window_breach = 1 if completed_time > promised_delivery_time else 0
```

---

## Methodology Pipeline

```text
Cainiao LaDe-P pickup event logs
        ↓
Standardized logistics event schema
        ↓
Proactive breach feature engineering
        ↓
Temporal train/test split
        ↓
Calibrated machine learning models
        ↓
Predicted breach probability
        ↓
Intervention urgency score
        ↓
Top-K dispatch prioritization
        ↓
Grounded dispatcher recommendation
```

---

## Proactive Feature Engineering

The model only uses information available at task acceptance time. This avoids data leakage from future events.

Key proactive features include:

```text
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
```

The core feasibility feature is:

```text
feasibility_margin_minutes =
    time_to_window_end_minutes - expected_travel_time_minutes
```

This measures whether there is enough remaining time to complete the pickup before the service-window deadline.

---

## Machine Learning Models

The project evaluates multiple breach prediction models:

* balanced logistic regression,
* balanced random forest,
* histogram gradient boosting,
* calibrated logistic regression,
* calibrated random forest,
* calibrated histogram gradient boosting.

Because service-window breaches are rare, the project does not rely on accuracy. Instead, it uses imbalance-aware and operations-oriented metrics.

---

## Evaluation Metrics

The main evaluation metrics are:

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

Top-K metrics are especially important because dispatchers usually review only a small number of high-risk tasks.

---

## Current Experimental Results

The best-performing model is:

```text
calibrated_random_forest
```

Current model results:

```text
PR-AUC:        0.1353
ROC-AUC:       0.7819
Precision@50:  6.00%
Recall@50:    27.27%
Lift@50:      10.91x
```

The intervention urgency ranking improves the operational dispatch queue.

Top-50 comparison:

```text
Model probability ranking:
- Captured breaches: 3
- Precision@50: 6.00%
- Recall@50: 27.27%

Intervention urgency ranking:
- Captured breaches: 4
- Precision@50: 8.00%
- Recall@50: 36.36%
- Lift@50: 14.55x
```

This shows that combining model risk with operational pressure signals can help dispatchers focus on a small subset of pickup tasks with disproportionately high service-window breach risk.

---

## Intervention Urgency Score

The intervention urgency score combines model ranking with operational risk factors.

```text
Intervention Urgency Score =
    0.65 × model_risk_rank_score
  + 0.20 × time_pressure_score
  + 0.10 × distance_feasibility_pressure_score
  + 0.05 × workload_pressure_score
```

This score is designed for dispatch prioritization rather than pure probability ranking.

It answers the operational question:

> Which pickup tasks should dispatch review first?

---

## Streamlit App Pages

The Streamlit app includes the following research and demo pages:

```text
Home
Data Overview
Delay Diagnosis
Severity Dashboard
Copilot Explanation
Research Methodology
Model Evaluation
Dataset Integration
Research Results
Proactive Risk Copilot
Breach Model Evaluation
```

The most important proactive research pages are:

```text
Home
Proactive Risk Copilot
Breach Model Evaluation
Research Results
Dataset Integration
```

---

## Main Dashboard Outputs

The Home dashboard provides a research snapshot including:

```text
Best model
PR-AUC
ROC-AUC
Model Lift@50
Urgency Top-50 Breaches
Urgency Lift@50
Top-K dispatch ranking comparison
Top 20 intervention queue
```

The Breach Model Evaluation page provides:

```text
Model comparison
Best model metrics
PR-AUC and Lift@50 charts
Probability calibration
Confusion matrix
Top-K dispatch prioritization
Model probability vs intervention urgency comparison
Paper-ready result summary
```

---

## Local Setup

Install dependencies:

```bash
uv sync
```

Prepare the proactive breach dataset:

```bash
uv run python scripts/prepare_breach_dataset.py
```

Train the breach prediction model:

```bash
uv run python scripts/train_breach_model.py
```

Run the Streamlit app:

```bash
uv run streamlit run app/home.py
```

---

## Key Output Files

```text
data/processed/lade_p_proactive_breach_sample.csv
data/processed/lade_p_breach_model_predictions_with_urgency.csv
data/processed/lade_p_breach_model_calibration.csv

models/breach_prediction/best_breach_model.joblib
models/breach_prediction/breach_model_metrics.csv
models/breach_prediction/breach_model_predictions.csv
models/breach_prediction/breach_model_calibration.csv
```

---

## Validation Commands

Run this to validate the model and dashboard outputs:

```bash
python - <<'PY'
import pandas as pd

pred = pd.read_csv("data/processed/lade_p_breach_model_predictions_with_urgency.csv")

for score_col in ["predicted_breach_probability", "intervention_urgency_score"]:
    sorted_df = pred.sort_values(score_col, ascending=False).reset_index(drop=True)
    top50 = sorted_df.head(50)

    captured = int(top50["service_window_breach"].sum())
    total_breaches = int(sorted_df["service_window_breach"].sum())

    print(f"\nTop 50 by {score_col}")
    print(f"Captured breaches: {captured}")
    print(f"Precision@50: {captured / 50:.2%}")
    print(f"Recall@50: {captured / total_breaches:.2%}")
PY
```

Expected result:

```text
Top 50 by predicted_breach_probability
Captured breaches: 3
Precision@50: 6.00%
Recall@50: 27.27%

Top 50 by intervention_urgency_score
Captured breaches: 4
Precision@50: 8.00%
Recall@50: 36.36%
```

---

## Research Contribution

This project contributes:

1. A proactive service-window breach prediction framework for last-mile pickup logistics.
2. A standardized logistics event schema for public courier event-log data.
3. Distance-aware and workload-aware feature engineering for breach prevention.
4. Calibrated machine learning models for rare-event logistics prediction.
5. An intervention urgency score for operational dispatch prioritization.
6. Top-K evaluation using Precision@K, Recall@K, and Lift@K.
7. A foundation for grounded GenAI dispatcher recommendations.

---

## GenAI Extension

The next stage of the project adds a local open-source LLM, such as Ollama, to generate dispatcher recommendations from structured model evidence.

The GenAI component will not replace the machine learning model. Instead, it will translate model outputs into clear operational guidance.

Example output:

```text
Risk Summary:
This pickup task has elevated service-window breach risk due to limited remaining time and high workload pressure.

Recommended Action:
Prioritize dispatcher review, confirm courier availability, and consider reassignment if nearby capacity exists.

Evidence:
- High intervention urgency score
- Elevated model risk rank
- Tight feasibility margin
- Courier workload pressure
```

---

## Paper Title

```text
A Hybrid Machine Learning and Generative AI Framework for Proactive Service-Window Breach Prevention in Last-Mile Logistics
```

---

## Status

Current status:

```text
Proactive breach feature engineering: complete
Calibrated ML breach prediction: complete
Intervention urgency ranking: complete
Top-K operational evaluation: complete
Home dashboard update: complete
Breach Model Evaluation dashboard: complete
GenAI/Ollama recommendation module: planned
```
