# Final Experiment Results

## Project Title

A Hybrid Machine Learning and Generative AI Framework for Proactive Service-Window
Breach Prevention in Last-Mile Logistics

---

## Dataset

The experiment uses the public Cainiao LaDe-P pickup logistics dataset.

The proactive prediction task is defined as:

```text
service_window_breach = 1 if completed_time > promised_delivery_time else 0
```

The model uses only information available at task acceptance time to avoid post-event
leakage.

---

## Proactive Breach Prediction Results

**Best model:**

```text
calibrated_random_forest
```

**Validated ML results:**

```text
PR-AUC:        0.1353
ROC-AUC:       0.7819
Precision:     16.67%
Recall:        27.27%
Precision@50:  6.00%
Recall@50:     27.27%
Lift@50:       10.91x
```

---

## Top-50 Dispatch Ranking Comparison

**Model probability ranking:**

```text
Captured breaches: 3
Precision@50:      6.00%
Recall@50:         27.27%
```

**Intervention urgency ranking:**

```text
Captured breaches: 4
Precision@50:      8.00%
Recall@50:         36.36%
Lift@50:           14.55x
```

---

## Intervention Urgency Score

```text
Intervention Urgency Score =
    0.65 × model_risk_rank_score
  + 0.20 × time_pressure_score
  + 0.10 × distance_feasibility_pressure_score
  + 0.05 × workload_pressure_score
```

---

## GenAI Recommendation Evaluation

The Ollama-based GenAI recommendation evaluation used:

```text
Model:                  qwen2:7b
Top-K evaluated tasks:  50
Recommendation source:  ollama
Actual breaches in top-K: 4
```

**Validated GenAI results:**

```text
Hallucination rate:             0.00%
Evidence coverage score:        0.828
Action alignment score:         0.620
Completeness score:             1.000
Readability score:              1.000
Overall recommendation score:   0.8534
```

---

## Main Research Claim

The calibrated random forest model can identify high-risk pickup tasks under a
rare-event setting. The proposed intervention urgency ranking improves operational
dispatch prioritization by capturing 4 actual breaches in the top 50 ranked tasks,
compared with 3 breaches using model probability alone.

The GenAI recommendation module translates structured model evidence into readable
dispatcher recommendations with zero detected unsupported operational claims in the
evaluated sample.

---

## Paper-Ready Result Statement

The calibrated random forest achieved the best predictive performance with PR-AUC of
0.1353 and ROC-AUC of 0.7819. Under model-probability ranking, the top 50 tasks
captured 3 actual breaches, corresponding to Precision@50 of 6.00% and Recall@50 of
27.27%. Using the proposed intervention urgency ranking, the top 50 dispatch queue
captured 4 actual service-window breaches, improving Precision@50 to 8.00%, Recall@50
to 36.36%, and Lift@50 to 14.55x over random selection. The GenAI dispatcher
recommendation module, evaluated using Ollama qwen2:7b on the top 50 urgency-ranked
tasks, achieved a hallucination rate of 0.00% and an overall recommendation quality
score of 0.8534.