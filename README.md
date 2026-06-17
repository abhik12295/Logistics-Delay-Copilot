# A Hybrid Machine Learning and Generative AI Framework for Proactive Service-Window Breach Prevention in Last-Mile Logistics

Proactive Dispatch Risk Copilot

- Dispatch teams need early warning for pickup tasks that are likely to miss their service window, so they can intervene before the failure happens.

## Research Goal

- Core contribution
- Predict pickup service-window breach risk using real public courier event logs.
- Engineer distance-aware, workload-aware, and time-pressure features.
- Create an intervention urgency score for dispatch prioritization.
- Use GenAI to generate grounded dispatcher recommendations.
- Evaluate against rule-based alerting and ML baselines using imbalanced classification and top-K operational metrics.

## Core Features

- Logistics event-log processing
- Delay detection
- Root-cause weak labeling
- Operational Exception Severity scoring
- GenAI-style explanation generation
- Streamlit public demo app

## Local Setup

```bash
uv sync
uv run streamlit run app/Home.py


## Standard Input Schema

LogiDelay Copilot uses a standard event-log schema.

Required columns:

```text
package_id
courier_id
city
zone_id
assigned_time
accepted_time
pickup_time
completed_time
promised_delivery_time
courier_workload_2h


## Research Output Pages

The Streamlit app includes research-oriented pages for:

- Data overview
- Delay diagnosis
- Operational Exception Severity dashboard
- Copilot explanation
- Model evaluation
- Dataset integration
- Research results summary

The Research Results page provides downloadable CSV tables that can support the experiment and results sections of the IEEE BigData paper.

# Public Dataset Integration Plan

## Purpose

This project uses a country-neutral standard event-log schema. Public logistics
datasets may use different column names, so we first map raw dataset columns into
the standard LogiDelay schema.

## Standard Schema

Required columns:

```text
package_id
courier_id
city
zone_id
assigned_time
accepted_time
pickup_time
completed_time
promised_delivery_time
courier_workload_2h

```markdown
## Public Dataset Integration

Raw public logistics datasets can be mapped into the standard LogiDelay schema
using:

```bash
uv run python scripts/prepare_public_dataset.py