# LogiDelay Copilot

Explainable AI for logistics delay diagnosis using event logs and Operational Exception Severity.

## Research Goal

This project builds a country-neutral AI logistics framework that detects delayed deliveries, diagnoses likely operational causes, assigns an Operational Exception Severity score, and generates planner-facing explanations and recommended actions.

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