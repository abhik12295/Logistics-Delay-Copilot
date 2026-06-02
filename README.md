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