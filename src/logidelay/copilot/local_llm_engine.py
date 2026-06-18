from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMRecommendation:
    """Structured output from the local LLM recommendation engine."""

    risk_summary: str
    evidence_summary: str
    recommended_action: str
    dispatcher_note: str
    confidence: str
    source: str = "ollama"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert values to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def build_dispatch_prompt(task_evidence: dict[str, Any]) -> str:
    """
    Build a grounded prompt for dispatcher recommendation generation.

    The prompt forces the LLM to use only the provided structured evidence and to avoid
    unsupported claims.
    """

    package_id = task_evidence.get("package_id", "unknown")
    city = task_evidence.get("city", "unknown")
    zone_id = task_evidence.get("zone_id", "unknown")
    intervention_priority = task_evidence.get("intervention_priority", "unknown")

    predicted_probability = _safe_float(
        task_evidence.get("predicted_breach_probability")
    )
    model_risk_rank_score = _safe_float(
        task_evidence.get("model_risk_rank_score")
    )
    intervention_urgency_score = _safe_float(
        task_evidence.get("intervention_urgency_score")
    )
    time_to_window_end_minutes = _safe_float(
        task_evidence.get("time_to_window_end_minutes")
    )
    distance_km = _safe_float(task_evidence.get("distance_km"))
    expected_travel_time_minutes = _safe_float(
        task_evidence.get("expected_travel_time_minutes")
    )
    feasibility_margin_minutes = _safe_float(
        task_evidence.get("feasibility_margin_minutes")
    )
    courier_workload_2h = _safe_float(
        task_evidence.get("courier_workload_2h")
    )
    time_pressure_score = _safe_float(
        task_evidence.get("time_pressure_score")
    )
    distance_feasibility_pressure_score = _safe_float(
        task_evidence.get("distance_feasibility_pressure_score")
    )
    workload_pressure_score = _safe_float(
        task_evidence.get("workload_pressure_score")
    )

    prompt = f"""
You are a logistics dispatch copilot.

Your task is to generate a short, factual dispatcher recommendation for a pickup task
that may miss its service window.

Use ONLY the evidence provided below. Do not invent facts. Do not mention information
that is not present in the evidence. Keep the recommendation operational and concise.

Task evidence:
- Package ID: {package_id}
- City: {city}
- Zone ID: {zone_id}
- Intervention priority: {intervention_priority}
- Predicted breach probability: {predicted_probability:.4f}
- Model risk rank score: {model_risk_rank_score:.4f}
- Intervention urgency score: {intervention_urgency_score:.4f}
- Time to service-window end: {time_to_window_end_minutes:.2f} minutes
- Distance estimate: {distance_km:.2f} km
- Expected travel time: {expected_travel_time_minutes:.2f} minutes
- Feasibility margin: {feasibility_margin_minutes:.2f} minutes
- Courier workload in 2-hour window: {courier_workload_2h:.0f} tasks
- Time pressure score: {time_pressure_score:.4f}
- Distance feasibility pressure score: {distance_feasibility_pressure_score:.4f}
- Workload pressure score: {workload_pressure_score:.4f}

Return your answer as valid JSON only with these exact keys:
{{
  "risk_summary": "...",
  "evidence_summary": "...",
  "recommended_action": "...",
  "dispatcher_note": "...",
  "confidence": "low|medium|high"
}}

Guidelines:
- risk_summary: one sentence explaining the risk level.
- evidence_summary: one sentence naming the strongest evidence.
- recommended_action: one dispatcher action.
- dispatcher_note: one short note a dispatcher can read quickly.
- confidence: choose low, medium, or high based only on the evidence strength.
"""
    return prompt.strip()


def parse_llm_json_response(text: str) -> LLMRecommendation:
    """
    Parse an LLM JSON response into a structured recommendation.

    If the response contains extra text, this attempts to extract the JSON object.
    """

    cleaned = text.strip()

    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        cleaned = cleaned[start:end]

    payload = json.loads(cleaned)

    return LLMRecommendation(
        risk_summary=str(payload.get("risk_summary", "")).strip(),
        evidence_summary=str(payload.get("evidence_summary", "")).strip(),
        recommended_action=str(payload.get("recommended_action", "")).strip(),
        dispatcher_note=str(payload.get("dispatcher_note", "")).strip(),
        confidence=str(payload.get("confidence", "medium")).strip().lower(),
        source="ollama",
    )


def generate_ollama_recommendation(
    task_evidence: dict[str, Any],
    model_name: str = "qwen2:7b",
    host: str = "http://localhost:11434",
    timeout_seconds: int = 30,
) -> LLMRecommendation:
    """
    Generate a dispatcher recommendation using a local Ollama model.

    This function calls the Ollama /api/generate endpoint. It raises an exception if
    Ollama is unavailable, the model is missing, or the response cannot be parsed.
    """

    prompt = build_dispatch_prompt(task_evidence)

    request_payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 350,
        },
    }

    request_data = json.dumps(request_payload).encode("utf-8")

    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not reachable. Make sure Ollama is running locally."
        ) from exc

    response_payload = json.loads(response_body)
    raw_text = response_payload.get("response", "")

    if not raw_text:
        raise RuntimeError("Ollama returned an empty response.")

    return parse_llm_json_response(raw_text)


def generate_rule_based_fallback(task_evidence: dict[str, Any]) -> LLMRecommendation:
    """
    Generate a deterministic fallback recommendation when Ollama is unavailable.
    """

    priority = str(task_evidence.get("intervention_priority", "Medium"))
    predicted_probability = _safe_float(
        task_evidence.get("predicted_breach_probability")
    )
    urgency = _safe_float(task_evidence.get("intervention_urgency_score"))
    model_rank = _safe_float(task_evidence.get("model_risk_rank_score"))
    time_remaining = _safe_float(task_evidence.get("time_to_window_end_minutes"))
    margin = _safe_float(task_evidence.get("feasibility_margin_minutes"))
    workload = _safe_float(task_evidence.get("courier_workload_2h"))

    risk_summary = (
        f"This pickup task is ranked as {priority.lower()} priority with an "
        f"intervention urgency score of {urgency:.3f}."
    )

    evidence_summary = (
        f"The model risk rank is {model_rank:.3f}, predicted breach probability is "
        f"{predicted_probability:.2%}, time remaining is {time_remaining:.1f} minutes, "
        f"feasibility margin is {margin:.1f} minutes, and courier workload is "
        f"{workload:.0f} tasks."
    )

    if priority in {"Critical", "High"}:
        recommended_action = (
            "Prioritize dispatcher review, confirm courier availability, and prepare "
            "reassignment if the task cannot be completed within the service window."
        )
        confidence = "high"
    elif priority == "Medium":
        recommended_action = (
            "Monitor the task and escalate if the feasibility margin decreases or "
            "courier workload increases."
        )
        confidence = "medium"
    else:
        recommended_action = (
            "Continue normal monitoring because the current evidence does not require "
            "immediate intervention."
        )
        confidence = "medium"

    dispatcher_note = (
        "Recommendation is generated from structured model evidence and operational "
        "pressure signals."
    )

    return LLMRecommendation(
        risk_summary=risk_summary,
        evidence_summary=evidence_summary,
        recommended_action=recommended_action,
        dispatcher_note=dispatcher_note,
        confidence=confidence,
        source="rule_based_fallback",
    )


def generate_dispatcher_recommendation(
    task_evidence: dict[str, Any],
    use_ollama: bool = False,
    model_name: str = "qwen2:7b",
    host: str = "http://localhost:11434",
) -> LLMRecommendation:
    """
    Generate a dispatcher recommendation.

    If use_ollama=True, the function tries the local Ollama model first. If Ollama fails,
    it automatically falls back to a deterministic rule-based recommendation.
    """

    if use_ollama:
        try:
            return generate_ollama_recommendation(
                task_evidence=task_evidence,
                model_name=model_name,
                host=host,
            )
        except Exception:
            return generate_rule_based_fallback(task_evidence)

    return generate_rule_based_fallback(task_evidence)