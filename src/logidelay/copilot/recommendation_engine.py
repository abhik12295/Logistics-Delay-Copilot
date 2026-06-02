from __future__ import annotations


def recommend_action(root_cause: str, severity_class: str) -> str:
    """
    Map root cause and severity to a planner-facing corrective action.
    """
    if root_cause == "Event-data inconsistency":
        return "Verify missing or abnormal event timestamps before operational escalation."

    if root_cause == "Courier acceptance delay":
        return "Review courier assignment responsiveness and consider earlier task reassignment."

    if root_cause == "Pickup/task start delay":
        return "Check origin readiness, appointment timing, and task start compliance."

    if root_cause == "Workload pressure":
        return "Rebalance courier workload or redistribute tasks in the affected zone."

    if root_cause == "Route execution instability":
        return "Review route plan, service area density, and travel-time assumptions."

    if root_cause == "Delivery execution delay":
        return "Monitor delivery execution and notify stakeholders if service window impact is high."

    if root_cause == "No significant delay":
        return "No immediate action required."

    return "Review the delivery record and escalate if customer service impact is expected."