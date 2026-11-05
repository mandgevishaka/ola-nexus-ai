"""The ticket lookup tool and its dataset-derived escalation score."""

from __future__ import annotations

from typing import Any

import numpy as np

from dataset import SUPPORT_TICKETS

# The threshold is derived from this generated dataset, not selected by guess.
_ESCALATION_SCORES = [
    0.7 * float(ticket["escalated"])
    + 0.3 * (1.0 - int(ticket["days_since_created"]) / 30.0)
    for ticket in SUPPORT_TICKETS
]
ESCALATION_THRESHOLD = float(np.percentile(_ESCALATION_SCORES, 80))


def check_support_ticket_status(record_id: str) -> dict[str, Any]:
    """Look up a ticket and calculate its designed escalation score.

    Formula: 0.7 * escalated + 0.3 * recency, where recency is
    (30 - days_since_created) / 30. A recently created ticket therefore has
    a larger recency contribution. The recommendation threshold is the 80th
    percentile of scores in this seeded dataset.
    """
    ticket = next((row for row in SUPPORT_TICKETS if row["record_id"] == record_id), None)
    if ticket is None:
        return {"record_id": record_id, "error": "Ticket not found."}

    recency = (30 - int(ticket["days_since_created"])) / 30.0
    score = round(0.7 * float(ticket["escalated"]) + 0.3 * recency, 4)
    return {
        "record_id": ticket["record_id"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "escalation_score": score,
        "escalation_threshold_80th_percentile": round(ESCALATION_THRESHOLD, 4),
        "recommended_escalation": score >= ESCALATION_THRESHOLD,
    }

