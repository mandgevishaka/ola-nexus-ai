"""Create and validate the deterministic Ola support-ticket dataset."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Product Defect",
    "General Inquiry",
]
STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
SEED = 42
RECORD_COUNT = 45


def generate_tickets(seed: int = SEED, count: int = RECORD_COUNT) -> list[dict[str, Any]]:
    """Generate balanced categories/statuses and seeded escalation flags.

    Resolution times range from 1 to 48 hours: simple account/billing queries
    can close quickly, while technical/product investigations can take longer.
    Category and status coverage is guaranteed by cycling their vocabularies;
    escalation flags are independent seeded 20% probability draws.
    """
    if count < 40:
        raise ValueError("The capstone requires at least 40 records.")

    rng = random.Random(seed)
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(count)]
    statuses = [STATUSES[i % len(STATUSES)] for i in range(count)]
    rng.shuffle(categories)
    rng.shuffle(statuses)

    tickets: list[dict[str, Any]] = []
    for index in range(count):
        tickets.append(
            {
                "record_id": f"TICK-{1001 + index}",
                "category": categories[index],
                "status": statuses[index],
                "resolution_time_hours": round(rng.uniform(1.0, 48.0), 1),
                "days_since_created": rng.randint(0, 30),
                "escalated": rng.random() < 0.20,
            }
        )
    return tickets


SUPPORT_TICKETS = generate_tickets()


def validate_dataset(tickets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Validate all structural requirements and return reproducible statistics."""
    records = tickets if tickets is not None else SUPPORT_TICKETS
    if len(records) < 40:
        raise ValueError(f"Expected at least 40 tickets; received {len(records)}.")

    category_counts = Counter(ticket["category"] for ticket in records)
    status_counts = Counter(ticket["status"] for ticket in records)
    escalation_count = sum(bool(ticket["escalated"]) for ticket in records)
    escalation_pct = 100.0 * escalation_count / len(records)

    missing_fields = {
        "record_id", "category", "status", "resolution_time_hours",
        "days_since_created", "escalated",
    }
    for ticket in records:
        missing = missing_fields.difference(ticket)
        if missing:
            raise ValueError(f"{ticket.get('record_id', 'Unknown record')} missing {sorted(missing)}")
        if not 0 <= int(ticket["days_since_created"]) <= 30:
            raise ValueError("days_since_created must be between 0 and 30.")
        if not isinstance(ticket["escalated"], bool):
            raise ValueError("escalated must be a boolean.")

    if any(category_counts[value] < 3 for value in CATEGORIES):
        raise ValueError(f"Category coverage failed: {dict(category_counts)}")
    if any(status_counts[value] < 1 for value in STATUSES):
        raise ValueError(f"Status coverage failed: {dict(status_counts)}")
    if not 10.0 <= escalation_pct <= 30.0:
        raise ValueError(
            f"Escalation rate {escalation_pct:.1f}% is outside 10%-30%. "
            "Change the seed/weights and regenerate; do not edit records manually."
        )

    return {
        "record_count": len(records),
        "category_counts": dict(category_counts),
        "status_counts": dict(status_counts),
        "escalated_count": escalation_count,
        "escalated_percentage": round(escalation_pct, 2),
        "seed": SEED,
        "resolution_time_range_hours": [1.0, 48.0],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_dataset(), indent=2))
