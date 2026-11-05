"""Write the twelve policy documents required by the capstone brief."""

from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"

# Each entry is intentionally short and self-contained so retrieval can return
# policy statements without needing to combine unrelated documents.
KB_TOPICS = {
    "ticket_priority_classification.md": (
        "Ticket-Priority Classification Rules",
        [
            "Support tickets are classified as Low, Medium, High, or Urgent.",
            "Priority is based on financial impact, safety implications, and the scope of service disruption.",
            "The intake system assigns an initial priority, and an authorised support agent may reclassify it when new evidence is available.",
        ],
    ),
    "sla_by_severity.md": (
        "SLA-by-Severity Policy",
        [
            "Service level agreements define the maximum initial response and resolution times for each severity.",
            "Urgent issues require an initial response within 15 minutes and resolution within 4 hours.",
            "High issues require a 1-hour response and 12-hour resolution; Medium issues have a 24-hour response target and Low issues have a 48-hour response target.",
        ],
    ),
    "escalation_matrix.md": (
        "Escalation Matrix",
        [
            "Tier 1 agents handle standard support questions and routine ticket updates.",
            "Complex technical or safety cases must be transferred to Tier 2 specialists within 2 hours.",
            "Executive escalation is triggered when a ticket exceeds its applicable SLA by more than 50 percent.",
        ],
    ),
    "refund_compensation_policy.md": (
        "Refund and Compensation Policy",
        [
            "Refunds or compensation may be considered for verified ride cancellations, billing discrepancies, or driver misconduct.",
            "A refund request must include the trip identifier and supporting evidence before it is reviewed.",
            "Refunds up to $50 may be approved automatically, while amounts above $50 require supervisor approval.",
        ],
    ),
    "customer_communication_channels.md": (
        "Customer Communication Channel Policy",
        [
            "Customer support communication must use in-app chat, registered email, or verified phone support.",
            "Agents must not use personal messaging applications or unencrypted channels to contact customers.",
            "Support interactions are retained in the approved system for audit purposes.",
        ],
    ),
    "business_hours_holiday_support.md": (
        "Business Hours and Holiday Support Policy",
        [
            "Critical safety and technical support is available 24 hours a day, including holidays.",
            "General billing and account enquiries are handled between 8:00 AM and 8:00 PM local time.",
            "Holiday staffing maintains emergency coverage, while response targets may be adjusted for non-critical enquiries.",
        ],
    ),
    "repeat_complaint_handling.md": (
        "Repeat Complaint Handling Policy",
        [
            "A complaint about a recurring issue from the same user within 30 days is flagged as a repeat complaint.",
            "Repeat complaints are routed to a senior support specialist rather than returned to the standard queue.",
            "The assigned specialist reviews the history and records the suspected root cause and follow-up action.",
        ],
    ),
    "service_credit_policy.md": (
        "Service Credit Policy",
        [
            "Service credits may be offered as goodwill or compensation for a minor service disruption.",
            "Credits are non-transferable, expire after 90 days, and cannot be exchanged directly for cash.",
            "The maximum service credit for one incident is $25.",
        ],
    ),
    "feedback_collection_process.md": (
        "Feedback Collection Process",
        [
            "Customers receive an automated feedback survey in the mobile application after a ticket is resolved.",
            "A rating below 3 stars creates a follow-up quality-assurance ticket automatically.",
            "The quality-assurance team reviews the follow-up ticket within 24 hours and records the outcome.",
        ],
    ),
    "vip_customer_handling.md": (
        "VIP Customer Handling Policy",
        [
            "VIP riders and high-volume corporate partners are routed through dedicated priority support queues.",
            "VIP tickets are handled by trained Tier 2 agents who can use the approved expedited compensation process.",
            "The applicable VIP service target is prioritised over the standard queue target.",
        ],
    ),
    "outage_communication_protocol.md": (
        "Outage Communication Protocol",
        [
            "During a major platform outage, status notifications are sent to affected users through push notifications and in-app banners.",
            "Support agents use approved status updates and standard response text while the incident is active.",
            "Agents must not speculate about the cause or recovery time when that information has not been confirmed.",
        ],
    ),
    "data_retention_policy.md": (
        "Data Retention Policy for Support Tickets",
        [
            "Customer support tickets, chat logs, and associated personal records are retained securely for 3 years.",
            "Access to retained records is limited to authorised support and compliance personnel.",
            "At the end of the retention period, records are anonymised or securely deleted according to the retention process.",
        ],
    ),
}


def create_knowledge_base() -> list[Path]:
    """Create the required Markdown files and return their paths."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, (title, sentences) in KB_TOPICS.items():
        # Keep each policy article within the brief's 2–5 sentence limit.
        if not 2 <= len(sentences) <= 5:
            raise ValueError(f"{filename} must contain 2–5 sentences.")
        content = f"# {title}\n\n" + " ".join(sentences) + "\n"
        path = KB_DIR / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    files = create_knowledge_base()
    print(f"Created {len(files)} policy documents in {KB_DIR}.")
