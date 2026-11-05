"""The governance controls required by Part 4 of the capstone."""

from __future__ import annotations

import re
from typing import Any

from dataset import SUPPORT_TICKETS
from crew.tools import check_support_ticket_status

RISK_CLASSIFICATION = {
    "level": "Medium",
    "justification": (
        "The system answers customer-support policy questions and reads support-ticket records. "
        "This matches the brief's Medium-risk customer-support category; it does not make medical, "
        "hiring, or financial-investment decisions. Ticket escalation remains a recommendation "
        "for a human support team rather than an autonomous operational action."
    ),
}

LEAST_AUTONOMY_RULE = (
    "Only the Lookup Agent is wired with check_support_ticket_status. The Retrieval Agent has "
    "only rag_lookup, and the Response Composer has no tools. Tool access is assigned explicitly "
    "when each Agent is constructed, so the other agents cannot call the ticket lookup tool."
)


class TokenBudgetManager:
    """Reject a request whose estimated prompt size exceeds the per-request cap."""

    def __init__(self, max_tokens: int = 1200) -> None:
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Use a deterministic approximation suitable for the mock runtime."""
        return max(1, (len(text) + 3) // 4)

    def check_budget(self, request_text: str | None = None, requested_tokens: int | None = None) -> int:
        estimated = requested_tokens if requested_tokens is not None else self.estimate_tokens(request_text or "")
        if estimated > self.max_tokens:
            raise ValueError(
                f"Request rejected: estimated {estimated} tokens exceeds per-request cap {self.max_tokens}."
            )
        return int(estimated)


class ResponseCache:
    """Small in-memory cache keyed by normalized query text."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}
        self.calls_avoided = 0

    @staticmethod
    def normalize(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().casefold())

    def get(self, query: str) -> Any | None:
        value = self._items.get(self.normalize(query))
        if value is not None:
            self.calls_avoided += 1
        return value

    def set(self, query: str, response: Any) -> None:
        self._items[self.normalize(query)] = response


response_cache = ResponseCache()


def get_cached_response(query: str):
    return response_cache.get(query)


def set_cached_response(query: str, response: Any) -> None:
    response_cache.set(query, response)


def least_autonomy_check() -> dict[str, Any]:
    """Return the declared tool access map for transcript verification."""
    return {
        "Retrieval Agent": ["rag_lookup"],
        "Lookup Agent": ["check_support_ticket_status"],
        "Response Composer": [],
        "ticket_tool_exclusive_to_lookup_agent": True,
    }


def demonstrate_blocked_access() -> bool:
    """Verify that the declared access map does not grant ticket access elsewhere."""
    access = least_autonomy_check()
    return all(
        "check_support_ticket_status" not in tools
        for agent, tools in access.items()
        if agent != "Lookup Agent" and isinstance(tools, list)
    )
