"""Two-agent AutoGen RoundRobinGroupChat review with structured Pydantic output."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.replay import ReplayChatCompletionClient


class StructuredVerdict(BaseModel):
    approved: bool = Field(description="Whether the original draft is approved unchanged.")
    final_answer: str = Field(description="Approved or corrected answer.")
    reason: str = Field(description="Short explanation for the decision.")


class PolicyComplianceReviewer(BaseChatAgent):
    """First agent: identify monetary claims not supported by supplied context."""

    def __init__(self) -> None:
        super().__init__(name="policy_reviewer", description="Checks draft claims against retrieved policy context.")

    @property
    def produced_message_types(self) -> Sequence[type]:
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token: CancellationToken) -> Response:
        payload = json.loads(messages[-1].content)
        draft = payload["draft"]
        context = payload["context"]
        unsupported = self._unsupported_claim(draft, context)
        review = {
            "draft": draft,
            "context": context,
            "unsupported": unsupported,
            "review_note": "Unsupported claim detected." if unsupported else "Draft is supported by context.",
        }
        return Response(chat_message=TextMessage(content=json.dumps(review), source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        return None

    @staticmethod
    def _unsupported_claim(draft: str, context: list[str]) -> bool:
        """Flag a dollar amount that does not appear in the retrieved policy text."""
        context_text = " ".join(context).casefold()
        amounts = re.findall(r"\$\s?\d+", draft)
        return any(amount.replace(" ", "") not in context_text.replace(" ", "") for amount in amounts)


def _mock_editor_verdict(draft: str, context: list[str]) -> StructuredVerdict:
    """Prepare the deterministic mock completion consumed by AutoGen's editor."""
    unsupported = PolicyComplianceReviewer._unsupported_claim(draft, context)
    if not unsupported:
        return StructuredVerdict(
            approved=True,
            final_answer=draft,
            reason="The draft is supported by the supplied policy context and is approved unchanged.",
        )

    # Replace the unsupported monetary claim with a sentence copied from policy.
    replacement = next(
        (
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", " ".join(context))
            if "$" in sentence
        ),
        "I cannot confirm that claim from the available policy context.",
    )
    return StructuredVerdict(
        approved=False,
        final_answer=replacement,
        reason=(
            "The draft contained a monetary claim not supported by the retrieved policy context; "
            "it was replaced with the supported statement."
        ),
    )


async def _run_review(draft: str, context: list[str]) -> StructuredVerdict:
    """Run the actual two-agent team; replay client keeps the editor offline."""
    verdict = _mock_editor_verdict(draft, context)
    model_client = ReplayChatCompletionClient(
        chat_completions=[verdict.model_dump_json()],
        model_info=ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            family=ModelFamily.UNKNOWN,
            structured_output=True,
        ),
    )
    reviewer = PolicyComplianceReviewer()
    editor = AssistantAgent(
        name="final_editor",
        model_client=model_client,
        system_message=(
            "Return the supplied policy review as the final verdict. Do not add facts. "
            "The response must conform to the StructuredVerdict schema."
        ),
        output_content_type=StructuredVerdict,
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[StructuredVerdict]],
    )
    try:
        result = await team.run(task=json.dumps({"draft": draft, "context": context}))
        final_message = result.messages[-1]
        # StructuredMessage.content is already a Pydantic object; validate it
        # again at the application boundary before returning it to the caller.
        return StructuredVerdict.model_validate(final_message.content)
    finally:
        await model_client.close()


def review_draft_answer(draft: str, context: str | list[str]) -> dict[str, Any]:
    """Synchronous wrapper used by the transcript runner."""
    context_list = [context] if isinstance(context, str) else context
    verdict = asyncio.run(_run_review(draft, context_list))
    return verdict.model_dump()
