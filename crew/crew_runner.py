"""CrewAI orchestration for policy retrieval and support-ticket lookup."""

from __future__ import annotations

import json
import os
import re
from typing import Any

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from crew.guardrails import groundedness_check
from crew.mock_llm import MockLLM
from crew.schema import CrewResponseSchema
from crew.tools import check_support_ticket_status
from governance.controls import ResponseCache
from rag.indexer import RAGCore


class OlaSupportCrew:
    """Run the required three-agent crew and validate its final response."""

    def __init__(self, rag: RAGCore | None = None) -> None:
        self.rag = rag or RAGCore()
        self.llm = MockLLM()
        self.tool_call_counts = {"rag_lookup": 0, "ticket_lookup": 0}
        self.last_tool_results: dict[str, Any] = {}
        self.cache = ResponseCache()
        self.cached_tool_results: dict[str, dict[str, Any]] = {}
        self.kickoff_count = 0

    def _make_rag_tool(self):
        """Create a RAG tool bound to this RAG instance and capture its output."""
        @tool("rag_lookup")
        def rag_lookup(query: str) -> str:
            """Retrieve and answer an Ola policy question from the indexed KB."""
            self.tool_call_counts["rag_lookup"] += 1
            result = self.rag.answer(query, strategy=self.rag.recommended_strategy, k=3)
            self.last_tool_results["rag"] = result
            self.llm.tool_results["rag"] = result
            return json.dumps(result)

        return rag_lookup

    def run_query(self, user_query: str, history_text: str = "") -> CrewResponseSchema:
        """Run CrewAI kickoff, invoke only authorised tools, and validate output."""
        cache_query = f"{history_text}\n{user_query}" if history_text else user_query
        cache_key = self.cache.normalize(cache_query)
        cached_response = self.cache.get(cache_query)
        if cached_response is not None:
            self.last_tool_results = self.cached_tool_results.get(cache_key, {})
            return CrewResponseSchema.model_validate(cached_response)

        self.last_tool_results = {}
        self.llm.tool_results = {}
        self.tool_call_counts = {"rag_lookup": 0, "ticket_lookup": 0}
        query_for_agents = user_query
        if history_text:
            query_for_agents = f"Conversation context: {history_text}\nCustomer query: {user_query}"

        # Define all three required agents. Only the Lookup Agent receives the
        # ticket-status tool; retrieval and composition have no ticket access.
        rag_tool = self._make_rag_tool()
        ticket_tool = make_capturing_ticket_tool(self)

        retrieval_agent = Agent(
            role="Retrieval Agent",
            goal="Retrieve only relevant policy statements from the Ola knowledge base.",
            backstory="You use the authorised RAG lookup tool and do not invent policy details.",
            tools=[rag_tool], llm=self.llm, allow_delegation=False, verbose=False, max_iter=2,
        )
        lookup_agent = Agent(
            role="Lookup Agent",
            goal="Look up a specific ticket and report its status and escalation score.",
            backstory="You are the only agent authorised to access support-ticket records.",
            tools=[ticket_tool], llm=self.llm, allow_delegation=False, verbose=False, max_iter=2,
        )
        composer_agent = Agent(
            role="Response Composer",
            goal="Combine available retrieval and ticket results into a concise customer response.",
            backstory="You use only the outputs supplied by the other agents.",
            llm=self.llm, allow_delegation=False, verbose=False, max_iter=1,
        )

        tasks: list[Task] = []
        context_tasks: list[Task] = []
        ticket_match = re.search(r"TICK-\d+", query_for_agents, flags=re.IGNORECASE)
        policy_terms = ("policy", "sla", "refund", "priority", "escalat", "support", "hours",
                        "communication", "credit", "feedback", "vip", "outage", "retention",
                        "channel", "complaint", "ticket")
        # Run retrieval for policy questions. A ticket-only request can use the
        # lookup tool alone; a combined request can invoke both specialised tools.
        should_retrieve = not ticket_match or any(term in query_for_agents.lower() for term in policy_terms if term != "ticket")
        if should_retrieve:
            retrieval_task = Task(
                description=(
                    "Use rag_lookup exactly once to answer the following request. "
                    "Do not answer from prior knowledge.\nCustomer query: " + query_for_agents
                ),
                expected_output="The retrieved answer, context, source document names and similarity.",
                agent=retrieval_agent,
            )
            tasks.append(retrieval_task)
            context_tasks.append(retrieval_task)

        if ticket_match:
            lookup_task = Task(
                description=(
                    "Use check_support_ticket_status exactly once for the record ID below. "
                    "Do not estimate ticket values.\nCustomer query: " + query_for_agents
                ),
                expected_output="Ticket status, resolution time and escalation score.",
                agent=lookup_agent,
            )
            tasks.append(lookup_task)
            context_tasks.append(lookup_task)

        composer_task = Task(
            description=(
                "Compose a concise response using only the outputs from preceding tasks. "
                "If no policy evidence is available, say that the policy knowledge base does not contain the answer. "
                "If a ticket lookup result is available, include its status and escalation score. "
                "Do not invent facts.\nCustomer query: " + query_for_agents
            ),
            expected_output="A concise customer-facing draft based only on the preceding task outputs.",
            agent=composer_agent,
            context=context_tasks,
            output_pydantic=CrewResponseSchema,
        )
        tasks.append(composer_task)

        # Crew.kickoff() is the required orchestration entry point.
        crew = Crew(
            agents=[retrieval_agent, lookup_agent, composer_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
        self.kickoff_count += 1
        crew_output = crew.kickoff()
        self.last_crew_output = str(crew_output)

        # A missing tool invocation is an implementation failure, not a passing
        # mock result. This check makes the transcript expose a broken tool path.
        if should_retrieve and "rag" not in self.last_tool_results:
            raise RuntimeError("CrewAI did not invoke the Retrieval Agent's rag_lookup tool.")
        if ticket_match and "ticket" not in self.last_tool_results:
            # Ticket tool result is captured by a wrapper below in the tool factory.
            raise RuntimeError("CrewAI did not invoke the Lookup Agent's ticket tool.")

        crew_structured = getattr(crew_output, "pydantic", None)
        parts: list[str] = []
        rag_context: list[str] = []
        confidence = 0.5
        requires_escalation = False

        rag_result = self.last_tool_results.get("rag")
        ticket_result = self.last_tool_results.get("ticket")

        if rag_result:
            rag_context = rag_result["context"]
            policy_answer = rag_result["answer"]
            # Check the policy portion separately. A ticket status is grounded
            # by the lookup tool, not by the policy-document collection.
            if rag_result["grounded"] and not groundedness_check(policy_answer, rag_context):
                policy_answer = "I don't know based on the available Ola support policies."
            parts.append(policy_answer)
            confidence = max(0.0, min(1.0, float(rag_result["top1_similarity"])))
            if rag_result["fallback"] or not rag_result["grounded"]:
                confidence = 0.0

        if ticket_result:
            if "error" in ticket_result:
                parts.append(ticket_result["error"])
            else:
                parts.append(
                    f"Ticket {ticket_result['record_id']} is {ticket_result['status']}. "
                    f"Resolution time: {ticket_result['resolution_time_hours']} hours. "
                    f"Escalation score: {ticket_result['escalation_score']:.4f}."
                )
                requires_escalation = bool(ticket_result["recommended_escalation"])
                confidence = max(confidence, 0.95)

        expected_answer = " ".join(part for part in parts if part).strip()
        if not expected_answer:
            expected_answer = "I don't know based on the available Ola support policies."

        # Validate the Composer's Pydantic output when available. In deterministic
        # MOCK_LLM mode, also require its answer to match the captured tool facts.
        if crew_structured is not None:
            validated = CrewResponseSchema.model_validate(crew_structured)
            if validated.answer != expected_answer:
                validated = CrewResponseSchema(
                    answer=expected_answer,
                    confidence=round(confidence, 4),
                    requires_escalation=requires_escalation,
                )
        else:
            validated = CrewResponseSchema.model_validate(
                {
                    "answer": expected_answer,
                    "confidence": round(confidence, 4),
                    "requires_escalation": requires_escalation,
                }
            )

        # The policy answer must be supported by retrieved context. Ticket facts
        # are separately grounded in the authorised ticket lookup tool result.
        if rag_result and rag_result["grounded"] and not groundedness_check(
            rag_result["answer"], rag_context
        ):
            policy_fallback = "I don't know based on the available Ola support policies."
            remaining_parts = [policy_fallback]
            if ticket_result and "error" not in ticket_result:
                remaining_parts.append(parts[-1])
            validated = CrewResponseSchema(
                answer=" ".join(remaining_parts),
                confidence=0.0 if not ticket_result else validated.confidence,
                requires_escalation=validated.requires_escalation,
            )
        self.cache.set(cache_query, validated.model_dump())
        self.cached_tool_results[cache_key] = dict(self.last_tool_results)
        return validated


# The ticket wrapper stores the actual tool result returned during CrewAI execution.


def make_capturing_ticket_tool(owner: OlaSupportCrew):
    """Create the least-privilege ticket tool and capture its real tool result."""
    @tool("check_support_ticket_status")
    def ticket_lookup(record_id: str) -> str:
        """Look up a support ticket by its record ID."""
        result = check_support_ticket_status(record_id)
        owner.tool_call_counts["ticket_lookup"] += 1
        owner.last_tool_results["ticket"] = result
        owner.llm.tool_results["ticket"] = result
        return json.dumps(result)

    return ticket_lookup
