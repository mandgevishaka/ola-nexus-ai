"""Deterministic CrewAI BaseLLM used for the zero-key capstone run."""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Set telemetry controls before CrewAI creates or runs a Crew.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai.llms.base_llm import BaseLLM


class MockLLM(BaseLLM):
    """A small deterministic LLM adapter; it never makes network requests.

    Native tool calls are emitted from the actual OpenAI-style tool schemas
    supplied by CrewAI. This works with CrewAI 1.15.x's default AgentExecutor.
    The selection logic uses declared argument fields, not tool-name matching.
    """

    def __init__(self) -> None:
        super().__init__(model="ola-deterministic-mock", temperature=0.0)
        self.tool_results: dict[str, Any] = {}

    @staticmethod
    def _message_text(messages: Any) -> tuple[str, list[tuple[str, str]]]:
        """Read conversation messages while excluding system prompt templates."""
        if isinstance(messages, str):
            return messages, [("user", messages)]
        visible: list[tuple[str, str]] = []
        for message in messages or []:
            if isinstance(message, dict):
                role = str(message.get("role", "user")).lower()
                content = message.get("content", "")
            else:
                role = str(getattr(message, "role", "user")).lower()
                content = getattr(message, "content", "")
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            if role not in {"system", "developer"} and content:
                visible.append((role, str(content)))
        return "\n".join(content for _, content in visible), visible

    @staticmethod
    def _tool_definitions(tools: Any, available_functions: Any = None, messages: Any = None) -> list[dict[str, Any]]:
        """Normalize OpenAI-style tool definitions supplied by CrewAI."""
        normalized = []
        for item in tools or []:
            if isinstance(item, dict):
                function = item.get("function", item)
                schema = function.get("parameters", {}) if isinstance(function, dict) else {}
                normalized.append(
                    {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "required": schema.get("required", []),
                        "properties": schema.get("properties", {}),
                    }
                )
            else:
                args_schema = getattr(item, "args_schema", None)
                schema = args_schema.model_json_schema() if args_schema else {}
                normalized.append(
                    {
                        "name": getattr(item, "name", ""),
                        "description": getattr(item, "description", ""),
                        "required": schema.get("required", []),
                        "properties": schema.get("properties", {}),
                    }
                )
        # CrewAI may provide the callable map separately from JSON tool schemas.
        # Inspect callable signatures to recover their declared argument fields.
        for name, function in (available_functions or {}).items():
            if any(item["name"] == name for item in normalized):
                continue
            import inspect

            signature = inspect.signature(function)
            properties = {
                parameter: {"type": "string"}
                for parameter in signature.parameters
                if parameter not in {"self", "kwargs"}
            }
            normalized.append(
                {
                    "name": name,
                    "description": getattr(function, "__doc__", ""),
                    "required": list(properties),
                    "properties": properties,
                }
            )
        # Some CrewAI ReAct versions include the authorised tool signatures in
        # the system tool catalogue but do not pass a separate `tools` argument.
        # Read signatures only for schema discovery; never parse observations or
        # generated tool results from the system prompt.
        if not normalized:
            system_texts = []
            if isinstance(messages, str):
                system_texts.append(messages)
            else:
                for message in messages or []:
                    if isinstance(message, dict) and str(message.get("role", "")).lower() == "system":
                        system_texts.append(str(message.get("content", "")))
                    elif str(getattr(message, "role", "")).lower() == "system":
                        system_texts.append(str(getattr(message, "content", "")))
            for system_text in system_texts:
                for match in re.finditer(r"(?m)^\s*[-*]?\s*([A-Za-z_]\w*)\(([^)]*)\)", system_text):
                    name, signature_text = match.groups()
                    properties = {}
                    for argument in signature_text.split(","):
                        argument = argument.strip()
                        if ":" in argument:
                            argument_name, argument_type = argument.split(":", 1)
                            properties[argument_name.strip()] = {"type": argument_type.strip()}
                    if properties:
                        normalized.append(
                            {
                                "name": name,
                                "description": "Tool signature from CrewAI's authorised tool catalogue.",
                                "required": list(properties),
                                "properties": properties,
                            }
                        )
        return normalized

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        """Return a native tool-call payload or deterministic final text.

        CrewAI 1.15.x uses AgentExecutor by default. Its native tool path expects
        a list of function-call objects, not a ReAct string containing Action /
        Action Input / PAUSE.
        """
        text, visible_messages = self._message_text(messages)
        definitions = self._tool_definitions(tools, available_functions, messages)

        # If a tool has already returned an observation, pass that real result
        # back to the agent instead of reading the system prompt's example text.
        tool_observations = [] if isinstance(messages, str) else [
            content for role, content in visible_messages
            if role in {"tool", "function"} or content.lstrip().startswith("Observation:")
        ]
        if tool_observations:
            observation = tool_observations[-1]
            return observation.rsplit("Observation:", 1)[-1].strip()

        if definitions:
            # The agent receives only its authorised tool(s). Choose based on
            # the required parameter schema, not on a tool-name substring.
            definition = definitions[0]
            required_fields = set(definition["required"])
            if "record_id" in required_fields or "record_id" in definition["properties"]:
                match = re.search(r"TICK-\d+", text, flags=re.IGNORECASE)
                if not match:
                    return "No ticket identifier was supplied; no lookup was performed."
                arguments = {"record_id": match.group(0).upper()}
            else:
                query = self._extract_query(text)
                arguments = {"query": query}
            return [
                {
                    "id": "call_ola_mock_1",
                    "type": "function",
                    "function": {
                        "name": definition["name"],
                        "arguments": json.dumps(arguments),
                    },
                }
            ]

        # The Composer emits a JSON object matching CrewResponseSchema. It uses
        # only results recorded by the tools that actually ran in this kickoff.
        result_parts = []
        confidence = 0.5
        requires_escalation = False
        rag_result = self.tool_results.get("rag")
        if rag_result:
            result_parts.append(rag_result["answer"])
            confidence = max(0.0, min(1.0, float(rag_result["top1_similarity"])))
        ticket_result = self.tool_results.get("ticket")
        if ticket_result:
            if "error" in ticket_result:
                result_parts.append(ticket_result["error"])
            else:
                result_parts.append(
                    f"Ticket {ticket_result['record_id']} is {ticket_result['status']}. "
                    f"Resolution time: {ticket_result['resolution_time_hours']} hours. "
                    f"Escalation score: {ticket_result['escalation_score']:.4f}."
                )
                confidence = max(confidence, 0.95)
                requires_escalation = bool(ticket_result["recommended_escalation"])
        answer = " ".join(part for part in result_parts if part).strip()
        if not answer:
            answer = "I don't know based on the available Ola support policies."
        if rag_result and rag_result.get("fallback"):
            confidence = 0.0
        return json.dumps(
            {
                "answer": answer,
                "confidence": round(confidence, 4),
                "requires_escalation": requires_escalation,
            }
        )

    @staticmethod
    def _extract_query(text: str) -> str:
        """Extract the user query from the task instructions when available."""
        markers = ("Customer query:", "User query:", "Query:")
        for marker in markers:
            position = text.lower().rfind(marker.lower())
            if position >= 0:
                return text[position + len(marker):].strip().splitlines()[0]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else text

    def supports_function_calling(self) -> bool:
        """Use CrewAI's native tool-call path (supported by AgentExecutor)."""
        return True
