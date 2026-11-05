"""In-process, session-based conversation memory using LangChain history."""

from __future__ import annotations

from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory


class SessionMemory:
    """Wrap the support response function with per-session chat history."""

    def __init__(self, response_function) -> None:
        self.response_function = response_function
        self.histories: dict[str, InMemoryChatMessageHistory] = {}

        def respond(inputs: dict[str, Any]) -> dict[str, str]:
            # Convert the stored messages into concise context for the next turn.
            history_lines = []
            for message in inputs.get("history", []):
                role = "Customer" if message.type == "human" else "Assistant"
                history_lines.append(f"{role}: {message.content}")
            history_text = "\n".join(history_lines)
            result = self.response_function(inputs["input"], history_text=history_text)
            return {"output": result.model_dump_json()}

        chain = RunnableLambda(respond)
        self.chain = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="output",
        )

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Return the existing history or create a new isolated session."""
        if session_id not in self.histories:
            self.histories[session_id] = InMemoryChatMessageHistory()
        return self.histories[session_id]

    def ask(self, session_id: str, query: str) -> str:
        """Invoke the wrapped chain and return the assistant's response text."""
        result = self.chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        return str(result["output"])
