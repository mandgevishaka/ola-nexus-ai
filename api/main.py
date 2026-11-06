"""FastAPI deployment for the Ola support crew."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from crew.crew_runner import OlaSupportCrew
from crew.guardrails import detect_prompt_injection, mask_pii
from crew.memory import SessionMemory
from crew.schema import CrewResponseSchema
from governance.controls import TokenBudgetManager
from rag.indexer import RAGCore

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "requests.jsonl"

# Each log record is a single JSON object on one line for ELK-style ingestion.
logger = logging.getLogger("ola_support_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

app = FastAPI(
    title="Ola Domain Support Agent",
    version="1.0.0",
    description="CrewAI support-policy and ticket lookup API running in deterministic MOCK_LLM mode.",
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class AskResponse(BaseModel):
    trace_id: str
    source: str
    response: CrewResponseSchema


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=20, max_length=5000)


class DocumentResponse(BaseModel):
    trace_id: str
    message: str
    filename: str


_service: OlaSupportCrew | None = None
_memory: SessionMemory | None = None
_budget = TokenBudgetManager(max_tokens=1200)


def get_service() -> tuple[OlaSupportCrew, SessionMemory]:
    """Initialize the local embedding index and crew once per server process."""
    global _service, _memory
    if _service is None:
        rag = RAGCore()
        _service = OlaSupportCrew(rag=rag)
        _memory = SessionMemory(_service.run_query)
    assert _memory is not None
    return _service, _memory


def write_log(entry: dict) -> None:
    """Write exactly one JSONL event without raw phone-number PII."""
    logger.info(json.dumps(entry, ensure_ascii=False, default=str))


@app.middleware("http")
async def structured_request_logging(http_request: Request, call_next):
    """Log every HTTP request once, including validation and guardrail failures."""
    trace_id = str(uuid.uuid4())
    http_request.state.trace_id = trace_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(http_request)
        status_code = response.status_code
        return response
    finally:
        # Endpoints attach only masked request text to state. Invalid requests
        # still get a trace/timing record, but their unparsed body is not logged.
        write_log(
            {
                "trace_id": trace_id,
                "event": "http_request",
                "method": http_request.method,
                "path": http_request.url.path,
                "query": getattr(http_request.state, "masked_query", None),
                "title": getattr(http_request.state, "document_title", None),
                "status_code": status_code,
                "status": "success" if status_code < 400 else "failed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )


@app.post("/ask", response_model=AskResponse)
def ask_support(http_request: Request, request: QueryRequest) -> AskResponse:
    """Mask and validate the request, enforce budget, then invoke the crew."""
    trace_id = http_request.state.trace_id
    masked_query = mask_pii(request.query)
    http_request.state.masked_query = masked_query
    try:
        if detect_prompt_injection(masked_query):
            raise HTTPException(status_code=400, detail="Prompt injection detected and blocked.")
        try:
            token_estimate = _budget.check_budget(request_text=masked_query)
        except ValueError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        _, memory = get_service()
        # SessionMemory wraps the actual crew response function with LangChain
        # InMemoryChatMessageHistory and returns its validated Pydantic JSON.
        response = CrewResponseSchema.model_validate_json(memory.ask(request.session_id, masked_query))
        return AskResponse(trace_id=trace_id, source="crew_or_cache", response=response)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Support request could not be processed.") from exc


@app.post("/add-document", response_model=DocumentResponse)
def add_document(http_request: Request, request: DocumentRequest) -> DocumentResponse:
    """Validate a 2–5 sentence policy article, persist it and index both chunks."""
    trace_id = http_request.state.trace_id
    http_request.state.document_title = mask_pii(request.title)
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", request.content.strip()) if part]
    if not 2 <= len(sentences) <= 5:
        raise HTTPException(status_code=422, detail="A policy document must contain 2–5 sentences.")

    try:
        service, _ = get_service()
        filename = re.sub(r"[^a-zA-Z0-9_-]", "_", request.title.strip()).strip("_") + ".md"
        service.rag.add_document(filename, f"# {request.title}\n\n{request.content}")
        return DocumentResponse(
            trace_id=trace_id,
            message="Policy document saved and indexed in both collections.",
            filename=filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document could not be indexed.") from exc

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Handle multi-turn chat, log each message, and survive client disconnects."""
    await websocket.accept()

    session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))
    connection_trace_id = str(uuid.uuid4())

    try:
        _, memory = get_service()

        while True:
            started = time.perf_counter()
            message_trace_id = str(uuid.uuid4())

            try:
                # Receive one message from the connected client.
                raw_query = await websocket.receive_text()

                # Mask PII before sending the query to the model or writing logs.
                masked_query = mask_pii(raw_query)

                if detect_prompt_injection(masked_query):
                    answer = "Request blocked by the prompt-injection guardrail."
                    status = "rejected"
                    error_type = None

                else:
                    try:
                        # Enforce the per-request token budget.
                        _budget.check_budget(request_text=masked_query)

                        # Use the same session memory across WebSocket messages.
                        answer = memory.ask(session_id, masked_query)
                        status = "success"
                        error_type = None

                    except ValueError:
                        answer = (
                            "Request rejected because it exceeds "
                            "the per-request token budget."
                        )
                        status = "rejected"
                        error_type = "TokenBudgetExceeded"

                    except Exception as exc:
                        # A processing failure should not terminate the connection.
                        answer = "Support request could not be processed. Please try again."
                        status = "error"
                        error_type = type(exc).__name__

                # Log one JSONL record for every received message.
                write_log(
                    {
                        "trace_id": message_trace_id,
                        "event": "websocket_message",
                        "session_id": session_id,
                        "query": masked_query,
                        "status": status,
                        "error_type": error_type,
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                    }
                )

                # Send the response without closing the connection.
                await websocket.send_text(
                    json.dumps(
                        {
                            "trace_id": message_trace_id,
                            "answer": answer,
                            "status": status,
                        }
                    )
                )

            except WebSocketDisconnect:
                # The client disconnected while receiving or sending a message.
                raise

    except WebSocketDisconnect:
        # Record a normal client disconnect.
        write_log(
            {
                "trace_id": connection_trace_id,
                "event": "websocket_disconnect",
                "session_id": session_id,
                "status": "disconnected",
            }
        )

    except Exception as exc:
        # Handle connection-level failures without crashing the API process.
        write_log(
            {
                "trace_id": connection_trace_id,
                "event": "websocket_error",
                "session_id": session_id,
                "error_type": type(exc).__name__,
                "status": "failed",
            }
        )

        try:
            await websocket.close(code=1011)
        except Exception:
            pass
