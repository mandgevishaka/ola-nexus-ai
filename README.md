# Final Capstone — Ola Domain Support Agent (CrewAI)

**Track completed: Ola — Business Operations / Customer Support.**

This repository contains one small, end-to-end implementation of the 16 tasks in the capstone brief. It uses a deterministic `MOCK_LLM` for CrewAI and AutoGen so the graded execution needs no paid LLM account, API key, or outbound LLM request.

## Execution mode and local model

- `MOCK_LLM` is the only language-model mode used by the transcript runner.
- CrewAI telemetry is disabled before CrewAI is imported using `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true`.
- SentenceTransformers uses the free local `all-MiniLM-L6-v2` embedding model. The model must be present in the local Hugging Face cache before the offline run. `local_files_only=True` prevents the application from downloading it during execution.
- ChromaDB stores both indexes locally under `data/chroma/`.
- No paid account, API key, external LLM, or remote vector database is used.

## Part 1 — Dataset design choices

`dataset.py` deterministically generates 45 records using seed `42`.

| Design choice | Value |
|---|---|
| Record count | 45 |
| Category distribution | Equal coverage; 20% target per category (9 records each) |
| Status distribution | Equal coverage; 20% target per status (9 records each) |
| Escalation flag | Independent seeded Bernoulli draw with 20% probability |
| Actual escalated records | 7 of 45 (15.56%) |
| Resolution time range | 1.0–48.0 hours |
| `days_since_created` | Integer from 0 through 30 |
| Monetary amount range | Not applicable: the required ticket-record schema does not define an amount field |

Categories and statuses are created by cycling through the required vocabulary and shuffling with the seeded random generator. This guarantees at least three records per category and at least one per status without editing individual records. The resolution-time range allows routine requests to resolve quickly and more involved technical/product cases to take longer.

Run `python dataset.py` to print and validate the generated distributions.

## Knowledge base

The `knowledge_base/` directory contains the twelve required Markdown policy documents. Each document has three policy sentences and covers one required topic. `setup_kb.py` recreates these files if needed.

## RAG implementation and calibration

`rag/chunking.py` implements:

1. Fixed-size character chunks (220 characters) with 40-character overlap.
2. Sentence-based chunks.

`rag/indexer.py` embeds both sets using the local SentenceTransformers model and upserts them into two separate persistent ChromaDB collections: `ola_kb_fixed` and `ola_kb_sentences`.

The similarity threshold is not a tutorial constant. At runtime, the code measures top-1 cosine similarity for three in-scope and two deliberately out-of-scope calibration queries. It chooses the midpoint between the lowest in-scope value and highest out-of-scope value, and refuses to silently continue if those measured clusters overlap. The actual measured values and selected threshold are written to `reports/rag_calibration.json` and printed in the transcript. They depend on the installed embedding-model version and therefore must be generated in the same environment used for grading; they are not fabricated in this source README.

`rag/evaluation.py` compares both collections on the same five labelled queries. It maps retrieved chunks to parent document names, deduplicates them, and reports TP/FP/FN arithmetic plus document-level precision and recall for every query. The measured recommendation is printed by `run_transcripts.py`.

## Part 2 — CrewAI, memory, schema and guardrails

- `crew/crew_runner.py` constructs the Retrieval Agent, Lookup Agent and Response Composer and runs them through `Crew.kickoff()`.
- Only the Lookup Agent receives `check_support_ticket_status`; the Retrieval Agent receives only `rag_lookup`; the Composer receives no tools.
- `crew/mock_llm.py` extends CrewAI's `BaseLLM` and returns deterministic ReAct tool actions based on the declared tool argument schema. It ignores system-prompt examples when checking tool observations.
- `crew/tools.py` calculates the escalation score as `0.7 * escalated + 0.3 * ((30 - days_since_created) / 30)`. The recommendation threshold is the 80th percentile of the generated dataset's score distribution (approximately `0.2640` for seed 42).
- `crew/memory.py` uses LangChain `InMemoryChatMessageHistory` and `RunnableWithMessageHistory` for separate in-process sessions.
- `crew/schema.py` validates each returned crew response with Pydantic.
- `crew/guardrails.py` masks fixed-format phone numbers, detects prompt-injection phrases, and rejects answers not copied from retrieved context.

The brief explicitly places free-text names, addresses and payment details outside the fixed-format masking requirement; examples in this repository are fabricated.

## Part 3 — API and evaluation

`api/main.py` exposes:

- `POST /ask` — invokes the CrewAI response function with session memory.
- `POST /add-document` — validates a 2–5 sentence document, saves it and indexes it in both collections.
- `WS /ws/chat` — supports multi-turn session chat and catches `WebSocketDisconnect`.

Every HTTP request and WebSocket message writes one JSON object per line to `logs/requests.jsonl`. The query is masked before it is written. Each entry includes a trace ID and duration.

`tests/evaluation_suite.py` runs 15 queries: one for each of the twelve policy topics, plus PII, out-of-scope and prompt-injection cases. Its deterministic mock judge scores Accuracy, Grounding, Completeness and Safety per query and writes `reports/evaluation_report.json` with all four averages.

## Part 4 — Review, governance and cache

- `review/autogen_stage.py` uses a real AutoGen `RoundRobinGroupChat` with two agents and `max_turns=2`. The Final Editor is an `AssistantAgent` configured with `output_content_type=StructuredVerdict`; the team registers `StructuredMessage[StructuredVerdict]`. A local AutoGen replay model client supplies deterministic mock completions, so the review makes no network call. The transcript demonstrates an unchanged approval and a correction of an unsupported refund amount.
- `governance/controls.py` declares Medium risk, verifies least-autonomy tool access, and enforces a per-request token-estimate cap.
- `crew/crew_runner.py` caches the grounded-generation response by normalized effective query. A cache hit returns before another CrewAI kickoff, avoiding a repeated tool/grounding run.

## Setup and execution

Use Python 3.11 in a virtual environment.

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare the local embedding model once, while network access is available.
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Generate the required policy files and validate the dataset.
python setup_kb.py
python dataset.py

# Run all acceptance demonstrations and generate the transcript/reports.
python run_transcripts.py

# Start the API.
uvicorn api.main:app --reload
```

The transcript runner writes `transcripts/capstone_transcript.txt`. Generated reports are written under `reports/`, and request logs are written under `logs/`. The first full run builds the local ChromaDB indexes under `data/chroma/`.

API documentation is available at `http://127.0.0.1:8000/docs` after starting Uvicorn.

## Acceptance evidence

| Brief tasks | Evidence in repository |
|---|---|
| 1–2 | `dataset.py`, `knowledge_base/`, `setup_kb.py` |
| 3–5 | `rag/chunking.py`, `rag/indexer.py`, `rag/evaluation.py`, calibration/comparison reports |
| 6–10 | `crew/`, `crew/tools.py`, `crew/memory.py`, `crew/guardrails.py` |
| 11–12 | `api/main.py`, `logs/requests.jsonl` after execution |
| 13 | `tests/evaluation_suite.py`, `reports/evaluation_report.json` after execution |
| 14 | `review/autogen_stage.py`, approval/revision transcript |
| 15–16 | `governance/controls.py`, cache and budget transcript |

All generated transcripts and similarity values must be produced by running the project in the grading environment. Do not treat source-code presence alone as proof that a runtime acceptance test passed.
