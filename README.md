# Ola Domain Support Agent (CrewAI Implementation)

This repository contains an end-to-end implementation of an autonomous customer support and business operations agent system for the Ola domain using CrewAI. The system utilizes a deterministic `MOCK_LLM` engine for both CrewAI and AutoGen, allowing fully offline execution without requiring paid LLM accounts, API keys, or external network requests.

## Execution Mode & Local Dependencies

- **Mock LLM:** `MOCK_LLM` serves as the primary language model mode for transcript generation and agent tool execution.
- **Telemetry:** CrewAI telemetry is disabled prior to import via `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true`.
- **Embeddings:** SentenceTransformers uses the free local `all-MiniLM-L6-v2` embedding model. The model must be present in the local Hugging Face cache; `local_files_only=True` prevents runtime downloads.
- **Vector Storage:** ChromaDB stores both knowledge base indexes locally under `data/chroma/`.
- **Environment:** No paid accounts, external API keys, remote LLMs, or cloud vector databases are required.

---

## Dataset Design & Generation

`dataset.py` deterministically generates 45 synthetic support records using random seed `42`.

| Parameter | Value |
|---|---|
| Record count | 45 |
| Category distribution | Equal coverage; 20% target per category (9 records each) |
| Status distribution | Equal coverage; 20% target per status (9 records each) |
| Escalation flag | Independent seeded Bernoulli draw with 20% probability |
| Actual escalated records | 7 of 45 (15.56%) |
| Resolution time range | 1.0–48.0 hours |
| `days_since_created` | Integer from 0 through 30 |
| Monetary amount range | Not applicable (ticket-record schema does not define a monetary amount field) |

Categories and statuses are created by cycling through the required vocabulary and shuffling with the seeded random generator, guaranteeing at least three records per category and one per status. The resolution-time range allows routine requests to resolve quickly while complex technical or product cases take longer.

Run `python dataset.py` to print and validate the generated distributions.

---

## Knowledge Base

The `knowledge_base/` directory contains twelve Markdown policy documents covering the required operational topics. Each document contains three core policy sentences. `setup_kb.py` can be used to regenerate these files if needed.

---

## RAG Implementation & Calibration

`rag/chunking.py` implements two chunking strategies:
1. Fixed-size character chunks (220 characters with a 40-character overlap).
2. Sentence-based chunks.

`rag/indexer.py` embeds both sets using the local SentenceTransformers model and upserts them into two separate persistent ChromaDB collections: `ola_kb_fixed` and `ola_kb_sentences`.

The similarity threshold is dynamically computed at runtime. The system measures top-1 cosine similarity for three in-scope and two deliberately out-of-scope calibration queries, selecting the midpoint between the lowest in-scope value and highest out-of-scope value. If clusters overlap, the application halts rather than proceeding silently. Measured values and thresholds are written to `reports/rag_calibration.json`.

`rag/evaluation.py` evaluates both collections across five labelled queries, mapping retrieved chunks to parent documents, deduplicating them, and reporting TP/FP/FN metrics alongside document-level precision and recall.

---

## Multi-Agent Workflow, Memory & Guardrails

- **Crew Structure:** `crew/crew_runner.py` initializes the Retrieval Agent, Lookup Agent, and Response Composer, executing them through `Crew.kickoff()`.
- **Tool Allocation:** The Lookup Agent exclusively receives `check_support_ticket_status`, the Retrieval Agent receives `rag_lookup`, and the Composer operates without direct tool access.
- **Mock Engine:** `crew/mock_llm.py` extends CrewAI's `BaseLLM` to return deterministic ReAct tool actions based on declared tool argument schemas, ignoring system-prompt formatting when evaluating tool observations.
- **Scoring & Escalation:** `crew/tools.py` calculates escalation priority using the formula:
  $$\text{Score} = 0.7 \times \text{escalated} + 0.3 \times \left(\frac{30 - \text{days\_since\_created}}{30}\right)$$
  The recommendation threshold is set to the 80th percentile of the dataset's score distribution (approximately `0.2640` for seed 42).
- **Session Memory:** `crew/memory.py` utilizes LangChain's `InMemoryChatMessageHistory` and `RunnableWithMessageHistory` to manage isolated conversational sessions.
- **Schema Validation:** `crew/schema.py` validates all agent responses using Pydantic models.
- **Guardrails:** `crew/guardrails.py` masks fixed-format phone numbers, scans for prompt-injection patterns, and verifies that generated answers rely directly on retrieved context.

Free-text names, addresses, and payment details are treated as external to the fixed-format masking rules; inline examples in this repository are synthetic.

---

## API Service & Evaluation Suite

`api/main.py` provides a FastAPI application exposing the following endpoints:
- `POST /ask` — Invokes the multi-agent response pipeline with session memory support.
- `POST /add-document` — Validates (2–5 sentences), saves, and indexes new policy documents across both RAG collections.
- `WS /ws/chat` — Manages multi-turn WebSocket sessions with integrated disconnect handling.

Every HTTP request and WebSocket message writes a structured JSON log entry to `logs/requests.jsonl` (with queries masked) containing trace IDs and execution durations.

`tests/evaluation_suite.py` executes 15 test queries covering all twelve policy topics, plus PII, out-of-scope, and prompt-injection scenarios. A deterministic mock judge scores Accuracy, Grounding, Completeness, and Safety, outputting aggregated metrics to `reports/evaluation_report.json`.

---

## Review, Governance & Caching

- **Autonomous Review Stage:** `review/autogen_stage.py` implements an AutoGen `RoundRobinGroupChat` with two agents and `max_turns=2`. The Final Editor acts as an `AssistantAgent` configured with `output_content_type=StructuredVerdict`. A local AutoGen replay model client provides deterministic mock completions for offline review.
- **Governance Controls:** `governance/controls.py` defines medium-risk profiles, enforces least-autonomy tool access constraints, and monitors per-request token usage estimates.
- **Response Caching:** `crew/crew_runner.py` caches grounded generation responses indexed by normalized queries, returning cached hits immediately to bypass redundant CrewAI kickoff cycles.

---

## Setup & Execution

Use Python 3.11 within a virtual environment.

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare the local embedding model once while online
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Generate policy files and validate the synthetic dataset
python setup_kb.py
python dataset.py

# Run acceptance demonstrations and generate transcripts/reports
python run_transcripts.py

# Start the API server
uvicorn api.main:app --reload