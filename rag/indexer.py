"""Local SentenceTransformers embeddings and two persistent Chroma indexes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import fixed_size_chunking, sentence_chunking

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / "data" / "chroma"
MODEL_NAME = "all-MiniLM-L6-v2"

# Five calibration prompts are deliberately fixed so the measured threshold is
# reproducible for a given embedding-model version and knowledge base.
CALIBRATION_IN_SCOPE = [
    "What is the initial response time for urgent tickets?",
    "How much refund can be approved automatically?",
    "How are VIP corporate partner tickets routed?",
]
CALIBRATION_OUT_OF_SCOPE = [
    "Give me a recipe for chocolate chip cookies.",
    "What is the weather forecast for Tokyo tomorrow?",
]


class RAGCore:
    """Index both chunking strategies and answer using retrieved policy text."""

    def __init__(self, rebuild: bool = False) -> None:
        if not KB_DIR.exists() or not list(KB_DIR.glob("*.md")):
            raise FileNotFoundError("Knowledge base is empty. Run `python setup_kb.py` first.")

        # local_files_only prevents a model download during a graded MOCK_LLM run.
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.fixed_collection = self.client.get_or_create_collection(
            "ola_kb_fixed", metadata={"hnsw:space": "cosine"}
        )
        self.sentence_collection = self.client.get_or_create_collection(
            "ola_kb_sentences", metadata={"hnsw:space": "cosine"}
        )
        if rebuild:
            self.client.delete_collection("ola_kb_fixed")
            self.client.delete_collection("ola_kb_sentences")
            self.fixed_collection = self.client.get_or_create_collection(
                "ola_kb_fixed", metadata={"hnsw:space": "cosine"}
            )
            self.sentence_collection = self.client.get_or_create_collection(
                "ola_kb_sentences", metadata={"hnsw:space": "cosine"}
            )
        self._index_documents()
        self.calibration: dict[str, Any] | None = None
        self.recommended_strategy = "sentence"

    def _index_documents(self) -> None:
        """Embed chunks in batches and upsert them with parent-document metadata."""
        fixed_rows: list[tuple[str, str, dict[str, str]]] = []
        sentence_rows: list[tuple[str, str, dict[str, str]]] = []

        for path in sorted(KB_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for index, chunk in enumerate(fixed_size_chunking(text)):
                fixed_rows.append((f"{path.stem}-fixed-{index}", chunk, {"source": path.name}))
            for index, chunk in enumerate(sentence_chunking(text)):
                sentence_rows.append((f"{path.stem}-sentence-{index}", chunk, {"source": path.name}))

        self._upsert_rows(self.fixed_collection, fixed_rows)
        self._upsert_rows(self.sentence_collection, sentence_rows)

    def _upsert_rows(self, collection: Any, rows: list[tuple[str, str, dict[str, str]]]) -> None:
        """Create embeddings once per batch, then upsert documents and metadata."""
        if not rows:
            raise ValueError(f"No chunks were produced for {collection.name}.")
        ids = [row[0] for row in rows]
        documents = [row[1] for row in rows]
        metadatas = [row[2] for row in rows]
        embeddings = self.model.encode(documents, normalize_embeddings=True).tolist()
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def retrieve(self, query: str, strategy: str = "sentence", k: int = 3) -> dict[str, Any]:
        """Return top-k chunks and cosine similarities for the selected index."""
        collection = self._get_collection(strategy)
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=query_embedding, n_results=min(k, collection.count()))
        distances = result.get("distances", [[]])[0]
        similarities = [max(-1.0, min(1.0, 1.0 - float(distance))) for distance in distances]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return {
            "documents": documents,
            "metadatas": metadatas,
            "similarities": similarities,
            "top1_similarity": similarities[0] if similarities else -1.0,
        }

    def _get_collection(self, strategy: str) -> Any:
        if strategy == "sentence":
            return self.sentence_collection
        if strategy == "fixed":
            return self.fixed_collection
        raise ValueError("strategy must be 'fixed' or 'sentence'.")

    def calibrate_threshold(self, strategy: str = "sentence") -> dict[str, Any]:
        """Measure 3 in-scope and 2 out-of-scope queries, then set midpoint."""
        in_scope = {q: self.retrieve(q, strategy, k=1)["top1_similarity"] for q in CALIBRATION_IN_SCOPE}
        out_scope = {q: self.retrieve(q, strategy, k=1)["top1_similarity"] for q in CALIBRATION_OUT_OF_SCOPE}
        lowest_in_scope = min(in_scope.values())
        highest_out_scope = max(out_scope.values())
        if lowest_in_scope <= highest_out_scope:
            raise ValueError(
                "Calibration clusters overlap: choose clearer calibration queries or inspect the KB. "
                f"Lowest in-scope={lowest_in_scope:.4f}; highest out-of-scope={highest_out_scope:.4f}."
            )
        threshold = (lowest_in_scope + highest_out_scope) / 2.0
        self.calibration = {
            "strategy": strategy,
            "in_scope_top1_cosine": in_scope,
            "out_of_scope_top1_cosine": out_scope,
            "lowest_in_scope": lowest_in_scope,
            "highest_out_of_scope": highest_out_scope,
            "chosen_threshold": threshold,
        }
        report_path = ROOT / "reports" / "rag_calibration.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.calibration, indent=2), encoding="utf-8")
        return self.calibration

    def answer(self, query: str, strategy: str = "sentence", k: int = 3) -> dict[str, Any]:
        """Return a context-grounded answer or an explicit abstention."""
        if self.calibration is None or self.calibration["strategy"] != strategy:
            self.calibrate_threshold(strategy)
        retrieved = self.retrieve(query, strategy, k)
        threshold = float(self.calibration["chosen_threshold"])
        if retrieved["top1_similarity"] < threshold:
            return {
                "answer": "I don't know based on the available Ola support policies.",
                "grounded": False,
                "fallback": True,
                "context": retrieved["documents"],
                "sources": sorted({m["source"] for m in retrieved["metadatas"]}),
                "top1_similarity": retrieved["top1_similarity"],
                "threshold": threshold,
            }

        # In MOCK_LLM mode the safest deterministic generator quotes the most
        # relevant retrieved policy sentence instead of inventing new facts.
        answer = self._select_supported_sentence(query, retrieved["documents"])
        grounded = self.is_grounded(answer, retrieved["documents"])
        if not grounded:
            answer = "I don't know based on the available Ola support policies."
        return {
            "answer": answer,
            "grounded": grounded,
            "fallback": not grounded,
            "context": retrieved["documents"],
            "sources": sorted({m["source"] for m in retrieved["metadatas"]}),
            "top1_similarity": retrieved["top1_similarity"],
            "threshold": threshold,
        }

    @staticmethod
    def _select_supported_sentence(query: str, documents: list[str]) -> str:
        """Choose the retrieved sentence with the greatest query-token overlap."""
        query_terms = set(re.findall(r"[a-z0-9$]+", query.lower()))
        candidates: list[tuple[int, str]] = []
        for document in documents:
            for sentence in re.split(r"(?<=[.!?])\s+", document):
                terms = set(re.findall(r"[a-z0-9$]+", sentence.lower()))
                candidates.append((len(query_terms & terms), sentence.strip()))
        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def is_grounded(answer: str, context: list[str]) -> bool:
        """Accept only an answer copied from the retrieved context."""
        normalized_answer = " ".join(answer.split()).casefold()
        normalized_context = " ".join(" ".join(context).split()).casefold()
        return bool(normalized_answer) and normalized_answer in normalized_context

    def add_document(self, filename: str, content: str) -> None:
        """Persist a new policy file and add its chunks to both indexes."""
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(filename).stem) + ".md"
        path = KB_DIR / safe_name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        self._index_documents()
