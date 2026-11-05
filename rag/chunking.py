"""The two chunking strategies required for the RAG comparison."""

import re


def fixed_size_chunking(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """Split text into fixed character windows with a controlled overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    clean_text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        # Move the end forward to a word boundary where practical.
        end = min(start + chunk_size, len(clean_text))
        if end < len(clean_text):
            boundary = clean_text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def sentence_chunking(text: str) -> list[str]:
    """Split a policy article into sentence-sized retrieval units."""
    clean_text = " ".join(text.split())
    # The policy documents use conventional sentence punctuation.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean_text) if part.strip()]
