"""Input and output guardrails required by Part 2 Task 10."""

import re

# Match common international and domestic fixed-format phone numbers. The
# capstone explicitly limits the masking demonstration to fixed-format phone PII.
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,12}\d(?!\w)")
INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"reveal\s+(?:the\s+)?system prompt",
    r"bypass\s+(?:all\s+)?guardrails",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
)


def mask_pii(text: str) -> str:
    """Mask fixed-format phone numbers before model processing or logging."""
    return PHONE_PATTERN.sub("[PHONE REDACTED]", text)


def detect_prompt_injection(text: str) -> bool:
    """Return True when a known instruction-override pattern is present."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)


def groundedness_check(answer: str, retrieved_context: list[str]) -> bool:
    """Require the deterministic mock answer to be present in retrieved context."""
    normalized_answer = " ".join(answer.split()).casefold()
    normalized_context = " ".join(" ".join(retrieved_context).split()).casefold()
    return bool(normalized_answer) and normalized_answer in normalized_context
