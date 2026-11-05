"""Pydantic response schema shared by the crew and API."""

from pydantic import BaseModel, Field


class CrewResponseSchema(BaseModel):
    answer: str = Field(min_length=1, description="Grounded answer or explicit fallback.")
    confidence: float = Field(ge=0.0, le=1.0)
    requires_escalation: bool
