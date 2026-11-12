"""Pydantic models for review findings."""

from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    id: str
    category: Literal["security", "tests", "correctness"]
    severity: Literal["high", "medium", "low"]
    title: str
    explanation: str
    file: str
    line: int
    suggestion: str | None = None
    status: Literal["open", "approved", "rejected", "applied", "dismissed"] = "open"
    comment_url: str = ""
    error: str = ""
    prior_author: str = ""
    prior_url: str = ""
    prior_is_bot: bool = False


class FindingList(BaseModel):
    findings: list[Finding]


class ReviewerFinding(BaseModel):
    """One issue found by an LLM reviewer. Ids, category and status are assigned by code."""

    severity: Literal["high", "medium", "low"] = Field(
        description="high = must fix before merge (exploitable security, data loss, crash in a common path); "
        "medium = should fix; low = nice to have."
    )
    title: str = Field(description="Short headline, at most 80 characters, no trailing period.")
    explanation: str = Field(description="1-3 sentences: what is wrong and why it matters.")
    file: str = Field(description="Exact filename as shown in a '### <filename>' header of the diff.")
    line: int | None = Field(
        description="New-file line number of a numbered line (added '+' or context) shown in that file's diff. "
        "Never a removed line, hunk header or a line that is not shown."
    )
    suggestion: str | None = Field(
        default=None,
        description="Complete replacement text for exactly that one line (same indentation, no line number, "
        "no '+' prefix, a single line without newlines), or null if a one-line fix is not possible.",
    )


class ReviewerOutput(BaseModel):
    findings: list[ReviewerFinding] = Field(
        default_factory=list,
        description="At most 8 findings, most important first. Empty list when nothing is worth flagging.",
    )
