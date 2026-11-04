"""Pydantic models for review findings."""

from typing import Literal

from pydantic import BaseModel


class Finding(BaseModel):
    id: str
    category: Literal["lint", "security", "tests"]
    severity: Literal["high", "medium", "low"]
    title: str
    explanation: str
    file: str
    line: int
    suggestion: str | None = None
    status: Literal["open", "approved", "rejected", "applied", "dismissed"] = "open"


class FindingList(BaseModel):
    findings: list[Finding]
