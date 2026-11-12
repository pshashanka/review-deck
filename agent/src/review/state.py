"""Graph state. Everything is JSON-serializable because it streams to the frontend."""

from typing import Annotated, Any

from copilotkit import CopilotKitState


# Reset sentinels: put RESET first in a findings update (or RESET_KEY in a
# node_status update) to discard the previous value before merging the rest.
RESET = {"id": "__reset__"}
RESET_KEY = "__reset__"


def merge_findings(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Merge finding dicts by ``id``: later wins, first-seen order is preserved.

    If ``right`` starts with ``RESET``, ``left`` is discarded first.
    """
    right = list(right or [])
    if right and right[0].get("id") == RESET["id"]:
        left, right = [], right[1:]
    merged: dict[str, dict] = {}
    for f in [*(left or []), *right]:
        merged[f["id"]] = f
    return list(merged.values())


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Shallow merge; a ``RESET_KEY`` in ``right`` discards ``left`` first."""
    right = dict(right or {})
    if RESET_KEY in right:
        right.pop(RESET_KEY)
        left = {}
    return {**(left or {}), **right}


def merge_errors(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append notes; a leading ``RESET_KEY`` entry in ``right`` discards ``left`` first."""
    right = list(right or [])
    if right and right[0] == RESET_KEY:
        left, right = [], right[1:]
    return [*(left or []), *right]


class ReviewState(CopilotKitState):
    pr: dict[str, Any] | None
    files: list[dict]
    findings: Annotated[list[dict], merge_findings]
    summary: str | None
    node_status: Annotated[dict[str, str], merge_dict]
    # node name -> model spec actually used by that LLM node; reset with node_status.
    node_models: Annotated[dict[str, str], merge_dict]
    # node name -> {input_tokens, output_tokens, cached_tokens, cost_usd|None, files?}; reset with node_status.
    node_usage: Annotated[dict[str, dict], merge_dict]
    # Totals for the review, set by summarize and cleared by fetch_pr:
    # {input_tokens, output_tokens, cached_tokens, cost_usd|None, skipped_files, files_total}.
    review_stats: dict[str, Any] | None
    error: str | None
    # Existing PR comments set by fetch_pr (None when disabled/failed):
    # {comments: [dataclasses.asdict(github.Comment)], truncated, resolved_known}.
    discussion: dict[str, Any] | None
    review_errors: Annotated[list[str], merge_errors]
    # Set by handle_action for render_findings (compact per-action surface); cleared there.
    action: dict[str, Any] | None
