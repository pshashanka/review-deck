"""Graph nodes, router and the ``render_review`` tool."""

import copy
import re
import uuid
from pathlib import Path
from typing import Any

from copilotkit import a2ui
from copilotkit.langgraph import copilotkit_emit_state
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src import github
from src.model import create_chat_model
from src.review.models import Finding
from src.review.state import RESET, RESET_KEY, ReviewState

CATALOG_ID = "copilotkit://app-dashboard-catalog"
ACTION_TOOL = "log_a2ui_event"
_PR_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/[^/\s]+/[^/\s]+/pull/\d+[^\s]*")

REVIEW_SCHEMA = a2ui.load_schema(Path(__file__).parent.parent / "a2ui" / "schemas" / "review_schema.json")


# ---------------------------------------------------------------- helpers


def _text(msg: AnyMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def find_pr_url(text: str) -> tuple[str, str, int] | None:
    for m in _PR_URL_RE.finditer(text):
        try:
            return github.parse_pr_url(m.group(0).rstrip(").,>"))
        except ValueError:
            continue
    return None


def _action_tool_call(messages: list[AnyMessage]) -> dict | None:
    """The ``log_a2ui_event`` call if the conversation ends with one (+ its result)."""
    for msg in reversed(messages[-3:]):
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                if tc["name"] == ACTION_TOOL:
                    return tc
            return None
        if isinstance(msg, HumanMessage):
            return None
    return None


def route_intake(state: ReviewState) -> str:
    """Pure router: handle_action | fetch_pr | chat."""
    messages = state.get("messages", [])
    if _action_tool_call(messages):
        return "handle_action"
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return "fetch_pr" if find_pr_url(_text(msg)) else "chat"
    return "chat"


async def _running(state: ReviewState, config: RunnableConfig, node: str, fresh: bool = False) -> None:
    """Mark ``node`` running and push a snapshot; no-op without a CopilotKit config."""
    snapshot = {
        "pr": state.get("pr"),
        "files": state.get("files", []),
        "findings": state.get("findings", []),
        "summary": state.get("summary"),
        "error": state.get("error"),
        "node_status": {**({} if fresh else state.get("node_status", {})), node: "running"},
    }
    try:
        await copilotkit_emit_state(config, snapshot)
    except Exception:
        pass


# ---------------------------------------------------------------- nodes


async def chat(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "chat", fresh=True)
    system = (
        "You are a PR review assistant. If the user has not given a GitHub PR URL, "
        "ask them for one. Otherwise answer briefly about the current review."
    )
    if state.get("pr"):
        titles = "; ".join(f"[{f['severity']}] {f['title']}" for f in state.get("findings", []))
        system += (
            f"\nCurrent review of PR #{state['pr']['number']} '{state['pr']['title']}'."
            f"\nSummary: {state.get('summary')}\nFindings: {titles or 'none'}"
        )
    model = create_chat_model("openai:gpt-5.4")
    # Drop tool traffic (A2UI JSON, log_a2ui_event pairs); the system prompt carries the review.
    history = [
        m for m in state["messages"]
        if not isinstance(m, ToolMessage) and not (isinstance(m, AIMessage) and m.tool_calls)
    ]
    reply = await model.ainvoke([SystemMessage(content=system), *history], config)
    return {
        "messages": [AIMessage(content=reply.content)],
        "node_status": {RESET_KEY: True, "chat": "done"},
    }


async def fetch_pr(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "fetch_pr", fresh=True)
    text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = _text(msg)
            break
    try:
        parsed = find_pr_url(text)
        if not parsed:
            raise ValueError("No GitHub PR URL found in the message.")
        pr = github.fetch_pr(*parsed)
    except (github.GitHubError, ValueError) as e:
        msg = e.message if isinstance(e, github.GitHubError) else str(e)
        return {"error": msg, "node_status": {RESET_KEY: True, "fetch_pr": "error"}}
    return {
        "error": None,
        "findings": [RESET],
        "summary": None,
        "pr": {
            "owner": pr.owner, "repo": pr.repo, "number": pr.number, "title": pr.title,
            "author": pr.author, "url": pr.url, "head_sha": pr.head_sha, "truncated": pr.truncated,
        },
        "files": [
            {"filename": f.filename, "status": f.status, "annotated": f.annotated,
             "valid_lines": sorted(f.valid_lines)}
            for f in pr.files
        ],
        "node_status": {RESET_KEY: True, "fetch_pr": "done"},
    }


def after_fetch(state: ReviewState) -> list[str]:
    if state.get("error"):
        return ["fetch_error"]
    return ["lint_review", "security_review", "tests_review"]


async def fetch_error(state: ReviewState, config: RunnableConfig):
    return {
        "messages": [AIMessage(content=f"I couldn't fetch that pull request: {state.get('error')}")],
        "pr": None,
        "files": [],
        "findings": [RESET],
        "summary": None,
        "node_status": {"fetch_error": "done"},
    }


def _stub_finding(state: ReviewState, category: str, severity: str, title: str, why: str) -> list[dict]:
    sha = ((state.get("pr") or {}).get("head_sha") or "run")[:7]
    for f in state.get("files", []):
        if f["valid_lines"]:
            return [Finding(
                id=f"{sha}-{category}-1", category=category, severity=severity, title=title,
                explanation=why, file=f["filename"], line=f["valid_lines"][0],
            ).model_dump()]
    return []


# STUB reviewers: canned findings, replaced by real LLM reviewers in Chunk 4.
async def lint_review(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "lint_review")
    return {
        "findings": _stub_finding(state, "lint", "medium", "Stub lint finding", "Canned lint finding (stub)."),
        "node_status": {"lint_review": "done"},
    }


async def security_review(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "security_review")
    return {
        "findings": _stub_finding(state, "security", "high", "Stub security finding", "Canned security finding (stub)."),
        "node_status": {"security_review": "done"},
    }


async def tests_review(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "tests_review")
    return {
        "findings": _stub_finding(state, "tests", "low", "Stub tests finding", "Canned tests finding (stub)."),
        "node_status": {"tests_review": "done"},
    }


async def summarize(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "summarize")
    findings = state.get("findings", [])
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in ("high", "medium", "low")}
    if not findings:
        reviewable = any(f["valid_lines"] for f in state.get("files", []))
        summary = "No findings." if reviewable else "Nothing reviewable: no changed lines with patches in this PR."
        return {"summary": summary, "node_status": {"summarize": "done"}}
    summary = (
        f"{len(findings)} finding(s): {counts['high']} high, "
        f"{counts['medium']} medium, {counts['low']} low."
    )
    return {"summary": summary, "node_status": {"summarize": "done"}}


async def render_findings(state: ReviewState, config: RunnableConfig):
    await _running(state, config, "render_findings")
    return {
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "render_review", "args": {}, "id": str(uuid.uuid4())}
        ])],
        "node_status": {"render_findings": "done"},
    }


async def handle_action(state: ReviewState, config: RunnableConfig):
    # STUB: real apply/dismiss logic arrives in Chunk 6.
    await _running(state, config, "handle_action", fresh=True)
    tc = _action_tool_call(state["messages"]) or {}
    name = (tc.get("args") or {}).get("name", "unknown")
    return {
        "messages": [AIMessage(content=f"Action received: {name}")],
        "node_status": {RESET_KEY: True, "handle_action": "done"},
    }


# ---------------------------------------------------------------- tool


def build_review_ops(state: dict[str, Any]) -> list[dict]:
    pr = state.get("pr") or {}
    number = pr.get("number", 0)
    sid = f"pr-review-{number}"
    findings = [f for f in state.get("findings", []) if f.get("status") != "rejected"]
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in ("high", "medium", "low")}
    data = {
        "pr": {
            "title": pr.get("title") or "Pull request review",
            "url": pr.get("url", ""),
            "ref": f"{pr.get('owner', '')}/{pr.get('repo', '')}#{number}",
            "author": pr.get("author", ""),
            "truncated": bool(pr.get("truncated")),
        },
        "counts": counts,
        "summary": state.get("summary") or "",
        "findings": [
            {
                "id": f["id"],
                "severity": f["severity"],
                "category": f["category"],
                "title": f["title"],
                "explanation": f["explanation"],
                "file": f["file"],
                "line": f["line"],
                "suggestion": f.get("suggestion") or "",
                "status": f.get("status", "open"),
            }
            for f in findings
        ],
    }
    return [
        a2ui.create_surface(sid, catalog_id=CATALOG_ID),
        a2ui.update_components(sid, copy.deepcopy(REVIEW_SCHEMA)),
        a2ui.update_data_model(sid, data),
    ]


@tool
def render_review(runtime: ToolRuntime) -> str:
    """Render the current PR review findings as an A2UI surface."""
    return a2ui.render(build_review_ops(runtime.state))
