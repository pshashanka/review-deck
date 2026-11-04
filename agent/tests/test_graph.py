import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import github
from src.agent import graph
from src.review import nodes
from src.review.state import merge_findings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


URL = "https://github.com/o/r/pull/7"


def _pr(number=7, sha="abc", files=None):
    return github.PullRequest(
        owner="o", repo="r", number=number, title=f"PR {number}", author="a", url=URL,
        head_sha=sha, base_ref="main", head_ref="x",
        files=files if files is not None else [PRF("empty.md", set()), PRF("a.py", {3, 4})],
    )


def PRF(name, lines):
    return github.PRFile(filename=name, status="modified", patch="", annotated="", valid_lines=lines)


def cfg(tid):
    return {"configurable": {"thread_id": tid}}


class FakeModel:
    calls: list = []

    async def ainvoke(self, messages, config=None):
        FakeModel.calls.append(messages)
        return AIMessage(content="Please send a PR URL.")


async def test_pr_flow(monkeypatch):
    monkeypatch.setattr(github, "fetch_pr", lambda o, r, n: _pr())
    out = await graph.ainvoke({"messages": [HumanMessage(content=f"review {URL}")]}, cfg("t1"))
    for n in ("fetch_pr", "lint_review", "security_review", "tests_review", "summarize", "render_findings"):
        assert out["node_status"][n] == "done"
    assert sorted(f["id"] for f in out["findings"]) == ["abc-lint-1", "abc-security-1", "abc-tests-1"]
    assert all(f["file"] == "a.py" and f["line"] == 3 for f in out["findings"])
    last = out["messages"][-1]
    assert isinstance(last, ToolMessage)
    ops = json.loads(last.content)["a2ui_operations"]
    kinds = {k for op in ops for k in op}
    assert {"createSurface", "updateComponents", "updateDataModel"} <= kinds
    comps = next(op["updateComponents"]["components"] for op in ops if "updateComponents" in op)
    assert {c["component"] for c in comps} >= {"FindingCard", "ReviewSummary"}
    data = next(op["updateDataModel"]["value"] for op in ops if "updateDataModel" in op)
    assert sorted(f["id"] for f in data["findings"]) == ["abc-lint-1", "abc-security-1", "abc-tests-1"]
    assert sum(data["counts"].values()) == 3
    assert data["pr"]["ref"] == "o/r#7"
    assert all(isinstance(f["suggestion"], str) for f in data["findings"])
    assert all(op[k].get("surfaceId") == "pr-review-7" for op in ops for k in op if isinstance(op[k], dict))


async def test_chat(monkeypatch):
    monkeypatch.setattr(nodes, "create_chat_model", lambda *a, **k: FakeModel())
    out = await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, cfg("t2"))
    assert out["messages"][-1].content == "Please send a PR URL."
    assert "fetch_pr" not in out.get("node_status", {})


async def test_github_error(monkeypatch):
    def boom(o, r, n):
        raise github.GitHubError("PR not found", 404)

    monkeypatch.setattr(github, "fetch_pr", boom)
    out = await graph.ainvoke({"messages": [HumanMessage(content=URL)]}, cfg("t3"))
    assert "PR not found" in out["messages"][-1].content
    assert out["error"] == "PR not found"
    assert "lint_review" not in out["node_status"]


async def test_action_routes_to_handle_action():
    args = {"name": "apply_fix", "surfaceId": "pr-review-7", "sourceComponentId": "b", "context": {}}
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "log_a2ui_event", "args": args, "id": "c1"}]),
        ToolMessage(content='User performed action "apply_fix"', tool_call_id="c1"),
    ]
    out = await graph.ainvoke({"messages": msgs}, cfg("t4"))
    assert out["node_status"]["handle_action"] == "done"
    assert out["node_status"]["render_findings"] == "done"
    assert any(getattr(m, "content", "") == "Action received: apply_fix" for m in out["messages"])
    assert isinstance(out["messages"][-1], ToolMessage)


def test_merge_findings():
    a = [{"id": "x", "t": 1}, {"id": "y", "t": 1}]
    b = [{"id": "x", "t": 2}, {"id": "z", "t": 2}]
    assert merge_findings(a, b) == [{"id": "x", "t": 2}, {"id": "y", "t": 1}, {"id": "z", "t": 2}]
    assert merge_findings(None, b) == b


def _url(n):
    return f"https://github.com/o/r/pull/{n}"


async def test_two_prs_same_thread(monkeypatch):
    monkeypatch.setattr(github, "fetch_pr", lambda o, r, n: _pr(n, sha=f"sha{n}aaaa"))
    await graph.ainvoke({"messages": [HumanMessage(content=_url(1))]}, cfg("t5"))
    out = await graph.ainvoke({"messages": [HumanMessage(content=_url(2))]}, cfg("t5"))
    assert out["pr"]["number"] == 2
    assert {f["id"] for f in out["findings"]} == {"sha2aaa-lint-1", "sha2aaa-security-1", "sha2aaa-tests-1"}
    ops = json.loads(out["messages"][-1].content)["a2ui_operations"]
    assert "sha1aaa" not in json.dumps(ops)
    assert "pr-review-2" in json.dumps(ops)


async def test_followup_chat_after_review(monkeypatch):
    monkeypatch.setattr(github, "fetch_pr", lambda o, r, n: _pr())
    monkeypatch.setattr(nodes, "create_chat_model", lambda *a, **k: FakeModel())
    await graph.ainvoke({"messages": [HumanMessage(content=URL)]}, cfg("t6"))
    FakeModel.calls = []
    out = await graph.ainvoke({"messages": [HumanMessage(content="what is the worst issue?")]}, cfg("t6"))
    assert out["node_status"] == {"chat": "done"}
    assert out["messages"][-1].content == "Please send a PR URL."
    (sent,) = FakeModel.calls
    assert not any(isinstance(m, ToolMessage) for m in sent)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in sent)


async def test_fetch_error_clears_previous(monkeypatch):
    monkeypatch.setattr(github, "fetch_pr", lambda o, r, n: _pr())
    await graph.ainvoke({"messages": [HumanMessage(content=URL)]}, cfg("t7"))

    def boom(o, r, n):
        raise github.GitHubError("nope", 404)

    monkeypatch.setattr(github, "fetch_pr", boom)
    out = await graph.ainvoke({"messages": [HumanMessage(content=_url(9))]}, cfg("t7"))
    assert out["pr"] is None and out["findings"] == [] and out["files"] == [] and out["summary"] is None
    assert set(out["node_status"]) == {"fetch_pr", "fetch_error"}


async def test_zero_valid_lines(monkeypatch):
    monkeypatch.setattr(github, "fetch_pr", lambda o, r, n: _pr(files=[PRF("x.md", set())]))
    out = await graph.ainvoke({"messages": [HumanMessage(content=URL)]}, cfg("t8"))
    assert out["findings"] == []
    assert "Nothing reviewable" in out["summary"]
    assert isinstance(out["messages"][-1], ToolMessage)


def test_reset_reducers():
    from src.review.state import RESET, RESET_KEY, merge_dict

    assert merge_findings([{"id": "a"}], [RESET, {"id": "b"}]) == [{"id": "b"}]
    assert merge_findings([{"id": "a"}], [RESET]) == []
    assert merge_dict({"x": "done"}, {RESET_KEY: True, "y": "done"}) == {"y": "done"}


def test_review_ops_exclude_rejected():
    base = {"file": "a.py", "line": 1, "explanation": "e", "suggestion": None, "category": "lint"}
    state = {
        "pr": {"owner": "o", "repo": "r", "number": 3, "title": "T"},
        "findings": [
            {**base, "id": "a", "severity": "high", "title": "A", "status": "open"},
            {**base, "id": "b", "severity": "low", "title": "B", "status": "rejected"},
        ],
    }
    ops = nodes.build_review_ops(state)
    data = next(op["updateDataModel"]["value"] for op in ops if "updateDataModel" in op)
    assert [f["id"] for f in data["findings"]] == ["a"]
    assert data["counts"] == {"high": 1, "medium": 0, "low": 0}


def test_review_schema_references_exist():
    ids = {c["id"] for c in nodes.REVIEW_SCHEMA}
    assert "root" in ids
    for c in nodes.REVIEW_SCHEMA:
        ch = c.get("children")
        refs = [ch["componentId"]] if isinstance(ch, dict) else (ch or [])
        assert set(refs) <= ids, (c["id"], refs)


def _collect_paths(node, out):
    if isinstance(node, dict):
        if set(node) == {"path"} and isinstance(node["path"], str):
            out.append(node["path"])
        for v in node.values():
            _collect_paths(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_paths(v, out)


def test_review_schema_paths_match_data_model():
    base = {"file": "a.py", "line": 1, "explanation": "e", "suggestion": "s", "category": "lint"}
    state = {
        "pr": {"owner": "o", "repo": "r", "number": 3, "title": "T", "url": "u", "author": "a"},
        "summary": "sum",
        "findings": [{**base, "id": "a", "severity": "high", "title": "A", "status": "open"}],
    }
    ops = nodes.build_review_ops(state)
    data = next(op["updateDataModel"]["value"] for op in ops if "updateDataModel" in op)
    by_id = {c["id"]: c for c in nodes.REVIEW_SCHEMA}

    template_ids = set()
    for c in nodes.REVIEW_SCHEMA:
        ch = c.get("children")
        if isinstance(ch, dict):
            template_ids.add(ch["componentId"])
            cur = data
            for part in ch["path"].strip("/").split("/"):
                cur = cur[part]
            assert isinstance(cur, list) and cur, ch

    for c in nodes.REVIEW_SCHEMA:
        paths: list[str] = []
        _collect_paths({k: v for k, v in c.items() if k != "children"}, paths)
        for path in paths:
            if c["id"] in template_ids:
                assert not path.startswith("/"), (c["id"], path)
                for item in data["findings"]:
                    assert path in item, (c["id"], path)
            else:
                assert path.startswith("/"), (c["id"], path)
                cur = data
                for part in path.strip("/").split("/"):
                    assert part in cur, (c["id"], path)
                    cur = cur[part]
    assert "finding-card" in by_id and template_ids == {"finding-card"}
