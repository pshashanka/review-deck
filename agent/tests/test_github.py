"""Tests for src.github (no network: httpx.MockTransport)."""

import json

import httpx
import pytest

from src.github import (
    GitHubError,
    annotate_patch,
    build_comment_body,
    fetch_pr,
    format_suggestion,
    parse_pr_url,
    post_suggestion,
)

META = {
    "title": "Fix thing",
    "user": {"login": "octo"},
    "html_url": "https://github.com/o/r/pull/7",
    "head": {"sha": "abc123", "ref": "feat"},
    "base": {"ref": "main"},
}


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_API_URL", raising=False)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/o/r/pull/7",
        "https://www.github.com/o/r/pull/7",
        "https://github.com/o/r/pull/7/files",
        "https://github.com/o/r/pull/7/commits?x=1#diff",
        "https://github.com/o/r/pull/7#issuecomment-1",
        "  https://github.com/o/r/pull/7  ",
    ],
)
def test_parse_pr_url_valid(url):
    assert parse_pr_url(url) == ("o", "r", 7)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://github.com/o/r",
        "https://github.com/o/r/issues/7",
        "https://github.com/o/r/pull/abc",
        "https://gitlab.com/o/r/pull/7",
        "https://github.com/o/pull/7",
    ],
)
def test_parse_pr_url_invalid(url):
    with pytest.raises(ValueError, match="pull request URL"):
        parse_pr_url(url)


PATCH = (
    "@@ -1,3 +1,4 @@\n"
    " a\n"
    "-b\n"
    "+B\n"
    "+B2\n"
    " c\n"
    "@@ -10 +11,2 @@ def f():\n"
    " x\n"
    "+y\n"
    "\\ No newline at end of file"
)


def test_annotate_patch_exact():
    text, valid = annotate_patch(PATCH)
    assert text == (
        "@@ -1,3 +1,4 @@\n"
        "    1 a\n"
        "     -b\n"
        "    2+B\n"
        "    3+B2\n"
        "    4 c\n"
        "@@ -10 +11,2 @@ def f():\n"
        "   11 x\n"
        "   12+y\n"
        "\\ No newline at end of file"
    )
    assert valid == {1, 2, 3, 4, 11, 12}


def test_annotate_patch_blank_context_line():
    text, valid = annotate_patch("@@ -1,2 +1,2 @@\n\n+z")
    assert text.split("\n")[1:] == ["    1 ", "    2+z"]
    assert valid == {1, 2}


def test_annotate_patch_deletion_only():
    text, valid = annotate_patch("@@ -5,2 +4,0 @@\n-x\n-y")
    assert text == "@@ -5,2 +4,0 @@\n     -x\n     -y"
    assert valid == set()


def file_item(name, patch="@@ -1 +1 @@\n-a\n+b"):
    item = {"filename": name, "status": "modified"}
    if patch is not None:
        item["patch"] = patch
    return item


def test_fetch_pr_pagination_and_binary():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        if req.url.path == "/repos/o/r/pulls/7":
            return httpx.Response(200, json=META)
        if req.url.params.get("page") == "2":
            return httpx.Response(200, json=[file_item("img.png", None)])
        return httpx.Response(
            200,
            json=[file_item("a.py")],
            headers={
                "Link": '<https://api.github.com/repos/o/r/pulls/7/files?per_page=100&page=2>; rel="next"'
            },
        )

    pr = fetch_pr("o", "r", 7, client=make_client(handler))
    assert (pr.title, pr.author, pr.head_sha, pr.base_ref, pr.head_ref) == (
        "Fix thing", "octo", "abc123", "main", "feat",
    )
    assert [f.filename for f in pr.files] == ["a.py", "img.png"]
    assert pr.files[0].valid_lines == {1}
    assert pr.files[0].annotated == "@@ -1 +1 @@\n     -a\n    1+b"
    assert pr.files[1].patch is None and pr.files[1].valid_lines == set()
    assert not pr.truncated
    assert any("per_page=100" in u for u in seen)
    assert len(seen) == 3


def test_fetch_pr_oversize_file_and_truncation():
    big = "@@ -1 +1 @@\n+" + "x" * 100

    def handler(req):
        if req.url.path.endswith("/files"):
            return httpx.Response(
                200, json=[file_item("huge.py", big + "x" * 100), file_item("a.py", big), file_item("b.py", big)]
            )
        return httpx.Response(200, json=META)

    pr = fetch_pr(
        "o", "r", 7, client=make_client(handler),
        max_total_chars=150, max_file_patch_chars=150,
    )
    assert [f.patch is None for f in pr.files] == [True, False, True]
    assert pr.files[0].valid_lines == set()
    assert pr.truncated is True


@pytest.mark.parametrize("token", [None, "tok"])
def test_fetch_pr_auth_header(monkeypatch, token):
    if token:
        monkeypatch.setenv("GITHUB_TOKEN", token)
    headers = []

    def handler(req):
        headers.append(req.headers)
        return httpx.Response(200, json=[] if req.url.path.endswith("/files") else META)

    fetch_pr("o", "r", 7, client=make_client(handler))
    for h in headers:
        assert h["accept"] == "application/vnd.github+json"
        assert h["x-github-api-version"] == "2022-11-28"
        assert h.get("authorization") == (f"Bearer {token}" if token else None)


def test_fetch_pr_api_url_override(monkeypatch):
    monkeypatch.setenv("GITHUB_API_URL", "https://ghe.example/api/v3/")
    urls = []

    def handler(req):
        urls.append(str(req.url))
        return httpx.Response(200, json=[] if req.url.path.endswith("/files") else META)

    fetch_pr("o", "r", 7, client=make_client(handler))
    assert urls[0] == "https://ghe.example/api/v3/repos/o/r/pulls/7"


def test_fetch_pr_404():
    def handler(req):
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GitHubError) as exc:
        fetch_pr("o", "r", 7, client=make_client(handler))
    assert exc.value.status == 404
    assert "Not Found" in str(exc.value) and "private" in str(exc.value)


def test_post_suggestion(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    captured = {}

    def handler(req):
        captured["url"] = str(req.url)
        captured["auth"] = req.headers["authorization"]
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/7#c1"})

    url = post_suggestion(
        "o", "r", 7, commit_id="abc", path="a.py", line=3, body="hi",
        client=make_client(handler),
    )
    assert url == "https://github.com/o/r/pull/7#c1"
    assert captured["url"] == "https://api.github.com/repos/o/r/pulls/7/comments"
    assert captured["auth"] == "Bearer tok"
    assert captured["body"] == {
        "body": "hi", "commit_id": "abc", "path": "a.py", "line": 3, "side": "RIGHT",
    }


def test_post_suggestion_requires_token():
    def handler(req):
        raise AssertionError("no request expected")

    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        post_suggestion(
            "o", "r", 7, commit_id="abc", path="a.py", line=3, body="hi",
            client=make_client(handler),
        )


def test_post_suggestion_error_includes_message(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    def handler(req):
        return httpx.Response(422, json={"message": "Validation Failed"})

    with pytest.raises(GitHubError, match="Validation Failed") as exc:
        post_suggestion(
            "o", "r", 7, commit_id="abc", path="a.py", line=3, body="hi",
            client=make_client(handler),
        )
    assert exc.value.status == 422


def test_format_suggestion():
    assert format_suggestion("Use B.\n", "x = 1\n") == (
        "Use B.\n\n```suggestion\nx = 1\n```"
    )


def test_annotate_patch_trailing_newline():
    text, valid = annotate_patch("@@ -1,1 +1,1 @@\n-a\n+b\n")
    assert valid == {1}
    assert text == "@@ -1,1 +1,1 @@\n     -a\n    1+b"


def test_non_json_error_body():
    def handler(req):
        return httpx.Response(502, text="<html>Bad gateway</html>")

    with pytest.raises(GitHubError, match="502") as exc:
        fetch_pr("o", "r", 7, client=make_client(handler))
    assert exc.value.status == 502


def test_connect_error_wrapped():
    def handler(req):
        raise httpx.ConnectError("boom")

    with pytest.raises(GitHubError, match="Could not reach GitHub"):
        fetch_pr("o", "r", 7, client=make_client(handler))


def _paged_handler(next_url, seen):
    def handler(req):
        seen.append(str(req.url))
        if req.url.path.endswith("/pulls/7"):
            return httpx.Response(200, json=META)
        if next_url and req.url.params.get("page") != "2":
            return httpx.Response(
                200, json=[file_item("a.py")],
                headers={"Link": f'<{next_url}>; rel="next"'},
            )
        return httpx.Response(200, json=[file_item("b.py")])

    return handler


def test_foreign_host_link_rejected(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    seen = []
    h = _paged_handler("https://evil.example/files?page=2", seen)
    with pytest.raises(GitHubError, match="different host"):
        fetch_pr("o", "r", 7, client=make_client(h))
    assert not any("evil.example" in u for u in seen)


def test_overridden_api_url_pagination(monkeypatch):
    monkeypatch.setenv("GITHUB_API_URL", "https://ghe.example/api/v3")
    seen = []
    h = _paged_handler("https://ghe.example/api/v3/repos/o/r/pulls/7/files?page=2", seen)
    pr = fetch_pr("o", "r", 7, client=make_client(h))
    assert [f.filename for f in pr.files] == ["a.py", "b.py"]


def test_repeated_next_url_stops():
    url = "https://api.github.com/repos/o/r/pulls/7/files?per_page=100"
    calls = []

    def handler(req):
        calls.append(1)
        if req.url.path.endswith("/pulls/7"):
            return httpx.Response(200, json=META)
        return httpx.Response(
            200, json=[file_item("a.py")], headers={"Link": f'<{url}>; rel="next"'}
        )

    pr = fetch_pr("o", "r", 7, client=make_client(handler))
    assert len(pr.files) == 1 and len(calls) == 2


def test_403_rate_limit_vs_forbidden(monkeypatch):
    def limited(req):
        return httpx.Response(
            403, json={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"},
        )

    with pytest.raises(GitHubError, match="rate limit exceeded.*1700000000"):
        fetch_pr("o", "r", 7, client=make_client(limited))

    def forbidden(req):
        return httpx.Response(403, json={"message": "Forbidden"})

    with pytest.raises(GitHubError, match="set GITHUB_TOKEN"):
        fetch_pr("o", "r", 7, client=make_client(forbidden))
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    with pytest.raises(GitHubError, match="token lacks access") as exc:
        fetch_pr("o", "r", 7, client=make_client(forbidden))
    assert "rate limit" not in str(exc.value)


@pytest.mark.parametrize("meta", [{"title": "x"}, {**META, "head": {}}, "notdict"])
def test_malformed_meta(meta):
    def handler(req):
        return httpx.Response(200, json=meta)

    with pytest.raises(GitHubError, match="Unexpected GitHub response"):
        fetch_pr("o", "r", 7, client=make_client(handler))


def test_non_json_success_and_missing_filename():
    def bad_body(req):
        return httpx.Response(200, text="not json")

    with pytest.raises(GitHubError, match="Unexpected GitHub response"):
        fetch_pr("o", "r", 7, client=make_client(bad_body))

    def no_name(req):
        if req.url.path.endswith("/files"):
            return httpx.Response(200, json=[{"status": "added"}])
        return httpx.Response(200, json=META)

    with pytest.raises(GitHubError, match="Unexpected GitHub response"):
        fetch_pr("o", "r", 7, client=make_client(no_name))


@pytest.mark.parametrize("empty", ["", "  ", None])
def test_build_comment_body_without_suggestion(empty):
    body = build_comment_body("Bad @thing", "Because <x>", empty)
    assert body.startswith("**Bad @\u200bthing**")
    assert "&lt;x" in body
    assert "```" not in body


def test_build_comment_body_with_suggestion():
    assert "```suggestion" in build_comment_body("T", "E", "x = 1")
