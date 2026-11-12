"""Minimal GitHub pull-request client for the PR-review agent.

Fetches a PR with line-numbered patches (so an LLM can cite new-file lines),
and posts review comments carrying ```suggestion blocks. ``GITHUB_TOKEN`` is
optional for reads of public repos and required for posting;
``GITHUB_API_URL`` overrides the API base (GitHub Enterprise).
"""

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.github.com"
TIMEOUT_SECONDS = 20.0

_PR_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)(?:/.*)?$"
)
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_NUM_WIDTH = 5
MAX_FILE_PAGES = 40  # GitHub caps a PR at 3000 files (30 pages of 100)
MAX_DISCUSSION_PAGES = 10
MAX_THREAD_PAGES = 5


class GitHubError(Exception):
    """A GitHub API failure with a human-readable message and HTTP status."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class PRFile:
    filename: str
    status: str
    patch: str | None
    annotated: str
    valid_lines: set[int] = field(default_factory=set)


@dataclass
class PullRequest:
    owner: str
    repo: str
    number: int
    title: str
    author: str
    url: str
    head_sha: str
    base_ref: str
    head_ref: str
    files: list[PRFile]
    truncated: bool = False
    skipped_files: list[str] = field(default_factory=list)
    # skipped filename -> (additions, deletions) as reported by GitHub
    skipped_changes: dict[str, tuple[int, int]] = field(default_factory=dict)


TITLE_LIMIT = 120
EXPLANATION_LIMIT = 1000


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Return ``(owner, repo, number)`` from a GitHub pull-request URL."""
    cleaned = url.strip().split("#", 1)[0].split("?", 1)[0]
    match = _PR_URL.match(cleaned)
    if not match:
        raise ValueError(
            f"Not a GitHub pull request URL: {url!r} "
            "(expected https://github.com/<owner>/<repo>/pull/<number>)"
        )
    owner, repo, number = match.groups()
    return owner, repo, int(number)


def annotate_patch(patch: str) -> tuple[str, set[int]]:
    """Number a unified-diff patch by new-file line.

    Added lines render as ``<n>+text``, context as ``<n> text``, removed as
    ``<blank>-text``; hunk headers are kept. Returns the text and the set of
    new-file lines a RIGHT-side review comment may target (added + context).
    """
    out: list[str] = []
    valid: set[int] = set()
    new_line: int | None = None  # None until the first hunk header
    blank = " " * _NUM_WIDTH
    lines = patch.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline, not a blank context line
    for raw in lines:
        header = _HUNK_HEADER.match(raw)
        if header:
            new_line = int(header.group(1))
            out.append(raw)
        elif new_line is None:
            out.append(raw)
        elif raw.startswith("+"):
            out.append(f"{new_line:>{_NUM_WIDTH}}+{raw[1:]}")
            valid.add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            out.append(f"{blank}-{raw[1:]}")
        elif raw.startswith("\\"):
            out.append(raw)
        else:  # context; a bare empty line is a context line whose space was stripped
            out.append(f"{new_line:>{_NUM_WIDTH}} {raw[1:]}")
            valid.add(new_line)
            new_line += 1
    return "\n".join(out), valid


def _neutralize(text: str, limit: int) -> str:
    """Make untrusted text safe to post: no @-mentions, no raw HTML, bounded length."""
    text = text.strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return text.replace("<", "&lt;").replace("@", "@\u200b")


def format_suggestion(explanation: str, suggestion: str) -> str:
    """Comment body: the explanation followed by a ```suggestion block.

    The fence is longer than any backtick run inside the suggestion so the code
    cannot close it early.
    """
    code = suggestion.rstrip("\n")
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{explanation.rstrip()}\n\n{fence}suggestion\n{code}\n{fence}"


def build_comment_body(title: str, explanation: str, suggestion: str | None = None) -> str:
    """Full sanitized review-comment body for a finding (title in bold).

    With an empty/None suggestion the body is just the title and explanation (no suggestion block).
    """
    head = f"**{_neutralize(title, TITLE_LIMIT)}**\n\n{_neutralize(explanation, EXPLANATION_LIMIT)}"
    if not suggestion or not suggestion.strip():
        return head
    return format_suggestion(head, suggestion)


def _base_url() -> str:
    return os.environ.get("GITHUB_API_URL", DEFAULT_API_URL).rstrip("/")


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        detail = resp.json().get("message", "")
    except (ValueError, AttributeError):
        detail = resp.text[:200]
    status = resp.status_code
    has_token = bool(os.environ.get("GITHUB_TOKEN"))
    hint = ""
    if status == 401:
        hint = "Bad or expired credentials; check GITHUB_TOKEN."
    elif status in (403, 429) and (
        resp.headers.get("X-RateLimit-Remaining") == "0"
        or "Retry-After" in resp.headers
        or status == 429
    ):
        hint = "GitHub rate limit exceeded; set GITHUB_TOKEN or retry later."
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset:
            hint += f" Limit resets at epoch {reset}."
        elif "Retry-After" in resp.headers:
            hint += f" Retry after {resp.headers['Retry-After']}s."
    elif status == 403:
        hint = (
            "token lacks access to this repo"
            if has_token
            else "set GITHUB_TOKEN with access to the repo"
        )
    elif status == 404:
        hint = "PR not found or private; set GITHUB_TOKEN if the repo is private."
    message = f"GitHub API error {status}" + (f": {detail}" if detail else "")
    raise GitHubError(f"{message} ({hint})" if hint else message, status)


def _json(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError as exc:
        raise GitHubError(f"Unexpected GitHub response: body is not JSON ({exc})") from exc


def _same_origin(url: str, base: str) -> bool:
    a, b = urlsplit(url), urlsplit(base)
    return (a.scheme, a.netloc.lower()) == (b.scheme, b.netloc.lower())


def _get(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    try:
        resp = client.get(url, headers=_headers(), **kwargs)
    except httpx.HTTPError as exc:
        raise GitHubError(f"Could not reach GitHub: {exc}") from exc
    _raise_for_status(resp)
    return resp


def _fetch_list(client: httpx.Client, url: str, label: str, max_pages: int) -> list[dict]:
    items: list[dict] = []
    params: dict[str, int] | None = {"per_page": 100}
    next_url: str | None = url
    seen: set[str] = {f"{url}?per_page=100"}  # as the first request is effectively keyed
    base = _base_url()
    pages = 0
    while next_url and pages < max_pages:
        pages += 1
        resp = _get(client, next_url, params=params)
        page = _json(resp)
        if not isinstance(page, list):
            raise GitHubError(f"Unexpected GitHub response: {label} page is not a list")
        items.extend(page)
        params = None  # the Link URL already carries the query
        next_url = resp.links.get("next", {}).get("url")
        if next_url in seen:
            break
        if next_url:
            seen.add(next_url)
        if next_url and not _same_origin(next_url, base):
            raise GitHubError(
                f"Refusing to follow pagination link to a different host: {next_url}"
            )
    return items


def _fetch_files(client: httpx.Client, url: str) -> list[dict]:
    return _fetch_list(client, url, "files", MAX_FILE_PAGES)


def fetch_pr(
    owner: str,
    repo: str,
    number: int,
    *,
    client: httpx.Client | None = None,
    max_total_chars: int = 60_000,
    max_file_patch_chars: int = 20_000,
    skip: Callable[[str], bool] | None = None,
) -> PullRequest:
    """Fetch a PR and its line-annotated file patches, within size budgets.

    Files for which ``skip(filename)`` is true are left out (and listed in
    ``skipped_files``) without counting against the size budgets.
    """
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        base = f"{_base_url()}/repos/{owner}/{repo}/pulls/{number}"
        meta = _json(_get(client, base))
        raw_files = _fetch_files(client, f"{base}/files")
    finally:
        if own_client:
            client.close()

    try:
        return _build_pr(owner, repo, number, meta, raw_files, max_total_chars, max_file_patch_chars, skip)
    except (KeyError, TypeError, AttributeError) as exc:
        raise GitHubError(f"Unexpected GitHub response: missing or invalid {exc}") from exc


def _build_pr(
    owner: str,
    repo: str,
    number: int,
    meta: dict,
    raw_files: list[dict],
    max_total_chars: int,
    max_file_patch_chars: int,
    skip: Callable[[str], bool] | None = None,
) -> PullRequest:
    files: list[PRFile] = []
    skipped: list[str] = []
    skipped_changes: dict[str, tuple[int, int]] = {}
    total = 0
    truncated = False
    for item in raw_files:
        name, status = item["filename"], item.get("status", "modified")
        if skip and skip(name):
            skipped.append(name)
            skipped_changes[name] = (int(item.get("additions") or 0), int(item.get("deletions") or 0))
            continue
        patch = item.get("patch")
        if not patch:
            note = "[no textual patch available (binary, empty or too large for GitHub)]"
            files.append(PRFile(name, status, None, note))
        elif len(patch) > max_file_patch_chars:
            note = f"[patch omitted: {len(patch)} chars exceeds {max_file_patch_chars}]"
            files.append(PRFile(name, status, None, note))
        elif truncated or total + len(patch) > max_total_chars:
            truncated = True
            files.append(
                PRFile(name, status, None, "[patch omitted: total size limit reached]")
            )
        else:
            annotated, valid = annotate_patch(patch)
            total += len(patch)
            files.append(PRFile(name, status, patch, annotated, valid))

    return PullRequest(
        owner=owner,
        repo=repo,
        number=number,
        title=meta.get("title", ""),
        author=(meta.get("user") or {}).get("login", ""),
        url=meta["html_url"],
        head_sha=meta["head"]["sha"],
        base_ref=meta["base"]["ref"],
        head_ref=meta["head"]["ref"],
        files=files,
        truncated=truncated,
        skipped_files=skipped,
        skipped_changes=skipped_changes,
    )


def post_suggestion(
    owner: str,
    repo: str,
    number: int,
    *,
    commit_id: str,
    path: str,
    line: int,
    body: str,
    client: httpx.Client | None = None,
) -> str:
    """Post a RIGHT-side review comment on ``path:line``; return its URL."""
    if not os.environ.get("GITHUB_TOKEN"):
        raise GitHubError("GITHUB_TOKEN is required to post review comments.", 401)
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        resp = client.post(
            f"{_base_url()}/repos/{owner}/{repo}/pulls/{number}/comments",
            headers=_headers(),
            json={
                "body": body,
                "commit_id": commit_id,
                "path": path,
                "line": line,
                "side": "RIGHT",
            },
        )
    except httpx.HTTPError as exc:
        raise GitHubError(f"Could not reach GitHub: {exc}") from exc
    finally:
        if own_client:
            client.close()
    _raise_for_status(resp)
    try:
        return _json(resp)["html_url"]
    except (KeyError, TypeError) as exc:
        raise GitHubError(f"Unexpected GitHub response: missing {exc}") from exc


@dataclass
class Comment:
    kind: str  # "issue" | "review" | "inline"
    author: str
    is_bot: bool
    body: str
    created_at: str
    url: str
    path: str | None = None
    line: int | None = None  # new-file line; None when outdated
    outdated: bool = False
    comment_id: int | None = None  # REST id


@dataclass
class Discussion:
    comments: list[Comment]
    truncated: bool = False
    resolved_known: bool = False
    errors: list[str] = field(default_factory=list)


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANK_RUNS = re.compile(r"\n{3,}")

_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved
          comments(first: 100) { nodes { databaseId } }
        }
      }
    }
  }
}
"""


def _clean_body(text: object, limit: int) -> str:
    text = _HTML_COMMENT.sub("", text if isinstance(text, str) else "").strip()
    text = _BLANK_RUNS.sub("\n\n", text)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return text


def _graphql_url() -> str:
    base = _base_url()
    if base.endswith("/api/v3"):
        return base[: -len("/v3")] + "/graphql"
    return f"{base}/graphql"


def _fetch_resolved_ids(client: httpx.Client, owner: str, repo: str, number: int) -> set[int]:
    """REST ids of review comments that sit in resolved threads."""
    resolved: set[int] = set()
    cursor: str | None = None
    for _ in range(MAX_THREAD_PAGES):
        try:
            resp = client.post(
                _graphql_url(),
                headers=_headers(),
                json={
                    "query": _THREADS_QUERY,
                    "variables": {"owner": owner, "name": repo, "number": number, "cursor": cursor},
                },
            )
        except httpx.HTTPError as exc:
            raise GitHubError(f"Could not reach GitHub: {exc}") from exc
        _raise_for_status(resp)
        data = _json(resp)
        if not isinstance(data, dict) or data.get("errors"):
            raise GitHubError("GraphQL returned errors")
        try:
            threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]
            for node in threads["nodes"]:
                if node["isResolved"]:
                    for c in node["comments"]["nodes"]:
                        if c.get("databaseId") is not None:
                            resolved.add(c["databaseId"])
            info = threads["pageInfo"]
            has_next, cursor = bool(info["hasNextPage"]), info.get("endCursor")
        except (KeyError, TypeError, AttributeError) as exc:
            raise GitHubError(f"Unexpected GraphQL response: missing or invalid {exc}") from exc
        if not has_next or not cursor:
            break
    return resolved


def _hunk_end_line(diff_hunk: object) -> int | None:
    """New-file line of the last line of ``diff_hunk`` (None if unknown or a removed line)."""
    if not isinstance(diff_hunk, str):
        return None
    lines = diff_hunk.split("\n")
    start = None
    for i, raw in enumerate(lines):
        m = _HUNK_HEADER.match(raw)
        if m:
            start, new_line = i, int(m.group(1))
    if start is None:
        return None
    body = lines[start + 1 :]
    if body and body[-1] == "":
        body.pop()
    last: int | None = None
    for raw in body:
        if raw.startswith("\\"):
            continue
        if raw.startswith("-"):
            last = None
        elif raw.startswith("+") or raw.startswith(" ") or raw == "":
            last = new_line
            new_line += 1
        else:
            return None
    return last


def _is_bot(user: dict) -> bool:
    return user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")


def fetch_discussion(
    owner: str,
    repo: str,
    number: int,
    *,
    client: httpx.Client | None = None,
    head_sha: str | None = None,
    max_total_chars: int = 20_000,
    max_comment_chars: int = 2_000,
) -> Discussion:
    """Fetch PR conversation (issue comments, review bodies, inline comments).

    Every source is best-effort: failures are recorded in ``errors`` and never
    raised. Inline comments in resolved threads are dropped when the GraphQL
    lookup succeeds (needs ``GITHUB_TOKEN``).
    """
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    errors: list[str] = []
    comments: list[Comment] = []
    resolved: set[int] = set()
    resolved_known = False
    base = f"{_base_url()}/repos/{owner}/{repo}"

    def load(path: str, label: str, build) -> None:
        try:
            for item in _fetch_list(client, f"{base}/{path}", label, MAX_DISCUSSION_PAGES):
                built = build(item)
                if built is not None:
                    comments.append(built)
        except GitHubError as exc:
            errors.append(f"{label}: {exc.message}")
        except (KeyError, TypeError, AttributeError) as exc:
            errors.append(f"{label}: unexpected response ({exc!r})")

    def common(item: dict, kind: str, when_key: str) -> Comment | None:
        user = item.get("user") or {}
        body = _clean_body(item.get("body"), max_comment_chars)
        if not body:
            return None
        return Comment(
            kind=kind,
            author=user.get("login", ""),
            is_bot=_is_bot(user),
            body=body,
            created_at=item.get(when_key) or "",
            url=item.get("html_url") or "",
            comment_id=item["id"] if isinstance(item.get("id"), int) else None,
        )

    def build_inline(item: dict) -> Comment | None:
        c = common(item, "inline", "created_at")
        if c is None:
            return None
        c.path = item.get("path")
        api_line = item.get("line")
        if api_line is None:
            c.line, c.outdated = None, True
        else:
            computed = _hunk_end_line(item.get("diff_hunk"))
            if computed is None or computed == item.get("original_line"):
                c.line = api_line
            else:
                # GitHub intermittently returns line=original_line=1; recompute from the hunk.
                logger.debug(
                    "inline comment %s: API line %s disagrees with diff_hunk line %s",
                    item.get("id"), api_line, computed,
                )
                c.line = computed if head_sha and item.get("commit_id") == head_sha else None
        return c

    try:
        load(f"issues/{number}/comments", "issue comments", lambda i: common(i, "issue", "created_at"))
        load(f"pulls/{number}/reviews", "reviews", lambda i: common(i, "review", "submitted_at"))
        load(f"pulls/{number}/comments", "inline comments", build_inline)
        if os.environ.get("GITHUB_TOKEN"):
            try:
                resolved = _fetch_resolved_ids(client, owner, repo, number)
                resolved_known = True
            except GitHubError as exc:
                errors.append(f"resolved threads: {exc.message}")
        if resolved:
            comments = [
                c for c in comments if not (c.kind == "inline" and c.comment_id in resolved)
            ]
    finally:
        if own_client:
            client.close()

    comments.sort(key=lambda c: c.created_at)
    truncated = False
    total = sum(len(c.body) for c in comments)
    drop = 0
    while total > max_total_chars and drop < len(comments):
        total -= len(comments[drop].body)
        drop += 1
        truncated = True
    return Discussion(comments[drop:], truncated, resolved_known, errors)
