"""Minimal GitHub pull-request client for the PR-review agent.

Fetches a PR with line-numbered patches (so an LLM can cite new-file lines),
and posts review comments carrying ```suggestion blocks. ``GITHUB_TOKEN`` is
optional for reads of public repos and required for posting;
``GITHUB_API_URL`` overrides the API base (GitHub Enterprise).
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

DEFAULT_API_URL = "https://api.github.com"
TIMEOUT_SECONDS = 20.0

_PR_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)(?:/.*)?$"
)
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_NUM_WIDTH = 5
MAX_FILE_PAGES = 40  # GitHub caps a PR at 3000 files (30 pages of 100)


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


def _fetch_files(client: httpx.Client, url: str) -> list[dict]:
    files: list[dict] = []
    params: dict[str, int] | None = {"per_page": 100}
    next_url: str | None = url
    seen: set[str] = {f"{url}?per_page=100"}  # as the first request is effectively keyed
    base = _base_url()
    pages = 0
    while next_url and pages < MAX_FILE_PAGES:
        pages += 1
        resp = _get(client, next_url, params=params)
        page = _json(resp)
        if not isinstance(page, list):
            raise GitHubError("Unexpected GitHub response: files page is not a list")
        files.extend(page)
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
    return files


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
