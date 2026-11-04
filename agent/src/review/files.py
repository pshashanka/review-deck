"""Classify PR files: skip generated/lock noise and give each reviewer only the files it needs."""

import os
from fnmatch import fnmatch
from pathlib import PurePosixPath

LOCKFILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "uv.lock", "poetry.lock",
    "Pipfile.lock", "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum",
})
_NOISE_SUFFIXES = (".min.js", ".min.css", ".map", ".snap", "_pb2.py", ".pb.go")
# Directories skipped at any depth: dependencies and snapshot output are never hand-written.
_NOISE_DIRS_ANYWHERE = frozenset({"node_modules", "__snapshots__"})
# Build/vendor output roots, skipped only as the FIRST path segment (so src/build/ or app/vendor/ stay).
_NOISE_ROOTS = frozenset({"dist", "build", "out", "coverage", "target", ".next", "vendor"})
# Test baselines (e.g. tests/baselines/reference/*.js, tsc/testdata/baselines/...): skipped only for
# non-source data/output files that sit under a "baselines" directory whose parent is a test directory.
_BASELINE_PARENTS = frozenset({"tests", "test", "testdata", "testing", "__tests__"})
_BASELINE_EXTS = (".js", ".json", ".txt", ".snap", ".symbols", ".types", ".log", ".d.ts", ".map")

_TEST_DIRS = frozenset({"tests", "test", "__tests__"})
_DOC_DIRS = frozenset({"docs", "doc"})
_DOC_EXTS = frozenset({".md", ".rst", ".txt", ".adoc"})
_REQ_PREFIXES = ("requirements", "constraints")
_CONFIG_EXTS = frozenset({".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".conf", ".properties", ".env"})
_CONFIG_NAMES = ("dockerfile", "makefile", "docker-compose", ".env", ".gitignore", ".dockerignore", ".editorconfig")

SOURCE, TEST, DOCS, CONFIG = "source", "test", "docs", "config"
# Which kinds of files each reviewer receives (noise is never sent to anyone).
SLICES: dict[str, frozenset[str]] = {
    "security": frozenset({SOURCE, TEST, CONFIG, DOCS}),
    # config included: pytest.ini/tox.ini/package.json/CI workflows change how tests run; it is small.
    "tests": frozenset({SOURCE, TEST, CONFIG}),
    "correctness": frozenset({SOURCE, TEST, CONFIG}),
}


def ignore_globs() -> list[str]:
    """Extra noise globs from ``REVIEW_IGNORE`` (comma-separated fnmatch patterns)."""
    return [g.strip() for g in (os.getenv("REVIEW_IGNORE") or "").split(",") if g.strip()]


def is_lockfile(path: str) -> bool:
    return PurePosixPath(path).name in LOCKFILES


def _is_baseline(parts: tuple[str, ...], name: str) -> bool:
    dirs = parts[:-1]
    under_test_baselines = any(
        d == "baselines" and i > 0 and dirs[i - 1].lower() in _BASELINE_PARENTS for i, d in enumerate(dirs)
    )
    return under_test_baselines and name.endswith(_BASELINE_EXTS)


def is_noise(path: str, extra_globs: list[str] | None = None) -> bool:
    """True for lockfiles, minified/bundled output, snapshots, dependency/build roots and test baselines.

    Directory names alone never hide source, config or script files, except under ``node_modules``
    and at the first path segment of build/vendor roots (``dist/``, ``vendor/`` ...).
    """
    p = PurePosixPath(path)
    name = p.name
    if name in LOCKFILES or name.endswith(_NOISE_SUFFIXES) or fnmatch(name, "*.generated.*"):
        return True
    dirs = p.parts[:-1]
    if any(d in _NOISE_DIRS_ANYWHERE for d in dirs) or (dirs and dirs[0] in _NOISE_ROOTS):
        return True
    if _is_baseline(p.parts, name):
        return True
    globs = ignore_globs() if extra_globs is None else extra_globs
    return any(fnmatch(path, g) or fnmatch(name, g) for g in globs)


def file_kind(path: str) -> str:
    """source | test | docs | config, by path and extension."""
    p = PurePosixPath(path)
    name, lower = p.name, p.name.lower()
    dirs = {d.lower() for d in p.parts[:-1]}
    ext = p.suffix.lower()
    stem = p.stem
    if (
        dirs & _TEST_DIRS
        or stem.startswith("test_") or stem.endswith("_test")
        or any(f".{m}." in lower for m in ("test", "spec"))
    ):
        return TEST
    if lower.startswith(_REQ_PREFIXES) and ext in (".txt", ".in"):
        return CONFIG
    if ext in _DOC_EXTS or dirs & _DOC_DIRS:
        return DOCS
    if ext in _CONFIG_EXTS or lower.startswith(_CONFIG_NAMES) or ".github" in p.parts:
        return CONFIG
    return SOURCE


def in_slice(path: str, category: str) -> bool:
    return file_kind(path) in SLICES[category]
