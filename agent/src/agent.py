"""PR-review graph.

START -> route_intake -> handle_action | fetch_pr | chat
fetch_pr -> [security_review, tests_review, correctness_review] (parallel) -> approval_gate -> summarize
         -> render_findings -> render_tools (ToolNode: render_review) -> END
fetch_pr (error) -> fetch_error -> END
handle_action -> render_findings (only when a finding changed) | END;  chat -> END
"""

import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.review import nodes
from src.review.state import ReviewState

logger = logging.getLogger(__name__)

# Default checkpoint database, relative to the repository root (agent/..).
DEFAULT_CHECKPOINT_DB = "agent/.checkpoints/checkpoints.db"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _checkpoint_db_path() -> str:
    """Absolute path of the checkpoint database.

    A relative ``LANGGRAPH_CHECKPOINT_DB`` (and the default) resolves against
    the repository root, matching how ``.env`` paths are written in the README.
    ``~`` is expanded here so the same path reaches ``mkdir`` and ``connect``.
    ``:memory:`` is returned unchanged (an in-memory database, not a file).
    """
    db = (os.getenv("LANGGRAPH_CHECKPOINT_DB") or "").strip() or DEFAULT_CHECKPOINT_DB
    if db == ":memory:":
        return db
    path = Path(db).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path)


async def enter_checkpointer(stack: AsyncExitStack):
    """Open a persistent SQLite checkpointer on ``stack``.

    The async saver must be created inside the running event loop (it binds its
    connection to that loop). Entering through ``AsyncSqliteSaver.from_conn_string``
    on the caller's ``AsyncExitStack`` hands connection teardown to the stack, and
    ``setup()`` runs eagerly so an unusable database (corrupt, locked,
    non-database, read-only) fails here and falls back to ``MemorySaver`` rather
    than failing every request later.

    Returns the saver, or ``MemorySaver`` when the sqlite packages are missing or
    the database cannot be opened.
    """
    db = _checkpoint_db_path()
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite (and its aiosqlite dependency) is not installed; "
            "checkpoints will not survive an agent restart (using in-memory state)"
        )
        return MemorySaver()

    try:
        if db != ":memory:":
            Path(db).parent.mkdir(parents=True, exist_ok=True)
        saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(db))
        await saver.setup()
        return saver
    except Exception:
        logger.exception("could not open checkpoint database at %s; using in-memory state", db)
        return MemorySaver()


def build_graph(checkpointer=None):
    """Compile the PR-review graph with ``checkpointer`` (defaults to ``MemorySaver``)."""
    b = StateGraph(ReviewState)
    for name in ("chat", "fetch_pr", "fetch_error", "security_review",
                 "tests_review", "correctness_review", "approval_gate", "summarize", "render_findings", "handle_action"):
        b.add_node(name, getattr(nodes, name))
    b.add_node("render_tools", ToolNode([nodes.render_review]))

    b.add_conditional_edges(START, nodes.route_intake, ["handle_action", "fetch_pr", "chat"])
    b.add_conditional_edges(
        "fetch_pr", nodes.after_fetch,
        ["fetch_error", "security_review", "tests_review", "correctness_review"],
    )
    b.add_edge(["security_review", "tests_review", "correctness_review"], "approval_gate")
    b.add_edge("approval_gate", "summarize")
    b.add_edge("summarize", "render_findings")
    b.add_conditional_edges(
        "handle_action", lambda st: "render_findings" if st.get("action") else END, ["render_findings", END]
    )
    b.add_edge("render_findings", "render_tools")
    b.add_edge("render_tools", END)
    b.add_edge("chat", END)
    b.add_edge("fetch_error", END)

    return b.compile(checkpointer=checkpointer or MemorySaver())


# Module-level graph for tests and scripts; the server builds its own
# persistent-checkpointed graph in main.py's lifespan.
graph = build_graph()
