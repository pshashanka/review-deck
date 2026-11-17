"""PR-review graph.

START -> route_intake -> handle_action | fetch_pr | chat
fetch_pr -> [security_review, tests_review, correctness_review] (parallel) -> approval_gate -> summarize
         -> render_findings -> render_tools (ToolNode: render_review) -> END
fetch_pr (error) -> fetch_error -> END
handle_action -> render_findings (only when a finding changed) | END;  chat -> END
"""

import logging
import os
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.review import nodes
from src.review.state import ReviewState

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DB = "checkpoints.db"


def _checkpoint_db_path() -> str:
    db = (os.getenv("LANGGRAPH_CHECKPOINT_DB") or "").strip()
    if db:
        return db
    return str(Path(__file__).resolve().parent.parent / ".checkpoints" / DEFAULT_CHECKPOINT_DB)


async def build_persistent_checkpointer():
    """SQLite-backed checkpointer so threads, pending approvals and posted
    comments survive an agent restart (uvicorn runs with reload, and the
    memory saver loses everything on every process bounce).

    ``LANGGRAPH_CHECKPOINT_DB`` overrides the database location (use
    ``:memory:`` to go back to an in-memory store); the default lives in
    ``agent/.checkpoints/checkpoints.db``. Falls back to ``MemorySaver`` when
    the async sqlite package is missing or the database can't be opened.

    Must be called from inside a running event loop: the async saver binds its
    connection to the loop it is created on.
    """
    db = _checkpoint_db_path()
    try:
        import aiosqlite

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        return MemorySaver()

    try:
        if db != ":memory:":
            Path(db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(db)
        return AsyncSqliteSaver(conn)
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
