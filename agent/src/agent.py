"""PR-review graph.

START -> route_intake -> handle_action | fetch_pr | chat
fetch_pr -> [lint_review, security_review, tests_review] (parallel) -> approval_gate -> summarize
         -> render_findings -> render_tools (ToolNode: render_review) -> END
fetch_pr (error) -> fetch_error -> END
handle_action -> render_findings;  chat -> END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.review import nodes
from src.review.state import ReviewState

_b = StateGraph(ReviewState)
for name in ("chat", "fetch_pr", "fetch_error", "lint_review", "security_review",
             "tests_review", "approval_gate", "summarize", "render_findings", "handle_action"):
    _b.add_node(name, getattr(nodes, name))
_b.add_node("render_tools", ToolNode([nodes.render_review]))

_b.add_conditional_edges(START, nodes.route_intake, ["handle_action", "fetch_pr", "chat"])
_b.add_conditional_edges(
    "fetch_pr", nodes.after_fetch,
    ["fetch_error", "lint_review", "security_review", "tests_review"],
)
_b.add_edge(["lint_review", "security_review", "tests_review"], "approval_gate")
_b.add_edge("approval_gate", "summarize")
_b.add_edge("summarize", "render_findings")
_b.add_edge("handle_action", "render_findings")
_b.add_edge("render_findings", "render_tools")
_b.add_edge("render_tools", END)
_b.add_edge("chat", END)
_b.add_edge("fetch_error", END)

graph = _b.compile(checkpointer=MemorySaver())
