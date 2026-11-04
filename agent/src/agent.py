"""PR-review graph.

START -> route_intake -> handle_action | fetch_pr | chat
fetch_pr -> [security_review, tests_review, correctness_review] (parallel) -> approval_gate -> summarize
         -> render_findings -> render_tools (ToolNode: render_review) -> END
fetch_pr (error) -> fetch_error -> END
handle_action -> render_findings (only when a finding changed) | END;  chat -> END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.review import nodes
from src.review.state import ReviewState

_b = StateGraph(ReviewState)
for name in ("chat", "fetch_pr", "fetch_error", "security_review",
             "tests_review", "correctness_review", "approval_gate", "summarize", "render_findings", "handle_action"):
    _b.add_node(name, getattr(nodes, name))
_b.add_node("render_tools", ToolNode([nodes.render_review]))

_b.add_conditional_edges(START, nodes.route_intake, ["handle_action", "fetch_pr", "chat"])
_b.add_conditional_edges(
    "fetch_pr", nodes.after_fetch,
    ["fetch_error", "security_review", "tests_review", "correctness_review"],
)
_b.add_edge(["security_review", "tests_review", "correctness_review"], "approval_gate")
_b.add_edge("approval_gate", "summarize")
_b.add_edge("summarize", "render_findings")
_b.add_conditional_edges(
    "handle_action", lambda st: "render_findings" if st.get("action") else END, ["render_findings", END]
)
_b.add_edge("render_findings", "render_tools")
_b.add_edge("render_tools", END)
_b.add_edge("chat", END)
_b.add_edge("fetch_error", END)

graph = _b.compile(checkpointer=MemorySaver())
