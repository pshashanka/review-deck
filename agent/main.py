import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the demo project root (one level up from agent/) BEFORE
# importing src.agent so COPILOTKIT_AGENT_MODEL and the provider key are in the
# environment before any node builds its chat model (models are created lazily, per call).
_demo_root = Path(__file__).parent.parent
for env_path in (_demo_root / ".env", Path(".env")):
    if env_path.is_file():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI
import uvicorn
from copilotkit import LangGraphAGUIAgent
from src.agent import build_graph, enter_checkpointer
from src.endpoint import add_cancellable_endpoint, drain_in_flight_runs, reset_in_flight_runs

# Filled by the lifespan; the endpoint (registered at import) resolves it per request.
_agent: LangGraphAGUIAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once per process with a persistent SQLite checkpointer.

    The async SQLite saver must be created inside the running event loop (it
    binds its connection to that loop), so it cannot be a module-level import
    like the old ``MemorySaver`` was. Threads, pending approvals and
    posted-comment status therefore survive a uvicorn reload.
    """
    global _agent
    async with AsyncExitStack() as stack:
        reset_in_flight_runs()
        checkpointer = await enter_checkpointer(stack)
        _agent = LangGraphAGUIAgent(
            name="sample_agent",
            description="An example agent to use as a starting point for your own agent.",
            graph=build_graph(checkpointer=checkpointer),
        )
        try:
            yield
        finally:
            _agent = None
            # Cancel and await in-flight runs before the stack closes the
            # checkpoint connection, so no task writes to a closed database.
            await drain_in_flight_runs()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


add_cancellable_endpoint(app=app, agent_factory=lambda: _agent, path="/")


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8123"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
if __name__ == "__main__":
    main()
