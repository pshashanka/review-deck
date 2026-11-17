"""AG-UI FastAPI endpoint that really cancels the run when the client disconnects.

With ``ag_ui_langgraph.add_langgraph_fastapi_endpoint``, a dropped client (the
user pressing Stop) cancels the request's streaming generator, but the LangGraph
node tasks keep running (and billing LLM calls) until they finish: cancellation
delivered inside Starlette's cancel scope does not reach the graph's tasks. Here the
agent runs in its own task, feeding a queue; when the response generator is
cancelled/closed (or ``is_disconnected`` reports a drop, for servers where
Starlette no longer listens for disconnects), that task is cancelled explicitly.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph import LangGraphAgent
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

POLL_SECONDS = 0.5
_DONE = object()

# Producer tasks for in-flight runs. Drained (cancelled + awaited) on shutdown so
# no task writes to a checkpoint connection that is about to close.
_in_flight_tasks: set[asyncio.Task] = set()

# Set once shutdown begins; requests that arrive afterwards are rejected with a
# 503 instead of starting a run against a connection that is about to close.
_shutting_down = asyncio.Event()


def _track(task: asyncio.Task) -> None:
    _in_flight_tasks.add(task)
    task.add_done_callback(_in_flight_tasks.discard)


def reset_in_flight_runs() -> None:
    """Clear the shutdown flag for a new process lifecycle (called at startup)."""
    _shutting_down.clear()


async def drain_in_flight_runs() -> None:
    """Cancel and await every in-flight run producer.

    Called before the checkpoint connection closes so a mid-review shutdown
    cannot leave a task writing to a closed database. Sets ``_shutting_down``
    first so requests that have not started yet are rejected rather than
    starting against the closing connection.
    """
    _shutting_down.set()
    tasks = list(_in_flight_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def iter_until_disconnect(
    events: AsyncIterator[Any],
    is_disconnected: Callable[[], Awaitable[bool]],
    poll: float = POLL_SECONDS,
) -> AsyncIterator[Any]:
    """Yield ``events``; on client disconnect or generator close, cancel the producer task."""
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except Exception as exc:  # re-raised to the consumer
            await queue.put(exc)
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(produce())
    _track(producer)
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), poll)
            except asyncio.TimeoutError:
                if await is_disconnected():
                    logger.info("client disconnected; cancelling the run")
                    return
                continue
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not producer.done():
            logger.info("run cancelled before completion")
            producer.cancel()  # not awaited: we may be inside a cancelled scope


def add_cancellable_endpoint(
    app: FastAPI,
    agent: LangGraphAgent | None = None,
    path: str = "/",
    agent_factory: Callable[[], LangGraphAgent | None] | None = None,
) -> None:
    """Register the AG-UI endpoint.

    Pass either a concrete ``agent`` or an ``agent_factory`` that resolves the
    agent per request (used when the agent is built in the app lifespan). Routes
    are registered once here, so they exist even if the lifespan has not run.
    """

    def resolve() -> LangGraphAgent | None:
        return agent_factory() if agent_factory is not None else agent

    @app.post(path)
    async def agent_endpoint(input_data: RunAgentInput, request: Request):
        # These checks are synchronous (no await), so shutdown cannot interleave
        # between them and the task registration below: a request is either
        # rejected here or its run task is registered before the drain runs.
        if _shutting_down.is_set():
            raise HTTPException(status_code=503, detail="agent is shutting down")
        resolved = resolve()
        if resolved is None:
            raise HTTPException(status_code=503, detail="agent is not ready; try again shortly")

        encoder = EventEncoder(accept=request.headers.get("accept"))
        request_agent = resolved.clone()  # per-request state, as in ag_ui_langgraph

        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        async def run() -> None:
            try:
                async for event in iter_until_disconnect(
                    request_agent.run(input_data), request.is_disconnected
                ):
                    await queue.put(encoder.encode(event))
            except Exception as exc:  # surface to the stream rather than drop it
                await queue.put(exc)
            finally:
                await queue.put(sentinel)

        # Created and tracked synchronously (before any await) so a shutdown
        # drain always sees this run even if the response stream hasn't started.
        task = asyncio.create_task(run())
        _track(task)

        async def stream():
            while True:
                item = await queue.get()
                if item is sentinel:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item

        return StreamingResponse(stream(), media_type=encoder.get_content_type())

    @app.get(f"{path.rstrip('/')}/health")
    def health():
        resolved = resolve()
        return {"status": "ok", "agent": {"name": resolved.name if resolved is not None else None}}
