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
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

POLL_SECONDS = 0.5
_DONE = object()


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


def add_cancellable_endpoint(app: FastAPI, agent: LangGraphAgent, path: str = "/") -> None:
    @app.post(path)
    async def agent_endpoint(input_data: RunAgentInput, request: Request):
        encoder = EventEncoder(accept=request.headers.get("accept"))
        request_agent = agent.clone()  # per-request state, as in ag_ui_langgraph

        async def event_generator():
            async for event in iter_until_disconnect(request_agent.run(input_data), request.is_disconnected):
                yield encoder.encode(event)

        return StreamingResponse(event_generator(), media_type=encoder.get_content_type())

    @app.get(f"{path.rstrip('/')}/health")
    def health():
        return {"status": "ok", "agent": {"name": agent.name}}
