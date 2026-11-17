import asyncio

import pytest

from src.endpoint import iter_until_disconnect

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_endpoint_state():
    """Reset the shutdown flag and in-flight task set between tests."""
    from src import endpoint

    endpoint.reset_in_flight_runs()
    endpoint._in_flight_tasks.clear()
    yield
    endpoint.reset_in_flight_runs()
    endpoint._in_flight_tasks.clear()


async def never():
    return False


async def test_passes_events_through():
    async def events():
        for i in range(3):
            yield i

    assert [e async for e in iter_until_disconnect(events(), never)] == [0, 1, 2]


async def test_errors_propagate():
    async def events():
        yield 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in iter_until_disconnect(events(), never):
            pass


async def test_disconnect_cancels_producer():
    state = {"cancelled": False}

    async def events():
        try:
            await asyncio.sleep(30)
            yield 1
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def gone():
        return True

    assert [e async for e in iter_until_disconnect(events(), gone, poll=0.01)] == []
    await asyncio.sleep(0.05)
    assert state["cancelled"]


async def test_cancelling_consumer_cancels_producer():
    state = {"cancelled": False, "started": asyncio.Event()}

    async def events():
        state["started"].set()
        try:
            await asyncio.sleep(30)
            yield 1
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def consume():
        async for _ in iter_until_disconnect(events(), never, poll=0.01):
            pass

    t = asyncio.create_task(consume())
    await state["started"].wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    await asyncio.sleep(0.05)
    assert state["cancelled"]


# ------------------------------------------------------------ route tests

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from copilotkit import LangGraphAGUIAgent

from src.agent import graph
from src.endpoint import add_cancellable_endpoint
from src.review import nodes


def _client(monkeypatch):
    class Chat:
        async def ainvoke(self, messages, config=None):
            return AIMessage(content="Send a PR URL.")

    monkeypatch.setattr(nodes, "create_chat_model", lambda *a, **k: Chat())
    app = FastAPI()
    agent = LangGraphAGUIAgent(name="a", description="d", graph=graph)
    add_cancellable_endpoint(app, agent, path="/")
    return TestClient(app), agent


def test_post_streams_run(monkeypatch):
    client, _ = _client(monkeypatch)
    body = {"threadId": "ep1", "runId": "r1", "state": {}, "tools": [], "context": [], "forwardedProps": {},
            "messages": [{"id": "m1", "role": "user", "content": "hello"}]}
    r = client.post("/", json=body, headers={"accept": "text/event-stream"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    types = [json.loads(line[6:])["type"] for line in r.text.splitlines() if line.startswith("data: ")]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"


def test_health_matches_stock(monkeypatch):
    from ag_ui_langgraph import add_langgraph_fastapi_endpoint

    client, agent = _client(monkeypatch)
    stock = FastAPI()
    add_langgraph_fastapi_endpoint(stock, agent, path="/")
    assert client.get("/health").json() == TestClient(stock).get("/health").json()


def test_503_when_agent_not_ready():
    app = FastAPI()
    add_cancellable_endpoint(app, agent_factory=lambda: None, path="/")
    client = TestClient(app)  # no lifespan: the route exists but the agent isn't set
    body = {"threadId": "ep2", "runId": "r2", "state": {}, "tools": [], "context": [],
            "forwardedProps": {}, "messages": [{"id": "m2", "role": "user", "content": "hello"}]}
    r = client.post("/", json=body, headers={"accept": "text/event-stream"})
    assert r.status_code == 503


async def test_drain_cancels_in_flight_runs_and_rejects_new():
    from src import endpoint

    endpoint.reset_in_flight_runs()
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(slow())
    endpoint._track(task)
    await started.wait()

    await endpoint.drain_in_flight_runs()
    assert task.cancelled()
    assert endpoint._shutting_down.is_set()
