import uuid
from threading import get_ident
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import chat as chat_routes
from app.api.routes import search as search_routes
from app.db.session import get_db


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_search", [False, True])
async def test_chat_search_owns_and_closes_its_session_in_worker(monkeypatch, fail_search):
    event_loop_thread = get_ident()
    repo_id = uuid.uuid4()
    context = {"limit": 3, "source_type": "issue"}
    events = []
    sessions = []

    class SearchSession:
        def __init__(self):
            sessions.append(self)
            events.append(("create", get_ident()))

        def __enter__(self):
            events.append(("enter", get_ident()))
            return self

        def __exit__(self, *_args):
            events.append(("close", get_ident()))

    def fake_search(repo, query, worker_context, db):
        events.append(("search", get_ident()))
        assert repo.id == repo_id
        assert query == "token failure"
        assert db is sessions[-1]
        assert worker_context == context
        assert worker_context is not context
        worker_context["limit"] = 1
        if fail_search:
            raise RuntimeError("retrieval failed")
        return "evidence", [], []

    monkeypatch.setattr(chat_routes, "SessionLocal", SearchSession)
    monkeypatch.setattr(chat_routes, "_semantic_search", fake_search)

    if fail_search:
        with pytest.raises(RuntimeError, match="retrieval failed"):
            await chat_routes._semantic_search_async(repo_id, "token failure", context)
    else:
        assert await chat_routes._semantic_search_async(repo_id, "token failure", context) == ("evidence", [], [])

    assert [event for event, _thread in events] == ["create", "enter", "search", "close"]
    assert all(thread != event_loop_thread for _event, thread in events)
    assert len({thread for _event, thread in events}) == 1
    assert context["limit"] == 3


@pytest.mark.asyncio
async def test_search_endpoint_runs_retrieval_off_event_loop(monkeypatch):
    event_loop_thread = get_ident()
    repo_id = uuid.uuid4()
    db = SimpleNamespace(get=lambda _model, _id: SimpleNamespace(id=repo_id))
    calls = []

    def fake_search(search_db, selected_repo_id, query, **kwargs):
        calls.append(get_ident())
        assert search_db is db
        assert selected_repo_id == repo_id
        assert query == "token failure"
        assert kwargs["metadata_filters"] == {"path": "auth.py"}
        return [{"id": "auth", "score": 0.9}]

    monkeypatch.setattr(search_routes, "search_similar_documents", fake_search)
    app = FastAPI()
    app.include_router(search_routes.router, prefix="/search")
    app.dependency_overrides[get_db] = lambda: db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/search", json={"repo_id": str(repo_id), "query": "token failure", "path": "auth.py"})

    assert response.status_code == 200
    assert response.json()["results"] == [{"id": "auth", "score": 0.9}]
    assert calls and all(thread != event_loop_thread for thread in calls)
