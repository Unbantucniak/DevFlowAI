import asyncio
import os
from contextlib import asynccontextmanager

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Document, Issue, PRFile, PRReviewComment, PullRequest, Repository, WorkflowRun
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import repository_sync_status, sync_repository


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_sync_repository_persists_facts_without_triggering_rag(monkeypatch) -> None:
    async def fake_list_issues(*_args, **_kwargs):
        return [
            {
                "id": 7800,
                "number": 78,
                "title": "Error: Could not import 'app'.",
                "body": "Realistic issue body",
                "state": "open",
                "labels": [{"name": "bug"}],
                "user": {"login": "contributor"},
                "assignees": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "closed_at": None,
            }
        ]

    def fail_if_vector_runtime_is_used(*_args, **_kwargs):
        raise AssertionError("GitHub fact sync must not depend on embeddings or Milvus")

    monkeypatch.setattr("app.services.repos_sync.list_issues", fake_list_issues)
    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.delete_milvus_documents", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.upsert_milvus_documents", fail_if_vector_runtime_is_used)

    db = _db_session()
    repo = Repository(
        owner="openai",
        name="openai-quickstart-python",
        full_name="openai/openai-quickstart-python",
        provider="github",
        api_base_url="https://api.github.com",
    )
    db.add(repo)
    db.commit()

    result = await sync_repository(
        db,
        repo,
        RepoSyncRequest(sync_pull_requests=False, sync_workflow_runs=False, limit=1),
    )

    issue = db.query(Issue).filter(Issue.repo_id == repo.id).one()
    assert result.status == "completed"
    assert result.synced == {"issues": 1, "pull_requests": 0, "workflow_runs": 0}
    assert issue.number == 78
    assert issue.title == "Error: Could not import 'app'."
    assert db.query(Document).filter(Document.repo_id == repo.id).count() == 0


def _repo_with_pull_request():
    db = _db_session()
    repo = Repository(owner="test", name="repo", full_name="test/repo", provider="github")
    db.add(repo)
    db.flush()
    pr = PullRequest(repo_id=repo.id, github_pr_id=123, number=1, title="Existing PR")
    db.add(pr)
    db.flush()
    db.add(PRFile(pr_id=pr.id, filename="old.py", status="modified"))
    db.add(PRReviewComment(pr_id=pr.id, github_comment_id=456, body="Existing review"))
    db.commit()
    return db, repo, pr


@pytest.mark.asyncio
async def test_overlapping_syncs_follow_existing_work_without_queueing_a_second_run(monkeypatch):
    db, repo, _ = _repo_with_pull_request()
    other_db = Session(bind=db.get_bind())
    other_repo = other_db.get(Repository, repo.id)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def list_prs(*args, **kwargs):
        calls.append("fetch")
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return []

    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", list_prs)
    payload = RepoSyncRequest(sync_issues=False, sync_workflow_runs=False)
    first = asyncio.create_task(sync_repository(db, repo, payload))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(sync_repository(other_db, other_repo, payload))
        duplicate = await asyncio.wait_for(second, timeout=1)
        assert calls == ["fetch"]
        assert duplicate.status == "running"
        assert not first.done()
        assert repository_sync_status(repo.id)["status"] == "running"
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, *([second] if second else [])), timeout=2)
        other_db.close()
        db.close()
    assert calls == ["fetch"]


@pytest.mark.asyncio
async def test_remote_io_finishes_before_any_database_writes(monkeypatch):
    db, repo, pr = _repo_with_pull_request()
    writes = []

    def track_writes(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            writes.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", track_writes)

    async def list_prs(*args, **kwargs):
        assert not writes
        return [{"id": 123, "number": 1, "title": "Updated PR", "state": "open"}]

    async def list_files(*args, **kwargs):
        assert not writes, "A database write must not hold locks while awaiting GitHub"
        await asyncio.sleep(0)
        return [{"filename": "new.py", "status": "added"}]

    async def list_comments(*args, **kwargs):
        assert not writes
        return [{"id": 789, "body": "New review"}]

    async def pr_detail(*args, **kwargs):
        assert not writes
        return {"mergeable": True}

    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", list_prs)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", list_files)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_review_comments", list_comments)
    monkeypatch.setattr("app.services.repos_sync.get_pull_request", pr_detail)
    result = await sync_repository(db, repo, RepoSyncRequest(sync_issues=False, sync_workflow_runs=False))
    assert result.synced["pull_requests"] == 1
    assert writes
    assert db.query(PRFile).filter_by(pr_id=pr.id).one().filename == "new.py"
    assert db.query(PRReviewComment).filter_by(pr_id=pr.id).one().body == "New review"
    db.close()


@pytest.mark.asyncio
async def test_failed_fetch_preserves_cached_records_and_allows_retry(monkeypatch):
    db, repo, pr = _repo_with_pull_request()
    pr.head_sha = "a" * 40
    db.commit()

    async def list_prs(*args, **kwargs):
        return [{"id": 123, "number": 1, "title": "Updated PR", "head": {"sha": "b" * 40}}]

    async def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("GitHub timed out")

    async def unavailable(*args, **kwargs):
        request = httpx.Request("GET", "https://api.github.com/test")
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError("Unavailable", request=request, response=response)

    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", list_prs)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", timeout)
    payload = RepoSyncRequest(sync_issues=False, sync_workflow_runs=False)
    with pytest.raises(HTTPException) as error:
        await sync_repository(db, repo, payload)
    assert error.value.status_code == 502
    assert repository_sync_status(repo.id)["status"] == "failed"
    assert db.get(PullRequest, pr.id).title == "Existing PR"
    assert db.query(PRFile).filter_by(pr_id=pr.id).one().filename == "old.py"

    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", unavailable)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_review_comments", unavailable)
    result = await asyncio.wait_for(sync_repository(db, repo, payload), timeout=2)
    assert result.status == "completed"
    assert db.query(PRFile).filter_by(pr_id=pr.id).one().filename == "old.py"
    assert db.query(PRReviewComment).filter_by(pr_id=pr.id).one().body == "Existing review"
    assert db.get(PullRequest, pr.id).head_sha is None
    db.close()


@pytest.mark.asyncio
async def test_pr_push_during_sync_does_not_bind_old_patch_to_new_head(monkeypatch):
    db, repo, pr = _repo_with_pull_request()

    async def prs(*args, **kwargs):
        return [{"id": 123, "number": 1, "title": "Updated PR", "state": "open", "head": {"sha": "a" * 40}}]

    async def detail(*args, **kwargs):
        return {"state": "open", "head": {"sha": "b" * 40}}

    async def files(*args, **kwargs):
        return [{"filename": "uncertain.py", "patch": "+uncertain"}]

    async def comments(*args, **kwargs):
        return []

    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", prs)
    monkeypatch.setattr("app.services.repos_sync.get_pull_request", detail)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", files)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_review_comments", comments)
    await sync_repository(db, repo, RepoSyncRequest(sync_issues=False, sync_workflow_runs=False))
    assert db.get(PullRequest, pr.id).head_sha is None
    assert db.query(PRFile).filter_by(pr_id=pr.id).one().filename == "old.py"
    assert repository_sync_status(repo.id)["warnings"] == 1
    db.close()


@pytest.mark.asyncio
async def test_full_refresh_fetches_all_details_with_bounded_parallelism(monkeypatch):
    db, repo, _ = _repo_with_pull_request()
    active = peak = 0

    async def issues(*args, **kwargs):
        return []

    async def prs(*args, **kwargs):
        return [{"id": 1000 + n, "number": n, "title": f"PR {n}"} for n in range(1, 13)]

    async def runs(*args, **kwargs):
        return [{"id": 2000 + n, "name": f"CI {n}", "status": "completed", "conclusion": "success"} for n in range(50)]

    async def details(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.005)
            return []
        finally:
            active -= 1

    monkeypatch.setattr("app.services.repos_sync.list_issues", issues)
    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", prs)
    monkeypatch.setattr("app.services.repos_sync.list_workflow_runs", runs)
    for name in ["list_pull_request_files", "list_pull_request_review_comments", "list_workflow_run_jobs"]:
        monkeypatch.setattr(f"app.services.repos_sync.{name}", details)
    result = await sync_repository(db, repo, RepoSyncRequest(limit=50))
    assert result.synced == {"issues": 0, "pull_requests": 12, "workflow_runs": 50}
    assert 1 < peak <= 4
    assert active == 0
    progress = repository_sync_status(repo.id)
    assert progress["status"] == "completed"
    assert progress["completed"] == progress["total"] == 74
    assert db.query(WorkflowRun).filter_by(repo_id=repo.id).count() == 50
    db.close()


@pytest.mark.asyncio
async def test_sync_deadline_cancels_workers_and_preserves_existing_data(monkeypatch):
    db, repo, pr = _repo_with_pull_request()
    active = 0

    @asynccontextmanager
    async def no_network_client():
        yield

    async def prs(*args, **kwargs):
        return [{"id": 123, "number": 1, "title": "Must not be saved"}]

    async def stalled(*args, **kwargs):
        nonlocal active
        active += 1
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    monkeypatch.setattr("app.services.repos_sync.github_sync_session", no_network_client)
    monkeypatch.setattr("app.services.repos_sync.SYNC_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", prs)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", stalled)
    with pytest.raises(HTTPException) as error:
        await sync_repository(db, repo, RepoSyncRequest(sync_issues=False, sync_workflow_runs=False))
    assert error.value.status_code == 504
    assert active == 0
    assert repository_sync_status(repo.id)["status"] == "failed"
    assert db.get(PullRequest, pr.id).title == "Existing PR"
    assert db.query(PRFile).filter_by(pr_id=pr.id).one().filename == "old.py"
    db.close()


@pytest.mark.asyncio
async def test_sync_reuses_one_http_client_for_detail_requests(monkeypatch):
    from app.services.github.client import GitHubClient, github_sync_session

    original_client = httpx.AsyncClient
    clients = []
    requests = []

    async def respond(request):
        requests.append(request.url.path)
        return httpx.Response(200, json=[])

    def create_client(**kwargs):
        client = original_client(**kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", create_client)
    async with github_sync_session():
        await asyncio.gather(GitHubClient().get("/first"), GitHubClient().get("/second"))
        await GitHubClient().get_bytes("/logs")
    assert len(clients) == 1
    assert clients[0].is_closed
    assert set(requests) == {"/first", "/second", "/logs"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mergeable", [True, False, None])
async def test_sync_preserves_github_mergeability_without_guessing_from_text(monkeypatch, mergeable):
    from app.schemas.pull_requests import PullRequestResponse

    db, repo, pr = _repo_with_pull_request()
    pr.mergeable = False
    db.commit()

    async def prs(*args, **kwargs):
        return [{"id": 123, "number": 1, "title": "Handle conflict errors", "state": "open"}]

    async def detail(*args, **kwargs):
        return {"state": "open", "mergeable": mergeable}

    async def empty(*args, **kwargs):
        return []

    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", prs)
    monkeypatch.setattr("app.services.repos_sync.get_pull_request", detail)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", empty)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_review_comments", empty)
    await sync_repository(db, repo, RepoSyncRequest(sync_issues=False, sync_workflow_runs=False))
    db.expire_all()
    saved = db.get(PullRequest, pr.id)
    assert saved.mergeable is mergeable
    assert PullRequestResponse.model_validate(saved).mergeable is mergeable
    db.close()


@pytest.mark.asyncio
async def test_sync_uses_new_merge_status_when_pr_changes_during_refresh(monkeypatch):
    db, repo, pr = _repo_with_pull_request()

    async def prs(*args, **kwargs):
        return [{"id": 123, "number": 1, "title": "Resolve conflicts", "state": "open"}]

    async def detail(*args, **kwargs):
        return {"state": "closed", "merged_at": "2026-11-15T14:37:14Z", "mergeable": None}

    async def empty(*args, **kwargs):
        return []

    monkeypatch.setattr("app.services.repos_sync.list_pull_requests", prs)
    monkeypatch.setattr("app.services.repos_sync.get_pull_request", detail)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_files", empty)
    monkeypatch.setattr("app.services.repos_sync.list_pull_request_review_comments", empty)
    await sync_repository(db, repo, RepoSyncRequest(sync_issues=False, sync_workflow_runs=False))
    db.expire_all()
    saved = db.get(PullRequest, pr.id)
    assert saved.state == "closed"
    assert saved.merged_at.isoformat().startswith("2026-11-15T14:37:14")
    assert saved.mergeable is None
    db.close()
