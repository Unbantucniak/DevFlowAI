import asyncio
import time
from datetime import datetime, timezone
from uuid import UUID
from weakref import WeakValueDictionary

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import Issue, PRFile, PRReviewComment, PullRequest, Repository, WorkflowRun
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.client import github_sync_session
from app.services.github.issues import list_issues
from app.services.github.pull_requests import get_pull_request, list_pull_request_files, list_pull_request_review_comments, list_pull_requests


_sync_locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()
_sync_states: dict[UUID, dict] = {}
SYNC_TIMEOUT_SECONDS = 90
SYNC_CONCURRENCY = 4


def repository_sync_status(repo_id: UUID) -> dict | None:
    state = _sync_states.get(repo_id)
    if state is None:
        return None
    result = {key: value for key, value in state.items() if not key.startswith("_")}
    result["elapsed_seconds"] = round(state.get("_finished", time.monotonic()) - state["_started"], 1)
    return result


def _sync_progress(repo_id: UUID, message: str, **values) -> None:
    _sync_states[repo_id].update(message=message, **values)


def parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def login_of(user: dict | None) -> str | None:
    return user.get("login") if user else None


async def sync_repository(db: Session, repo: Repository, payload: RepoSyncRequest) -> RepoSyncResponse:
    lock = _sync_locks.setdefault(repo.id, asyncio.Lock())
    if lock.locked():
        # The caller can follow /sync-status instead of queueing the same work.
        return RepoSyncResponse(repo_id=repo.id, status="running", synced={}, message="已有同步正在进行，正在跟踪进度。")
    async with lock:
        repo_id = repo.id
        state = {
            "repo_id": str(repo_id), "status": "running", "message": "正在连接 GitHub…",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed": 0, "total": 0, "warnings": 0, "_started": time.monotonic(),
        }
        _sync_states[repo_id] = state
        try:
            async with asyncio.timeout(SYNC_TIMEOUT_SECONDS), github_sync_session():
                result = await _sync_repository_locked(db, repo, payload)
        except TimeoutError as exc:
            db.rollback()
            message = f"GitHub 同步超过 {SYNC_TIMEOUT_SECONDS} 秒，已停止本次同步并保留原有数据，请稍后重试。"
            state.update(status="failed", message=message)
            raise HTTPException(status_code=504, detail=message) from exc
        except httpx.HTTPError as exc:
            db.rollback()
            message = "连接 GitHub 超时或中断，本次同步未完成，原有数据已保留，请稍后重试。"
            state.update(status="failed", message=message)
            raise HTTPException(status_code=502, detail=message) from exc
        except BaseException as exc:
            db.rollback()
            message = str(exc.detail) if isinstance(exc, HTTPException) else "同步未完成，原有数据已保留。"
            state.update(status="failed", message=message)
            raise
        else:
            message = "同步完成"
            if state["warnings"]:
                message += f"；{state['warnings']} 项明细暂未更新，已保留原有内容"
            state.update(status="completed", message=message, synced=result.synced)
            result.message = message
            result.elapsed_seconds = round(time.monotonic() - state["_started"], 1)
            return result
        finally:
            state.update(_finished=time.monotonic(), finished_at=datetime.now(timezone.utc).isoformat())


async def _sync_repository_locked(db: Session, repo: Repository, payload: RepoSyncRequest) -> RepoSyncResponse:
    if repo.provider not in {"github", "github_compatible"}:
        raise HTTPException(status_code=400, detail=f"{repo.provider} sync is not implemented yet")

    synced = {"issues": 0, "pull_requests": 0, "workflow_runs": 0}
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    base_url = repo.api_base_url

    # Complete remote I/O before changing local rows. In particular, never hold
    # DELETE locks while awaiting GitHub: another request may need those rows.
    remote_issues = []
    remote_prs = []
    remote_runs = []
    pr_details = {}
    run_details = {}
    if payload.sync_issues:
        _sync_progress(repo.id, "正在读取 Issue 列表…")
        try:
            remote_issues = await list_issues(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=400, detail=f"GitHub issues sync failed: {exc.response.status_code} {exc.response.text}") from exc

    if payload.sync_pull_requests:
        _sync_progress(repo.id, "正在读取 PR 列表…")
        try:
            remote_prs = await list_pull_requests(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=400, detail=f"GitHub pull requests sync failed: {exc.response.status_code} {exc.response.text}") from exc

    if payload.sync_workflow_runs:
        _sync_progress(repo.id, "正在读取 CI 列表…")
        try:
            remote_runs = await list_workflow_runs(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError:
            remote_runs = []
            _sync_states[repo.id]["warnings"] += 1

    total = (
        len(remote_prs) * 2
        + sum(item.get("state") == "open" and not item.get("merged_at") for item in remote_prs)
        + sum(1 + (item.get("conclusion") == "failure") for item in remote_runs if item.get("id"))
    )
    _sync_progress(repo.id, f"正在读取 PR / CI 明细：0/{total}", total=total)
    semaphore = asyncio.Semaphore(SYNC_CONCURRENCY)

    async def fetch_detail(function, *args):
        async with semaphore:
            try:
                return await function(repo.owner, repo.name, *args, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                _sync_states[repo.id]["warnings"] += 1
                return None
            finally:
                state = _sync_states[repo.id]
                state["completed"] += 1
                state["message"] = f"正在读取 PR / CI 明细：{state['completed']}/{total}"

    async def fetch_pr(item):
        number = item.get("number", 0)
        listed_sha = (item.get("head") or {}).get("sha")
        files = await fetch_detail(list_pull_request_files, number)
        comments = await fetch_detail(list_pull_request_review_comments, number)
        mergeable = None
        if item.get("state") == "open" and not item.get("merged_at"):
            detail = await fetch_detail(get_pull_request, number)
            if detail is not None:
                mergeable = detail.get("mergeable")
                # A PR can be merged between fetching the list and its details.
                for key in ("state", "merged_at", "updated_at", "head", "base"):
                    if key in detail:
                        item[key] = detail[key]
                if listed_sha and listed_sha != (item.get("head") or {}).get("sha"):
                    # Do not associate a diff fetched during a push with the new head.
                    files = None
                    _sync_states[repo.id]["warnings"] += 1
        pr_details[number] = (files, comments, mergeable)

    async def fetch_run(item):
        run_id = item.get("id")
        jobs = logs = None
        if run_id:
            jobs = await fetch_detail(list_workflow_run_jobs, run_id)
            if item.get("conclusion") == "failure":
                logs = await fetch_detail(get_workflow_run_logs, run_id)
        run_details[run_id] = (jobs, logs)

    # Await every worker before applying writes or closing the shared HTTP client.
    results = await asyncio.gather(
        *(fetch_pr(item) for item in remote_prs),
        *(fetch_run(item) for item in remote_runs),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    _sync_progress(repo.id, "远程数据已读取，正在保存更新…")

    if payload.sync_issues:
        for item in remote_issues:
            issue = db.query(Issue).filter(Issue.repo_id == repo.id, Issue.github_issue_id == item.get("id")).one_or_none()
            if issue is None:
                issue = Issue(repo_id=repo.id, github_issue_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(issue)
            issue.number = item.get("number", issue.number)
            issue.title = item.get("title") or issue.title
            issue.body = item.get("body")
            issue.state = item.get("state") or "open"
            issue.labels = [label.get("name") for label in item.get("labels", []) if label.get("name")]
            issue.author = login_of(item.get("user"))
            issue.assignees = [login_of(user) for user in item.get("assignees", []) if login_of(user)]
            issue.created_at = parse_github_datetime(item.get("created_at"))
            issue.updated_at = parse_github_datetime(item.get("updated_at"))
            issue.closed_at = parse_github_datetime(item.get("closed_at"))
            synced["issues"] += 1

    if payload.sync_pull_requests:
        for item in remote_prs:
            pr = db.query(PullRequest).filter(PullRequest.repo_id == repo.id, PullRequest.github_pr_id == item.get("id")).one_or_none()
            if pr is None:
                pr = PullRequest(repo_id=repo.id, github_pr_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(pr)
                db.flush()
            pr.number = item.get("number", pr.number)
            pr.title = item.get("title") or pr.title
            pr.body = item.get("body")
            pr.state = item.get("state") or "open"
            pr.author = login_of(item.get("user"))
            pr.base_branch = (item.get("base") or {}).get("ref")
            pr.head_branch = (item.get("head") or {}).get("ref")
            pr.merged_at = parse_github_datetime(item.get("merged_at"))
            pr.created_at = parse_github_datetime(item.get("created_at"))
            pr.updated_at = parse_github_datetime(item.get("updated_at"))

            files, review_comments, mergeable = pr_details[pr.number]
            # An old cached patch must never be labelled with a newly fetched head.
            pr.head_sha = (item.get("head") or {}).get("sha") if files is not None else None
            pr.mergeable = mergeable if isinstance(mergeable, bool) else None
            if files is not None:
                db.query(PRFile).filter(PRFile.pr_id == pr.id).delete()
            for file_item in files or []:
                db.add(
                    PRFile(
                        pr_id=pr.id,
                        filename=file_item.get("filename") or "",
                        status=file_item.get("status") or "modified",
                        additions=file_item.get("additions") or 0,
                        deletions=file_item.get("deletions") or 0,
                        patch=file_item.get("patch"),
                    )
                )
            if review_comments is not None:
                db.query(PRReviewComment).filter(PRReviewComment.pr_id == pr.id).delete()
            for comment_item in review_comments or []:
                db.add(
                    PRReviewComment(
                        pr_id=pr.id,
                        github_comment_id=comment_item.get("id"),
                        body=comment_item.get("body"),
                        path=comment_item.get("path"),
                        line=comment_item.get("line") or comment_item.get("original_line"),
                        author=login_of(comment_item.get("user")),
                        created_at=parse_github_datetime(comment_item.get("created_at")),
                        updated_at=parse_github_datetime(comment_item.get("updated_at")),
                    )
                )
            synced["pull_requests"] += 1

    if payload.sync_workflow_runs:
        for item in remote_runs:
            run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.github_run_id == item.get("id")).one_or_none()
            if run is None:
                run = WorkflowRun(repo_id=repo.id, github_run_id=item.get("id"), name=item.get("name") or "workflow")
                db.add(run)
            run.name = item.get("name") or run.name
            run.status = item.get("status") or "unknown"
            run.conclusion = item.get("conclusion")
            run.html_url = item.get("html_url")
            run.created_at = parse_github_datetime(item.get("created_at"))
            run.updated_at = parse_github_datetime(item.get("updated_at"))
            jobs, logs = run_details[item.get("id")]
            if jobs is not None:
                run.jobs = [
                    {
                        "id": job.get("id"),
                        "name": job.get("name"),
                        "status": job.get("status"),
                        "conclusion": job.get("conclusion"),
                        "html_url": job.get("html_url"),
                        "started_at": job.get("started_at"),
                        "completed_at": job.get("completed_at"),
                        "steps": [
                            {
                                "name": step.get("name"),
                                "status": step.get("status"),
                                "conclusion": step.get("conclusion"),
                                "number": step.get("number"),
                            }
                            for step in job.get("steps", [])
                        ],
                    }
                    for job in jobs
                ]
            if logs is not None:
                run.logs_text = logs
            synced["workflow_runs"] += 1

    repo.last_sync_at = datetime.now(timezone.utc)
    db.flush()
    db.commit()
    return RepoSyncResponse(repo_id=repo.id, status="completed", synced=synced)


async def sync_repository_by_id(repo_id: UUID, payload: RepoSyncRequest) -> RepoSyncResponse | None:
    from app.db.session import SessionLocal
    from app.services.code_analysis import queue_code_sync, sync_repository_code_analysis

    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        if (repo.code_sync or {}).get("status") in {"pending", "running"}:
            return RepoSyncResponse(repo_id=repo.id, status="running", synced={}, code_sync=repo.code_sync,
                                    message="源码更新正在进行，跳过重复同步。")
        result = await sync_repository(db, repo, payload)
        if result.status == "completed":
            queue_code_sync(db, repo)
            await sync_repository_code_analysis(db, repo)
            result.code_sync = repo.code_sync
            result.message = f"{result.message}；{repo.code_sync['message']}"
        return result
    finally:
        db.close()
