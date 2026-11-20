from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.routes import chat
from app.db.models import Base, PRFile, PullRequest, Repository, WorkflowRun
from app.services.agents.chat_agent import ChatAgent
from app.services.github import actions


SHA = "6" * 40


@pytest.fixture
def scope(tmp_path, monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="team", name="repo", full_name="team/repo", local_path=str(tmp_path))
        db.add(repo)
        db.flush()
        pr = PullRequest(repo_id=repo.id, number=14, title="Fix startup race", body="Current PR description",
                         head_sha=SHA, state="closed", merged_at=datetime.now(timezone.utc))
        db.add(pr)
        db.flush()
        patch = "@@ -364,3 +364,5 @@\n context\n context\n context\n+R83 fix\n+current evidence\n"
        db.add(PRFile(pr_id=pr.id, filename="notes.md", status="modified", patch=patch))
        db.add(WorkflowRun(repo_id=repo.id, github_run_id=1, name="Old failure", status="completed", conclusion="failure"))
        db.commit()
        (tmp_path / "notes.md").write_text("historic context\n", encoding="utf-8")
        monkeypatch.setattr(chat, "prepare_pr_worktree_by_id", lambda *args: {
            "status": "ready", "snapshot_path": str(tmp_path), "commit_sha": SHA,
        })
        async def review(*args):
            return {"summary": "reviewed", "risk_points": [], "test_suggestions": []}
        monkeypatch.setattr(chat.PRReviewAgent, "run", review)
        yield db, repo, pr, tmp_path


@pytest.mark.asyncio
async def test_search_pattern_survives_normalization_and_tool_adapter(scope):
    db, repo, pr, root = scope
    (root / "worker.py").write_text("semantic_version = 1\n审查 = True\nsemantic_version = 2\n", encoding="utf-8")
    agent = ChatAgent(model=object())
    args = agent._normalize_action_input("workspace.search_code", {"pattern": "semantic_version", "query": "审查PR14"}, {"message": "审查PR14", "context": {}})
    result = await chat._build_conversation_tools(repo, None, db, "审查PR14")["workspace.search_code"](args, {})
    assert args["query"] == "semantic_version"
    assert "worker.py:1:" in result["answer"] and "worker.py:3:" in result["answer"]
    assert "worker.py:2:" not in result["answer"]


@pytest.mark.asyncio
async def test_pr_returns_authoritative_metadata_patch_and_changed_lines(scope):
    db, repo, pr, _ = scope
    result = await chat._build_conversation_tools(repo, None, db, "审查PR14")["pr_review_agent.analyze_pr"]({}, {"server_context": "PR #14：50万预算实际验证"})
    answer = result["answer"]
    assert "标题：Fix startup race" in answer
    assert "状态：closed；已合并：" in answer
    assert "Current PR description" in answer
    assert "新增/替换的新版本行：367-368" in answer
    assert "+R83 fix" in answer and "@@ -364,3 +364,5 @@" in answer
    assert "50万预算实际验证" not in answer
    assert SHA in answer


def test_diff_ranges_keep_deletions_and_missing_patches_explicit():
    pr = SimpleNamespace(files=[
        SimpleNamespace(filename="gone.py", status="removed", patch="@@ -1,2 +0,0 @@\n-old\n-old2"),
        SimpleNamespace(filename="binary.png", status="modified", patch=None),
        SimpleNamespace(filename="one.py", status="modified", patch="@@ -7 +7 @@\n-old\n+new"),
    ])
    result = chat._pr_diff_evidence(pr)
    assert "新版本行：无；删除/替换的旧版本行：1-2" in result
    assert "新版本行：7；删除/替换的旧版本行：7" in result
    assert "patch 未提供" in result


def run_record(identifier=22, **kwargs):
    return {"id": identifier, "name": "Tests", "workflow_id": 5, "run_number": identifier,
            "run_attempt": 1, "head_sha": SHA, "head_branch": "feature", "event": "pull_request",
            "pull_requests": [{"number": 14}], "status": "completed", "conclusion": "success",
            "html_url": f"https://github.com/team/repo/actions/runs/{identifier}", **kwargs}


@pytest.mark.asyncio
async def test_ci_filters_commit_pr_and_uses_latest_run_including_pending(monkeypatch):
    async def get(self, path, params):
        assert params["head_sha"] == SHA and "status" not in params
        return {"total_count": 5, "workflow_runs": [
            run_record(20, conclusion="failure"), run_record(21),
            run_record(22, status="queued", conclusion=None),
            run_record(23, head_sha="old"), run_record(24, pull_requests=[{"number": 13}]),
        ]}
    monkeypatch.setattr(actions.GitHubClient, "get", get)
    result = await actions.list_pr_workflow_runs("team", "repo", 14, SHA)
    assert result["complete"] and [r["id"] for r in result["runs"]] == [22]
    assert result["runs"][0]["status"] == "queued"


@pytest.mark.asyncio
async def test_ci_paginates_and_keeps_different_workflow_events(monkeypatch):
    pages = []
    async def get(self, path, params):
        pages.append(params["page"])
        return {"total_count": 101, "workflow_runs":
                [run_record(1, head_sha="old")] * 99 + [run_record(22)] if params["page"] == 1 else
                [run_record(23, event="push", pull_requests=[])]}
    monkeypatch.setattr(actions.GitHubClient, "get", get)
    result = await actions.list_pr_workflow_runs("team", "repo", 14, SHA)
    assert pages == [1, 2] and result["complete"]
    assert {r["event"] for r in result["runs"]} == {"push", "pull_request"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "queued", "empty", "unavailable", "incomplete"])
async def test_scoped_ci_never_substitutes_repository_old_failure(scope, monkeypatch, mode):
    db, repo, pr, _ = scope
    async def read(owner, name, number, sha, **kwargs):
        assert number == 14 and sha == SHA
        if mode == "unavailable":
            raise httpx.ConnectError("offline")
        return {"complete": mode != "incomplete", "runs": [] if mode == "empty" else [
            run_record(status="queued", conclusion=None) if mode == "queued" else run_record()
        ]}
    monkeypatch.setattr(chat, "list_pr_workflow_runs", read)
    async def no_debug(*args):
        pytest.fail("A successful, pending or unknown CI must not invoke failure analysis")
    monkeypatch.setattr(chat.CIDebugAgent, "run", no_debug)
    result = await chat._build_conversation_tools(repo, None, db, "审查PR14")["ci_debug_agent.analyze_workflow_run"]({}, {})
    assert SHA in result["answer"] and "Old failure" not in result["answer"]
    assert "均已通过" in result["answer"] if mode == "success" else "均已通过" not in result["answer"]
    if mode in {"empty", "unavailable"}:
        assert "CI 状态未知" in result["answer"]


@pytest.mark.asyncio
async def test_scoped_failure_loads_only_matching_run_logs(scope, monkeypatch):
    db, repo, pr, _ = scope
    async def read(*args, **kwargs):
        return {"complete": True, "runs": [run_record(conclusion="failure")]}
    async def jobs(owner, name, run_id, **kwargs):
        assert run_id == 22
        return [{"name": "Windows", "conclusion": "failure"}]
    async def logs(owner, name, run_id, **kwargs):
        assert run_id == 22
        return "CURRENT_COMMIT_FAILURE"
    async def debug(self, payload, context):
        assert payload["logs_text"] == "CURRENT_COMMIT_FAILURE"
        return {"failure_summary": "matched failure", "debug_steps": []}
    monkeypatch.setattr(chat, "list_pr_workflow_runs", read)
    monkeypatch.setattr(chat, "list_workflow_run_jobs", jobs)
    monkeypatch.setattr(chat, "get_workflow_run_logs", logs)
    monkeypatch.setattr(chat.CIDebugAgent, "run", debug)
    result = await chat._build_conversation_tools(repo, None, db, "审查PR14")["ci_debug_agent.analyze_workflow_run"]({}, {})
    assert "run 22 的失败诊断：matched failure" in result["answer"]
    assert "Old failure" not in result["answer"]
