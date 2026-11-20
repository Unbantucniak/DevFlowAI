from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.routes.chat import WorkflowTargetResolutionError, _build_conversation_tools, _build_engineering_workflow_runtime, chat_plan
from app.db.models import ActionDraft, PullRequest, Repository, WorkflowRun
from app.db.session import Base
from app.schemas.chat import ChatRequest
from app.schemas.workflow import WorkflowSpec
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.llm.client import LLMClient


@pytest.fixture
def plan_repo():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="demo", name="test", full_name="demo/test")
        other = Repository(owner="demo", name="other", full_name="demo/other")
        db.add_all([repo, other])
        db.flush()
        db.add_all([
            PullRequest(repo_id=repo.id, number=9, title="review", state="open"),
            WorkflowRun(repo_id=repo.id, github_run_id=11, name="build", status="completed", conclusion="failure"),
            WorkflowRun(repo_id=other.id, github_run_id=12, name="other", status="completed", conclusion="failure"),
        ])
        db.commit()
        yield db, repo.id
    engine.dispose()


def model_reply(monkeypatch, reply):
    calls = []

    async def chat_json(self, prompt, payload):
        calls.append(payload)
        return reply

    monkeypatch.setattr(LLMClient, "chat_json", chat_json)
    return calls


@pytest.mark.asyncio
async def test_ci_only_plan_and_replan(monkeypatch, plan_repo):
    db, repo_id = plan_repo
    calls = model_reply(monkeypatch, {"tasks": ["pr_review", "ci_debug"], "excluded": ["pr_review"], "exclusive": True})
    message = "只查 CI #11，不要审查 PR #999"
    result = await chat_plan(ChatRequest(repo_id=repo_id, message=message, context={"tools": ["untrusted"]}), db)
    spec = result["plan"]
    assert calls == [{"message": message}]
    assert [task["task_type"] for task in spec["tasks"]] == ["ci_debug"]
    task = spec["tasks"][0]
    assert task["claim"]["target_id"] == str(db.query(WorkflowRun).filter_by(repo_id=repo_id).one().id)
    assert task["dependencies"] == []
    assert "PR 审查" not in result["answer"].split("验收标准：")[1]
    replanned = await PlannerAgent().replan({
        "spec": spec,
        "observation": {"findings": [{"finding_type": "evidence_gap", "severity": "warning", "message": "缺少证据"}]},
    }, {"has_ci_run": True})
    assert replanned["tasks"] == spec["tasks"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("reply", "message", "status"), [
    ({}, "只查 CI", 503),
    ({"tasks": ["shell"], "excluded": [], "exclusive": False}, "分析仓库", 503),
    ({"tasks": [], "excluded": [], "exclusive": False}, "介绍天气", 422),
    ({"tasks": ["ci_debug"], "excluded": [], "exclusive": True}, "只查 CI #12", 422),
])
async def test_unusable_selection_never_runs_a_fallback(monkeypatch, plan_repo, reply, message, status):
    db, repo_id = plan_repo
    model_reply(monkeypatch, reply)
    with pytest.raises(HTTPException) as failure:
        await chat_plan(ChatRequest(repo_id=repo_id, message=message), db)
    assert failure.value.status_code == status


@pytest.mark.asyncio
async def test_model_cannot_skip_required_safety_review(monkeypatch, plan_repo):
    db, repo_id = plan_repo
    model_reply(monkeypatch, {"tasks": ["pr_review"], "excluded": ["safety_review"], "exclusive": True})
    result = await chat_plan(ChatRequest(repo_id=repo_id, message="post review comment on PR #9"), db)
    spec = result["plan"]
    assert [task["task_type"] for task in spec["tasks"]] == ["pr_review", "safety_review"]
    assert "safety_review" not in spec["excluded_task_types"]
    assert spec["requires_human_confirmation"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "只看最新失败的 CI，跳过 CI #13",
    "查最新失败的 CI，跳过 #13",
    "查 CI #11，跳过 CI #13",
    "跳过 CI #13，查 CI #11",
    "跳过 #13，查最新失败的 CI",
    "检查最新失败的 CI（CI #13 除外）",
])
async def test_ci_exclusions_apply_to_preview_and_execution(monkeypatch, plan_repo, message):
    db, repo_id = plan_repo
    repo = db.get(Repository, repo_id)
    db.add(WorkflowRun(repo_id=repo_id, github_run_id=13, name="newer", status="completed",
                       conclusion="failure", updated_at=datetime.now(timezone.utc) + timedelta(days=1)))
    db.commit()
    model_reply(monkeypatch, {"tasks": ["ci_debug"], "excluded": [], "exclusive": True})
    response = await chat_plan(ChatRequest(repo_id=repo_id, message=message), db)
    spec = WorkflowSpec.model_validate(response["plan"])
    task = spec.tasks[0]
    context, runners = _build_engineering_workflow_runtime(repo, db, message, {})
    assert task.input_hint["workflow_run_number"] == context["workflow_run_number"] == 11
    assert task.claim.target_id == context["workflow_run_id"]
    monkeypatch.setattr(CIDebugAgent, "run", AsyncMock(return_value={}))
    result = await runners["ci_debug"](task, spec, context)
    assert result.citations[0]["id"] == task.claim.target_id


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["查最新失败的 CI，排除 #11", "查 CI #11 和 CI #12", "查 CI #11，但跳过 CI #11"])
async def test_no_eligible_or_ambiguous_ci_returns_422(monkeypatch, plan_repo, message):
    db, repo_id = plan_repo
    model_reply(monkeypatch, {"tasks": ["ci_debug"], "excluded": [], "exclusive": True})
    with pytest.raises(HTTPException) as error:
        await chat_plan(ChatRequest(repo_id=repo_id, message=message), db)
    assert error.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(("message", "tasks", "legacy"), [
    ("post review comment on PR #9", ["pr_review", "safety_review"], False),
    ("draft a comment on PR #9", ["safety_review"], False),
    ("draft a comment on the latest PR", ["safety_review"], False),
    ("给 PR #9 写一条评论草稿", ["safety_review"], False),
    ("draft a comment on the latest PR", ["pr_review", "safety_review"], True),
])
async def test_comment_draft_keeps_previewed_pr(monkeypatch, plan_repo, message, tasks, legacy):
    db, repo_id = plan_repo
    repo = db.get(Repository, repo_id)
    original = db.query(PullRequest).filter_by(repo_id=repo_id, number=9).one()
    if "#9" in message:
        db.add(PullRequest(repo_id=repo_id, number=10, title="newer at preview", state="open",
                           updated_at=datetime.now(timezone.utc) + timedelta(hours=1)))
        db.commit()
    model_reply(monkeypatch, {"tasks": tasks, "excluded": [], "exclusive": True})
    response = await chat_plan(ChatRequest(repo_id=repo_id, message=message), db)
    spec = WorkflowSpec.model_validate(response["plan"])
    db.add(PullRequest(repo_id=repo_id, number=14, title="newer PR", state="open",
                       updated_at=datetime.now(timezone.utc) + timedelta(days=1)))
    db.commit()
    context, runners = _build_engineering_workflow_runtime(repo, db, message, {}, excluded_task_types=tuple(spec.excluded_task_types))
    safety = next(task for task in spec.tasks if task.task_type == "safety_review")
    assert safety.claim.target_id == str(original.id)
    assert "PR #9" in safety.objective
    assert safety.claim.write_intent and spec.requires_human_confirmation
    if legacy:
        safety.claim.target_type = "external_action"
        safety.claim.target_id = str(repo_id)
    result = await runners["safety_review"](safety, spec, context)
    assert "针对 PR #9：" in result.output["draft"]
    assert [ref["id"] for ref in result.citations if ref["type"] == "pull_request"] == [str(original.id)]
    assert result.output["requires_confirmation"]


@pytest.mark.asyncio
async def test_comment_target_missing_at_preview_or_execution(monkeypatch, plan_repo):
    db, repo_id = plan_repo
    repo = db.get(Repository, repo_id)
    model_reply(monkeypatch, {"tasks": ["safety_review"], "excluded": [], "exclusive": True})
    with pytest.raises(HTTPException) as error:
        await chat_plan(ChatRequest(repo_id=repo_id, message="draft a comment on PR #20"), db)
    assert error.value.status_code == 422

    message = "draft a comment on the latest PR"
    response = await chat_plan(ChatRequest(repo_id=repo_id, message=message), db)
    spec = WorkflowSpec.model_validate(response["plan"])
    db.delete(db.query(PullRequest).filter_by(repo_id=repo_id, number=9).one())
    db.add(PullRequest(repo_id=repo_id, number=14, title="replacement", state="open"))
    db.commit()
    context, runners = _build_engineering_workflow_runtime(repo, db, message, {})
    with pytest.raises(WorkflowTargetResolutionError):
        await runners["safety_review"](spec.tasks[0], spec, context)
    # Previews created before a comment-only task pinned a PR must not silently pick its replacement.
    spec.tasks[0].claim.target_type = "external_action"
    spec.tasks[0].claim.target_id = str(repo_id)
    with pytest.raises(WorkflowTargetResolutionError, match="重新生成计划"):
        await runners["safety_review"](spec.tasks[0], spec, context)


@pytest.mark.asyncio
async def test_chat_comment_draft_metadata_matches_body_and_citation(plan_repo):
    db, repo_id = plan_repo
    repo = db.get(Repository, repo_id)
    original = db.query(PullRequest).filter_by(repo_id=repo_id, number=9).one()
    db.add(PullRequest(repo_id=repo_id, number=14, title="newer PR", state="open",
                       updated_at=datetime.now(timezone.utc) + timedelta(days=1)))
    db.commit()
    tools = _build_conversation_tools(repo, None, db)
    response = await tools["safety_agent.classify_action_risk"]({"message": "draft a comment on PR #9"}, {})
    draft = db.query(ActionDraft).one()
    assert draft.status == "pending_confirmation"
    assert "针对 PR #9：" in draft.content
    assert draft.target_id == original.id and draft.meta["pr_number"] == 9
    assert response["citations"][-1]["id"] == str(original.id)
