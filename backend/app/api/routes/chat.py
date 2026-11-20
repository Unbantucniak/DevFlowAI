import asyncio
import uuid
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, ChatMessage, Conversation, Issue, PullRequest, Repository, WorkflowRun
from app.db.session import SessionLocal, get_db
from app.schemas.chat import ChatHistoryMessage, ChatPlanExecuteRequest, ChatRequest, ChatResponse, PromptRecommendationRequest, PromptRecommendationResponse
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.chat_agent import ChatAgent, encode_sse
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.planner_agent import PlannerAgent, PlanSelectionError, select_plan_intent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.recommendation_agent import RecommendationAgent
from app.services.agents.report_agent import ReportAgent
from app.services.agents.safety_agent import SafetyAgent
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
from app.services.chat_memory import ContextAssembler, EvidenceStore, MessageStore, ProgressiveContextManager, SessionSealer
from app.services.code_search import query_terms, search_checkout_code
from app.services.github.actions import list_pr_workflow_runs, list_workflow_run_jobs, get_workflow_run_logs
from app.services.worktree_manager import prepare_pr_worktree_by_id
from app.services.conversations import ensure_conversation, touch_conversation
from app.services.memory_hub import record_recall_event
from app.services.mcp_client import StdioMcpClient
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.vector_store import add_document
from app.schemas.workflow import WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.skills import SkillRegistry, get_skill_registry
from app.services.permissions import ensure_demo_user, write_audit_log

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
WORKSPACE_SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}
WORKSPACE_TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "package.json", "requirements.txt", "pyproject.toml"}
WORKSPACE_TEXT_EXTENSIONS = {
    ".vue",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".md",
    ".mdx",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
}


@router.get("/workflow-runs")
async def list_workflow_runs(
    repo_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    query = db.query(AgentWorkflowRun)
    if repo_id is not None:
        query = query.filter(AgentWorkflowRun.repo_id == repo_id)
    rows = query.order_by(AgentWorkflowRun.created_at.desc()).limit(limit).all()
    workflow_ids = [row.id for row in rows]
    tasks = (
        db.query(AgentTaskRun)
        .filter(AgentTaskRun.workflow_id.in_(workflow_ids))
        .order_by(AgentTaskRun.started_at.asc(), AgentTaskRun.task_id.asc())
        .all()
        if workflow_ids
        else []
    )
    tasks_by_workflow: dict[uuid.UUID, list[AgentTaskRun]] = {}
    for task in tasks:
        tasks_by_workflow.setdefault(task.workflow_id, []).append(task)
    return [
        {
            "id": str(row.id),
            "repo_id": str(row.repo_id) if row.repo_id else None,
            "conversation_id": str(row.conversation_id) if row.conversation_id else None,
            "goal": row.goal,
            "status": row.status,
            "spec": row.spec_json or {},
            "observation": row.observation_json or {},
            "final_answer": row.final_answer,
            "metrics": row.metrics or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "tasks": [
                {
                    "id": str(task.id),
                    "task_id": task.task_id,
                    "agent_name": task.agent_name,
                    "task_type": task.task_type,
                    "status": task.status,
                    "summary": str((task.result_json or {}).get("summary") or ""),
                    "error": task.error,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                }
                for task in tasks_by_workflow.get(row.id, [])
            ],
        }
        for row in rows
    ]


@router.get("/history", response_model=list[ChatHistoryMessage])
async def chat_history(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    before_message_id: uuid.UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    _repo_or_404(db, repo_id)
    conversation = ensure_conversation(db, repo_id, conversation_id)
    conversation_id = conversation.id
    db.commit()
    rows = [
        item
        for item in MessageStore(db).timeline(
            repo_id,
            conversation_id,
            limit=limit,
            before_message_id=before_message_id,
        )
        if item.role in {"user", "assistant"}
    ]
    return [
        {
            "id": str(item.id),
            "conversation_id": str(item.conversation_id) if item.conversation_id else None,
            "role": item.role,
            "content": item.content,
            "route": item.route,
            "tool_calls": item.tool_calls or [],
            "agent_events": (item.meta or {}).get("agent_events", []),
            "compression_stats": (item.meta or {}).get("compression_stats", {}),
            "progressive_compaction": (item.meta or {}).get("progressive_compaction", {}),
            "model_usage": (item.meta or {}).get("model_usage", []),
            "context_diagnostics": (item.meta or {}).get("context_diagnostics", {}),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in rows
    ]


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到指定仓库。")
    return repo


def _client_context(value: dict[str, Any] | None) -> dict[str, Any]:
    server_owned = {
        "repo_id",
        "conversation_id",
        "tools",
        "server_context",
        "server_messages",
        "evidence",
        "compression_stats",
        "progressive_compaction",
        "registered_skills",
        "skill_description_by_tool",
    }
    return {key: item for key, item in (value or {}).items() if key not in server_owned}


def _item_time(item: Any) -> datetime:
    return item.updated_at or item.created_at or datetime.min


def _join_items(items: list[str], empty_text: str) -> str:
    return "; ".join(items) if items else empty_text


def _source_type_from_message(message: str) -> str | None:
    lowered = message.lower()
    if "issue" in lowered:
        return "issue"
    if "pr" in lowered or "pull request" in lowered or "diff" in lowered:
        return "pull_request"
    if "ci" in lowered or "workflow" in lowered:
        return "workflow_run"
    if any(token in lowered for token in ["knowledge", "知识库", "上传资料", "上传文档"]):
        return "knowledge_file"
    return None


def _risky_action(message: str) -> str | None:
    lowered = message.lower()
    if any(token in lowered for token in ["comment", "reply", "post"]):
        return "comment"
    if any(token in lowered for token in ["create issue", "new issue"]):
        return "create_issue"
    if any(token in lowered for token in ["send", "send_report"]):
        return "send_report"
    if any(token in lowered for token in ["close issue", "label", "write"]):
        return "write_back"
    return None


def _date_window_from_context(context: dict[str, Any]) -> tuple[datetime, datetime]:
    end_at = datetime.now(timezone.utc)
    time_range = str(context.get("time_range") or "7d").lower()
    days = int(time_range[:-1]) if time_range.endswith("d") and time_range[:-1].isdigit() else 7
    start_at = end_at - timedelta(days=max(1, days) - 1)
    if context.get("start_date"):
        start_at = datetime.fromisoformat(str(context["start_date"])).replace(tzinfo=timezone.utc)
    if context.get("end_date"):
        end_at = datetime.fromisoformat(str(context["end_date"])).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    return start_at, end_at


def _within(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    if not value:
        return False
    current = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return start_at <= current <= end_at


def _repo_payload(repo_id: uuid.UUID, db: Session) -> tuple[list[Issue], list[PullRequest], list[WorkflowRun]]:
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    return issues, prs, runs


def _recommendation_payload(
    repo: Repository | None,
    issues: list[Issue],
    prs: list[PullRequest],
    runs: list[WorkflowRun],
    exclude: list[str],
    limit: int,
) -> dict[str, Any]:
    return {
        "repo": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "issues": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "labels": item.labels or [],
                "assignees": item.assignees or [],
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "closed_at": item.closed_at,
            }
            for item in issues
        ],
        "pull_requests": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "author": item.author,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "merged_at": item.merged_at,
            }
            for item in prs
        ],
        "workflow_runs": [
            {
                "id": str(item.id),
                "name": item.name,
                "status": item.status,
                "conclusion": item.conclusion,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in runs
        ],
        "exclude": exclude,
        "limit": limit,
    }


def _repo_health_answer(repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> str:
    open_issues = [item for item in issues if item.state == "open"]
    open_prs = [item for item in prs if item.state == "open"]
    merged_prs = [item for item in prs if item.merged_at]
    failed_runs = [item for item in runs if item.conclusion == "failure"]
    latest_issues = sorted(open_issues, key=_item_time, reverse=True)[:3]
    latest_prs = sorted(prs, key=_item_time, reverse=True)[:3]
    issue_titles = _join_items([f"#{item.number} {item.title}" for item in latest_issues], "没有打开的 Issue")
    pr_titles = _join_items([f"#{item.number} {item.title}" for item in latest_prs], "没有 PR 记录")
    ci_text = f"已同步 {len(failed_runs)} 次失败 CI 运行" if failed_runs else "还没有同步到失败 CI 运行"
    return (
        f"当前仓库：{repo.full_name}\n\n"
        f"- Issues：{len(open_issues)} 个打开。近期关注：{issue_titles}。\n"
        f"- PRs：{len(open_prs)} 个打开，{len(merged_prs)} 个已合并。近期 PR：{pr_titles}。\n"
        f"- CI：{ci_text}。\n\n"
        "建议下一步：先看最新打开的 Issue，或检查最新 PR 风险；如果同步后出现失败 CI，可以让我继续排障。"
    )


def _semantic_search(repo: Repository, message: str, context: dict[str, Any], db: Session) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    filters = {
        "label": context.get("label"),
        "path": context.get("path"),
        "module": context.get("module"),
        "state": context.get("state"),
        "start_date": context.get("start_date"),
        "end_date": context.get("end_date"),
    }
    results = search_similar_documents(
        db,
        repo.id,
        message,
        source_type=context.get("source_type") or _source_type_from_message(message),
        limit=int(context.get("limit", 5)),
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    if not results:
        return "没有找到足够相似的历史记录。可以尝试放宽标签、路径或时间过滤条件。", [], []
    lines = ["找到以下相关历史记录："]
    citations: list[dict[str, str]] = []
    for index, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        number = meta.get("number")
        prefix = f"#{number} " if number else ""
        lines.append(f"- [{index}] {item['source_type']} {prefix}{item['title']}：相似度 {item['score']}")
        if meta.get("path"):
            lines.append(f"  来源路径：{meta['path']}")
        snippet = str(item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  证据：{snippet}")
        citations.append({"type": item["source_type"], "id": item["source_id"], "title": item["title"]})
    return "\n".join(lines), citations, results


async def _semantic_search_async(
    repo_id: uuid.UUID, message: str, context: dict[str, Any]
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    # Workflow tools can run concurrently. Keep GPU work off the event loop and
    # give the read-only retrieval worker its own Session rather than sharing
    # the chat request's write transaction between threads.
    search_context = dict(context)

    def run_search() -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
        with SessionLocal() as search_db:
            return _semantic_search(Repository(id=repo_id), message, search_context, search_db)

    return await asyncio.to_thread(run_search)


def _checkout_root() -> Path:
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = BACKEND_ROOT / root
    return root.resolve()


def _repo_checkout_path(repo: Repository) -> Path:
    local_path = getattr(repo, "local_path", None)
    if local_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    checkout_root = _checkout_root()
    expected = (checkout_root / f"{repo.id}-{safe_name}").resolve()
    if expected.exists():
        return expected

    # 仓库记录可能被删除后重建，而本地 clone 仍保留旧 UUID。
    # 复用同名 checkout，避免工作区工具静默失去源码访问能力。
    candidates = [
        path
        for path in checkout_root.glob(f"*-{safe_name}")
        if path.is_dir() and path.resolve() != expected
    ]
    if candidates:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0].resolve()
    return expected


def _workspace_root(repo: Repository | None) -> Path:
    if repo is None:
        return PROJECT_ROOT
    code_sync = getattr(repo, "code_sync", None) or {}
    if repo.checkout_mode != "local" and code_sync.get("status") in {"pending", "running", "failed"}:
        raise ValueError(f"当前源码尚未就绪：{code_sync.get('message') or code_sync['status']}；不能把旧文件作为最新源码。")
    checkout_path = _repo_checkout_path(repo)
    if not checkout_path.exists():
        raise ValueError(
            f"{repo.full_name} 的仓库代码还没有克隆到本地。"
            "读取源码前，请先同步仓库或重新连接仓库。"
        )
    return checkout_path


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    return path.relative_to(workspace_root).as_posix()


def _workspace_path(workspace_root: Path, raw_path: str | None = None) -> Path:
    normalized_path = str(raw_path or ".").strip()
    if normalized_path in {"", ".", "./", "/", "\\"}:
        normalized_path = "."
    elif not re.match(r"^[a-zA-Z]:", normalized_path):
        normalized_path = normalized_path.lstrip("/\\") or "."
    candidate = (workspace_root / normalized_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("路径位于项目工作区之外。") from exc
    if any(part in WORKSPACE_SKIP_DIRS for part in candidate.relative_to(workspace_root).parts):
        raise ValueError("路径位于被跳过的工作区目录中。")
    if candidate.name in WORKSPACE_SECRET_FILENAMES:
        raise ValueError("不会向对话 Agent 暴露敏感配置文件。")
    return candidate


def _is_workspace_text(path: Path) -> bool:
    if path.name in WORKSPACE_SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in WORKSPACE_TEXT_FILENAMES or path.suffix.lower() in WORKSPACE_TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _iter_workspace_files(root: Path, workspace_root: Path, max_files: int = 1000):
    if root.is_file():
        yield root
        return
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        relative_parts = path.relative_to(workspace_root).parts
        if any(part in WORKSPACE_SKIP_DIRS for part in relative_parts):
            continue
        if not _is_workspace_text(path):
            continue
        count += 1
        yield path


def _read_workspace_text(path: Path, max_bytes: int = 250_000) -> str:
    if not path.exists() or not path.is_file():
        raise ValueError("未找到工作区文件。")
    if not _is_workspace_text(path):
        raise ValueError("只能读取非敏感的文本工作区文件。")
    data = path.read_bytes()
    if len(data) > max_bytes or b"\x00" in data[:4096]:
        raise ValueError("文件过大，或看起来是二进制文件。")
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法将工作区文件解码为文本。")


_PR_SUBJECT = r"(?<![a-z0-9_])(?:pr|pull[\s_-]*request)(?![a-z0-9_])|合并请求"


def _draft_action(message: str, risky_action: str | None) -> str:
    # Choose the draft format after routing. The permission/risk gate remains independent.
    if risky_action is None and re.search(_PR_SUBJECT, message, re.IGNORECASE):
        if re.search(r"(?:起草|草拟|写|留|生成)(?:一条|条)?(?:评审|审查)?评论|评论草稿", message):
            return "comment"
    return risky_action or "write_back"


def _comment_pr(message: str, prs: list[PullRequest], confirmed_pr: PullRequest | None = None) -> PullRequest | None:
    number = _requested_workflow_target_number(message, "pull_request")
    if confirmed_pr is not None:
        if number is not None and confirmed_pr.number != number:
            raise WorkflowTargetResolutionError(f"评论请求的 PR #{number} 与计划确认的 PR #{confirmed_pr.number} 不一致。")
        return confirmed_pr
    if number is not None:
        return _target_by_number(prs, number, target_type="pull_request", number_attribute="number")
    if not prs and re.search(_PR_SUBJECT, message, re.IGNORECASE):
        raise WorkflowTargetResolutionError("当前仓库没有可用于评论的 PR。")
    return max(prs, key=_item_time, default=None)


def _draft_for_action(
    action: str, message: str, repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun],
    *, comment_pr: PullRequest | None = None,
) -> tuple[str, list[dict[str, str]]]:
    citations: list[dict[str, str]] = [{"type": "repository", "id": str(repo.id), "title": repo.full_name}]
    latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    failed_runs = [run for run in runs if run.conclusion == "failure"]
    if action == "create_issue":
        return f"## Issue 草稿\n标题：{message.strip()[:120]}\n\n### 背景\n请补充影响范围、复现步骤和预期行为。\n\n### 验收标准\n- 预期行为清晰\n- 包含验证步骤\n", citations
    if action == "comment":
        comment_pr = comment_pr or _comment_pr(message, prs)
        if comment_pr is not None:
            citations.append({"type": "pull_request", "id": str(comment_pr.id), "title": comment_pr.title})
            return f"## PR 评论草稿\n针对 PR #{comment_pr.number}：{comment_pr.title}\n\n合并前请确认测试覆盖、未解决的 Review 评论，以及是否存在高风险文件。\n", citations
    if action == "comment" and latest_issue:
        citations.append({"type": "issue", "id": str(latest_issue.id), "title": latest_issue.title})
        return f"## Issue 评论草稿\n针对 Issue #{latest_issue.number}：{latest_issue.title}\n\n请补充复现步骤、影响范围、预期行为，以及当前是否有临时绕过方案。\n", citations
    if action == "send_report":
        draft = (
            "## 报告发送草稿\n"
            f"仓库：{repo.full_name}\n\n"
            f"- 打开的 Issue：{len([issue for issue in issues if issue.state == 'open'])}\n"
            f"- 打开的 PR：{len([pr for pr in prs if pr.state == 'open'])}\n"
            f"- 失败 CI 运行：{len(failed_runs)}\n\n"
            "发送前请确认收件人、日期范围和敏感日志内容。\n"
        )
        return draft, citations
    return f"## 写操作草稿\n这可能会写入 GitHub 或外部渠道。MVP 模式只生成动作草稿。\n\n用户请求：{message.strip()}\n", citations


def _safe_confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _skipped_workflow_result(task: WorkflowTask, summary: str) -> WorkflowTaskResult:
    return WorkflowTaskResult(
        task_id=task.id,
        agent_name=task.agent_name,
        task_type=task.task_type,
        status="skipped",
        summary=summary,
        confidence=0.0,
    )


def _agent_context(context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}


class WorkflowTargetResolutionError(ValueError):
    """Raised when an explicitly requested workflow target cannot be resolved safely."""


_WORKFLOW_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "issue": (
        r"(?<![A-Za-z0-9_])issue(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?:工单|议题)\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "pull_request": (
        r"(?<![A-Za-z0-9_])(?:pr|pull[\s_-]*request)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"合并请求\s*(?:#|＃|编号)?\s*(\d+)",
    ),
}

_WORKFLOW_TARGET_LABELS = {
    "issue": "Issue",
    "pull_request": "PR",
    "workflow_run": "CI/Workflow Run",
}


def _single_target_number(numbers: set[int], target_type: str) -> int | None:
    if len(numbers) > 1:
        references = "、".join(f"#{number}" for number in sorted(numbers))
        raise WorkflowTargetResolutionError(
            f"一次工作流只能为一个 {_WORKFLOW_TARGET_LABELS[target_type]} 建立明确 claim；检测到多个目标：{references}。"
        )
    return next(iter(numbers), None)


def _requested_workflow_target_number(message: str, target_type: str) -> int | None:
    if target_type == "workflow_run":
        return _ci_target_numbers(message)[0]
    numbers = {
        int(match.group(1))
        for pattern in _WORKFLOW_TARGET_PATTERNS[target_type]
        for match in re.finditer(pattern, message, flags=re.IGNORECASE)
    }
    return _single_target_number(numbers, target_type)


_CI_REFERENCES = re.compile(
    r"(?P<ci>(?<![a-z0-9_])(?:ci(?:\s*运行)?|workflow[\s_-]*run|run)(?![a-z0-9_])|工作流运行|流水线)"
    r"(?:\s*(?:[#＃]|no\.?|number|id|编号)?\s*(?P<number>\d+))?"
    rf"|(?P<other>{_PR_SUBJECT}|(?<![a-z0-9_])issue(?![a-z0-9_])|工单|议题)"
    r"|[#＃]\s*(?P<short>\d+)"
    r"|(?P<skip>跳过|略过|忽略|排除|除了|除(?=\s*(?:ci|[#＃]|流水线|工作流))|不要|不用|别查|不查|不检查|不看|不包括|不包含|"
    r"\b(?:skip|except|exclude|excluding|ignore|without|not|don['’]t)\b)"
    r"|(?P<reset>[，,](?!\s*[#＃])|[。；;.!?！？\n]|而是|然后|改为|(?:改|只|再)(?:查|看)|\b(?:instead|but)\b)",
    re.IGNORECASE,
)


def _ci_target_numbers(message: str) -> tuple[int | None, set[int]]:
    included: set[int] = set()
    excluded: set[int] = set()
    tokens = list(_CI_REFERENCES.finditer(message))
    # "跳过 #11，查最新 CI" also has a unique subject if no other object type is mentioned.
    in_ci = any(token["ci"] for token in tokens) and not any(token["other"] for token in tokens)
    skipping = False
    for token in tokens:
        if token["skip"] or token["reset"]:
            skipping = bool(token["skip"])
        elif token["other"]:
            in_ci = False
        else:
            # A bare "#11" inherits the preceding object type, including across "，跳过".
            if token["ci"]:
                in_ci = True
            number = token["number"] or token["short"]
            if in_ci and number:
                if skipping or re.match(r"\s*除外", message[token.end():]):
                    excluded.add(int(number))
                else:
                    included.add(int(number))
    if included & excluded:
        raise WorkflowTargetResolutionError("同一个 CI 编号同时被指定和排除，请明确要检查的运行。")
    return _single_target_number(included, "workflow_run"), excluded


def _target_by_number(
    items: list[Any],
    number: int | None,
    *,
    target_type: str,
    number_attribute: str,
) -> Any | None:
    if number is None:
        return None
    target = next((item for item in items if getattr(item, number_attribute, None) == number), None)
    if target is None:
        raise WorkflowTargetResolutionError(
            f"当前仓库没有同步 {_WORKFLOW_TARGET_LABELS[target_type]} #{number}，已停止生成计划；不会回退到最新对象。"
        )
    return target


def _build_engineering_workflow_runtime(
    repo: Repository,
    db: Session,
    message: str,
    context: dict[str, Any],
    *,
    excluded_task_types: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    issues, prs, runs = _repo_payload(repo.id, db)
    action = _risky_action(message)
    draft_action = _draft_action(message, action)
    failed_runs = [run for run in runs if run.conclusion == "failure"]
    requested_issue_number = None if "issue_analysis" in excluded_task_types else _requested_workflow_target_number(message, "issue")
    requested_pr_number = None if "pr_review" in excluded_task_types else _requested_workflow_target_number(message, "pull_request")
    requested_run_number, excluded_runs = (None, set()) if "ci_debug" in excluded_task_types else _ci_target_numbers(message)
    selected_issue = _target_by_number(
        issues,
        requested_issue_number,
        target_type="issue",
        number_attribute="number",
    )
    selected_pr = _target_by_number(
        prs,
        requested_pr_number,
        target_type="pull_request",
        number_attribute="number",
    )
    selected_run = _target_by_number(
        runs,
        requested_run_number,
        target_type="workflow_run",
        number_attribute="github_run_id",
    )
    if selected_issue is None:
        selected_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    if selected_pr is None:
        selected_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    if selected_run is None:
        selected_run = max((run for run in failed_runs if run.github_run_id not in excluded_runs), key=_item_time, default=None)

    def claimed_target(task: WorkflowTask, target_type: str, items: list[Any], fallback: Any | None) -> Any | None:
        if task.claim.target_type != target_type:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 类型为 {task.claim.target_type!r}，预期为 {target_type!r}。"
            )
        if not task.claim.target_id:
            return fallback
        try:
            target_id = uuid.UUID(str(task.claim.target_id))
        except (TypeError, ValueError) as exc:
            raise WorkflowTargetResolutionError(f"任务 {task.id} 的 claim.target_id 不是有效 UUID。") from exc
        target = next((item for item in items if item.id == target_id), None)
        if target is None:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 目标不存在或不属于仓库 {repo.full_name}；不会回退到最新对象。"
            )
        if target_type == "workflow_run" and target.github_run_id in excluded_runs:
            raise WorkflowTargetResolutionError(f"计划中的 CI #{target.github_run_id} 已被排除，请重新生成计划。")
        return target

    safe_context = _agent_context(context)
    skill_registry = context.get("_skill_registry")
    agent_skill_cache: dict[str, dict[str, Any]] = {}

    def agent_skill_context(entrypoint: str) -> dict[str, Any]:
        if entrypoint in agent_skill_cache:
            return agent_skill_cache[entrypoint]
        if not isinstance(skill_registry, SkillRegistry):
            agent_skill_cache[entrypoint] = {}
            return {}
        activation = skill_registry.activate_entrypoint(entrypoint)
        if activation is None:
            agent_skill_cache[entrypoint] = {}
            return {}
        payload = activation.model_dump(mode="json")
        runtime = {
            "active_skills": [payload],
            "skill_instructions": activation.prompt_block(),
            "skill_runtime": {
                **activation.trace_metadata(),
                "entrypoint": activation.entrypoint,
                "output_contract": activation.output_contract,
                "safety_level": activation.safety_level,
                "instructions_loaded": True,
            },
        }
        agent_skill_cache[entrypoint] = runtime
        return runtime

    def with_skill_runtime(result: dict[str, Any], entrypoint: str) -> dict[str, Any]:
        runtime = agent_skill_context(entrypoint).get("skill_runtime")
        return {**result, "_skill_runtime": {**runtime, "validation": "entrypoint_completed"}} if runtime else result

    workflow_context = {
        **safe_context,
        "repo_id": str(repo.id),
        "repo_full_name": repo.full_name,
        "has_repo": True,
        "has_issue": selected_issue is not None,
        "issue_id": str(selected_issue.id) if selected_issue else None,
        "issue_number": selected_issue.number if selected_issue else None,
        "issue_explicit": requested_issue_number is not None,
        "has_pr": selected_pr is not None,
        "pr_id": str(selected_pr.id) if selected_pr else None,
        "pr_number": selected_pr.number if selected_pr else None,
        "pr_explicit": requested_pr_number is not None,
        "has_ci_run": selected_run is not None,
        "has_failed_ci": selected_run is not None and selected_run.conclusion == "failure",
        "workflow_run_id": str(selected_run.id) if selected_run else None,
        "workflow_run_number": selected_run.github_run_id if selected_run else None,
        "workflow_run_explicit": requested_run_number is not None,
        "requires_safety": bool(action),
        "comment_pr_id": str(selected_pr.id) if selected_pr and draft_action == "comment" else None,
        "team_members": safe_context.get("team_members") or [],
    }

    async def run_repo_health(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        answer = _repo_health_answer(repo, issues, prs, runs)
        open_issue_count = len([item for item in issues if item.state == "open"])
        open_pr_count = len([item for item in prs if item.state == "open"])
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"{repo.full_name}: {open_issue_count} open issues, {open_pr_count} open PRs, {len(failed_runs)} failed CI runs.",
            output={
                "answer": answer,
                "open_issues": open_issue_count,
                "open_prs": open_pr_count,
                "failed_ci": len(failed_runs),
            },
            evidence_count=len(issues) + len(prs) + len(runs),
            confidence=0.82,
            citations=[{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        )

    async def run_issue_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        issue = claimed_target(task, "issue", issues, selected_issue)
        if issue is None:
            return _skipped_workflow_result(task, "No synced issue is available.")
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            {**safe_context, **agent_skill_context("issue_agent.analyze_issue")},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Issue #{issue.number}: {result.get('conclusion')} / {result.get('priority')} / owner {result.get('suggested_owner')}.",
            output=with_skill_runtime(result, "issue_agent.analyze_issue"),
            evidence_count=len(result.get("evidence") or []),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "issue", "id": str(issue.id), "title": issue.title}],
        )

    async def run_pr_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        pull_request = claimed_target(task, "pull_request", prs, selected_pr)
        if pull_request is None:
            return _skipped_workflow_result(task, "No synced pull request is available.")
        ci_task = next((item for item in spec.tasks if item.task_type == "ci_debug"), None)
        claimed_ci_run = claimed_target(ci_task, "workflow_run", runs, selected_run) if ci_task else selected_run
        pr_context = {
            **safe_context,
            **agent_skill_context("pr_review_agent.analyze_pr"),
            "repo_id": str(repo.id),
            "ci_summary": {
                "failed_runs": len(failed_runs),
                "latest_failed_run": claimed_ci_run.name if claimed_ci_run and claimed_ci_run.conclusion == "failure" else None,
            },
        }
        result = await PRReviewAgent().run(
            {
                "number": pull_request.number,
                "title": pull_request.title,
                "body": pull_request.body,
                "state": pull_request.state,
                "author": pull_request.author,
                "merged_at": pull_request.merged_at.isoformat() if pull_request.merged_at else None,
                "files": [
                    {
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "patch": file.patch,
                    }
                    for file in pull_request.files
                ],
                "comments": [
                    {
                        "body": comment.body,
                        "path": comment.path,
                        "line": comment.line,
                        "author": comment.author,
                    }
                    for comment in pull_request.review_comments
                ],
            },
            pr_context,
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"PR #{pull_request.number}: {result.get('merge_recommendation')} with {len(result.get('risk_points') or [])} risk points.",
            output=with_skill_runtime(result, "pr_review_agent.analyze_pr"),
            evidence_count=len(pull_request.files) + len(pull_request.review_comments),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "pull_request", "id": str(pull_request.id), "title": pull_request.title}],
        )

    async def run_ci_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        workflow_run = claimed_target(task, "workflow_run", runs, selected_run)
        if workflow_run is None:
            return _skipped_workflow_result(task, "No synced CI run is available.")
        result = await CIDebugAgent().run(
            {
                "name": workflow_run.name,
                "status": workflow_run.status,
                "conclusion": workflow_run.conclusion,
                "jobs": workflow_run.jobs or [],
                "logs_text": workflow_run.logs_text,
            },
            {**safe_context, **agent_skill_context("ci_debug_agent.analyze_workflow_run"), "repo_id": str(repo.id)},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"CI {workflow_run.name}: {result.get('failure_type')}, merge blocking={result.get('is_merge_blocking')}.",
            output=with_skill_runtime(result, "ci_debug_agent.analyze_workflow_run"),
            evidence_count=len(workflow_run.jobs or []) + (1 if workflow_run.logs_text else 0),
            confidence=_safe_confidence(result.get("confidence"), 0.58),
            citations=[{"type": "workflow_run", "id": str(workflow_run.id), "title": workflow_run.name}],
        )

    async def run_rag_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        query = message or spec.goal
        answer, citations, results = await _semantic_search_async(repo.id, query, safe_context)
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=uuid.UUID(str(context["conversation_id"])) if context.get("conversation_id") else None,
            tool_name="workflow.rag_search",
            query=query,
            results=results,
            citations=citations,
            scope="project",
            mode="hybrid",
            metadata={"route": "workflow_rag_task", "task_id": task.id},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Retrieved {len(citations)} related historical records.",
            output={"answer": answer, "citations": citations},
            evidence_count=len(citations),
            confidence=0.68 if citations else 0.45,
            citations=citations,
        )

    async def run_safety_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        comment_pr = None
        if draft_action == "comment":
            # Older previews used a repository claim for safety; reuse their reviewed PR if present.
            pr_task = task if task.claim.target_type == "pull_request" else next(
                (item for item in spec.tasks if item.task_type == "pr_review"), None
            )
            if pr_task and pr_task.claim.target_id:
                confirmed_pr = claimed_target(pr_task, "pull_request", prs, selected_pr)
                comment_pr = _comment_pr(message or spec.goal, prs, confirmed_pr)
            elif re.search(_PR_SUBJECT, message, re.IGNORECASE) and _requested_workflow_target_number(message, "pull_request") is None:
                raise WorkflowTargetResolutionError("计划没有记录评论的目标 PR，请重新生成计划。")
        draft, citations = _draft_for_action(draft_action, message or spec.goal, repo, issues, prs, runs, comment_pr=comment_pr)
        result = await SafetyAgent().run(
            {"action": draft_action, "draft": draft},
            {**safe_context, **agent_skill_context("safety_agent.classify_action_risk")},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Safety risk={result.get('risk_level')}, confirmation required={result.get('requires_confirmation')}.",
            output=with_skill_runtime(result, "safety_agent.classify_action_risk"),
            evidence_count=1,
            confidence=0.9,
            citations=citations,
        )

    return workflow_context, {
        "repo_health": run_repo_health,
        "issue_analysis": run_issue_task,
        "pr_review": run_pr_task,
        "ci_debug": run_ci_task,
        "rag_search": run_rag_task,
        "safety_review": run_safety_task,
    }


def _persist_engineering_workflow(
    db: Session,
    repo: Repository | None,
    conversation: Conversation | None,
    report: dict[str, Any],
) -> uuid.UUID:
    spec = report.get("spec") or {}
    task_by_id = {
        str(task.get("id")): task
        for task in spec.get("tasks", [])
        if isinstance(task, dict) and task.get("id")
    }
    workflow = AgentWorkflowRun(
        repo_id=repo.id if repo else None,
        conversation_id=conversation.id if conversation else None,
        goal=str(spec.get("goal") or ""),
        spec_json=spec,
        observation_json=report.get("observation") or {},
        final_answer=str(report.get("final_answer") or ""),
        metrics=report.get("metrics") or {},
        status="error" if int((report.get("metrics") or {}).get("error_count") or 0) else "success",
        completed_at=datetime.now(timezone.utc),
    )
    db.add(workflow)
    db.flush()
    for result in report.get("task_results") or []:
        if not isinstance(result, dict):
            continue
        task_id = str(result.get("task_id") or "")
        task = task_by_id.get(task_id, {})
        db.add(
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id=task_id,
                agent_name=str(result.get("agent_name") or ""),
                task_type=str(result.get("task_type") or ""),
                claim_json=task.get("claim") if isinstance(task.get("claim"), dict) else {},
                input_json={"objective": task.get("objective"), "dependencies": task.get("dependencies", [])},
                result_json=result,
                status=str(result.get("status") or "unknown"),
                error=result.get("error"),
            )
        )
    return workflow.id


def _pr_diff_evidence(pr: PullRequest) -> str:
    def ranges(numbers: list[int]) -> str:
        groups: list[str] = []
        start = end = None
        for number in sorted(set(numbers)):
            if end is not None and number == end + 1:
                end = number
                continue
            if start is not None:
                groups.append(str(start) if start == end else f"{start}-{end}")
            start = end = number
        if start is not None:
            groups.append(str(start) if start == end else f"{start}-{end}")
        return ", ".join(groups) or "无"

    sections = []
    for file in pr.files:
        patch = file.patch or ""
        added, removed, headers = [], [], []
        old_line = new_line = None
        for line in patch.splitlines():
            hunk = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk:
                old_line, new_line = int(hunk[1]), int(hunk[3])
                headers.append(hunk[0])
            elif old_line is not None and new_line is not None:
                if line.startswith("+"):
                    added.append(new_line)
                    new_line += 1
                elif line.startswith("-"):
                    removed.append(old_line)
                    old_line += 1
                elif line.startswith(" "):
                    old_line += 1
                    new_line += 1
        scope = (f"新增/替换的新版本行：{ranges(added)}；删除/替换的旧版本行：{ranges(removed)}\n"
                 f"diff 段（包含上下文）：{'；'.join(headers)}") if headers else "未提供可解析的 diff 行段；不能将整个文件视为本次新增。"
        sections.append(f"文件：{file.filename}；状态：{file.status}\n{scope}\n"
                        + (f"```diff\n{patch}\n```" if patch else "patch 未提供（可能为二进制、过大或同步缺失）。"))
    return "\n\n".join(sections)


def _build_conversation_tools(repo: Repository | None, conversation: Conversation | None, db: Session, message: str = "") -> dict[str, Any]:
    mcp_client = StdioMcpClient()
    bound_pr: PullRequest | None = None
    snapshots: dict[uuid.UUID, dict[str, Any]] = {}
    pr_reviews: dict[uuid.UUID, dict[str, Any]] = {}
    snapshot_lock = asyncio.Lock()
    explicit_numbers = {int(n) for n in re.findall(r"(?<![A-Za-z_])(?:pr|pull\s*request|合并请求)\s*[#＃]?\s*(\d+)", message, re.I)}
    is_pr_request = bool(explicit_numbers or re.search(r"(?<![A-Za-z_])pr(?![A-Za-z_])|pull\s*request|合并请求", message, re.I))

    def select_pr(input_data: dict[str, Any]) -> PullRequest:
        nonlocal bound_pr
        if repo is None:
            raise ValueError("当前没有可用仓库")
        number = input_data.get("pr_number")
        requested_id = input_data.get("pr_id")
        if number is not None:
            number = int(number)
            if explicit_numbers and number not in explicit_numbers:
                raise ValueError("请求的 PR 与用户指定的编号不一致")
        elif len(explicit_numbers) == 1:
            number = next(iter(explicit_numbers))
        elif len(explicit_numbers) > 1 and not requested_id:
            raise ValueError("涉及多个 PR，请在工具参数 pr_number 中明确本次读取的 PR")
        if number is None and not requested_id and bound_pr is not None:
            return bound_pr
        query = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.repo_id == repo.id)
        if number is not None:
            query = query.filter(PullRequest.number == number)
        if requested_id:
            query = query.filter(PullRequest.id == uuid.UUID(str(requested_id)))
        pr = query.order_by(PullRequest.created_at.desc(), PullRequest.number.desc()).first()
        if pr is None or (explicit_numbers and pr.number not in explicit_numbers):
            raise ValueError("当前仓库未同步指定 PR，请先同步；不会回退到其他 PR")
        bound_pr = pr
        return pr

    async def pr_snapshot(pr: PullRequest) -> dict[str, Any]:
        async with snapshot_lock:
            if pr.id not in snapshots:
                snapshots[pr.id] = await asyncio.to_thread(prepare_pr_worktree_by_id, repo.id, pr.id)
            return snapshots[pr.id]

    async def workspace_for(input_data: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        if bound_pr is not None or is_pr_request or input_data.get("pr_number") or input_data.get("pr_id"):
            pr = select_pr(input_data)
            snapshot = await pr_snapshot(pr)
            if snapshot.get("status") != "ready":
                raise ValueError(snapshot.get("message") or "PR 源码未准备好；不会用主分支代替")
            return Path(snapshot["snapshot_path"]).resolve(), {"pr_number": pr.number, "commit_sha": snapshot["commit_sha"]}
        return _workspace_root(repo), {}

    def source_label(version: dict[str, Any]) -> str:
        return f"PR #{version['pr_number']} @ {version['commit_sha']}\n" if version else ""

    def agent_context(context: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}

    async def call_memory_mcp(tool_name: str, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        arguments = dict(input_data)
        if repo:
            arguments["repo_id"] = str(repo.id)
        if conversation:
            arguments["conversation_id"] = str(conversation.id)
        if tool_name == "devflow_search_evidence":
            arguments.setdefault("query", input_data.get("message") or "")
        result = await mcp_client.call_tool(tool_name, arguments)
        text_parts = [
            str(item.get("text") or "")
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return {
            "route": tool_name.replace("devflow_", ""),
            "tool_calls": [{"tool_name": tool_name, "arguments": arguments, "status": "success"}],
            "answer": "\n".join(text_parts).strip(),
            "citations": [],
        }

    async def search_evidence_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_search_evidence", input_data, context)

    async def get_thread_context_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_get_thread_context", input_data, context)

    async def read_session_events_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_read_session_events", input_data, context)

    async def list_workspace_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        workspace_root, version = await workspace_for(input_data)
        root = _workspace_path(workspace_root, str(input_data.get("path") or "."))
        limit = min(max(int(input_data.get("limit") or 80), 1), 200)
        raw_pattern = str(input_data.get("pattern") or "").strip().lower()
        rows: list[str] = []
        for path in _iter_workspace_files(root, workspace_root):
            relative = _workspace_relative(path, workspace_root)
            if raw_pattern and raw_pattern not in relative.lower():
                continue
            rows.append(relative)
            if len(rows) >= limit:
                break
        answer = "\n".join(rows) if rows else "没有找到匹配的工作区文件。"
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": {"path": str(input_data.get("path") or "."), "pattern": raw_pattern, "limit": limit, **version}, "status": "success"}],
            "answer": source_label(version) + answer,
            "citations": [{"type": "workspace", "id": str(workspace_root), "title": repo.full_name if repo else "项目工作区"}],
        }

    async def read_workspace_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(input_data.get("path") or "").strip()
        if not raw_path:
            raise ValueError("path 为必填项")
        workspace_root, version = await workspace_for(input_data)
        path = _workspace_path(workspace_root, raw_path)
        if not path.is_file():
            raise ValueError(f"{source_label(version).strip() or '当前源码'} 中不存在文件 {raw_path}；请核对实际文件列表。PR 删除的文件请依据 patch 审查。")
        text = _read_workspace_text(path)
        lines = text.splitlines()
        start_line = max(int(input_data.get("start_line") or 1), 1)
        line_count = min(max(int(input_data.get("line_count") or 160), 1), 400)
        end_line = min(start_line + line_count - 1, len(lines))
        numbered = [f"{line_number}: {lines[line_number - 1]}" for line_number in range(start_line, end_line + 1)]
        relative = _workspace_relative(path, workspace_root)
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": {"path": relative, "start_line": start_line, "line_count": line_count, **version}, "status": "success"}],
            "answer": source_label(version) + f"{relative}:{start_line}-{end_line}\n" + "\n".join(numbered),
            "citations": [{"type": "workspace_file", "id": relative, "title": relative, **version}],
        }

    async def search_workspace_code(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        pattern = str(input_data.get("pattern") or "").strip() or None
        query = pattern or str(input_data.get("query") or input_data.get("message") or "").strip()
        if not pattern and not query_terms(query):
            raise ValueError("query 为必填项")
        workspace_root, version = await workspace_for(input_data)
        relative_path = str(input_data.get("path") or ".")
        _workspace_path(workspace_root, relative_path)
        limit = min(max(int(input_data.get("limit") or 12), 1), 50)
        matches = search_checkout_code(workspace_root, query, limit=limit, relative_path=relative_path, pattern=pattern)
        rows = [item["snippet"] for item in matches]
        citations = [
            {"type": "workspace_file", "id": item["source_id"], "title": item["title"]}
            for item in matches
        ]
        return {
            "route": "workspace_search_code",
            "tool_calls": [{"tool_name": "workspace.search_code", "arguments": {"query": query, "pattern": pattern, "path": relative_path, "limit": limit, **version}, "status": "success"}],
            "answer": source_label(version) + ("\n".join(rows) if rows else "没有找到匹配的工作区代码。"),
            "citations": citations,
        }

    async def repo_health(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return {"route": "unknown", "tool_calls": [], "answer": "当前还没有可用仓库，请先连接仓库。", "citations": []}
        issues, prs, runs = _repo_payload(repo.id, db)
        return {
            "route": "repo_health",
            "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": _repo_health_answer(repo, issues, prs, runs),
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    async def semantic_search(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)
        original_message = str(input_data.get("message") or "").strip()
        query = str(input_data.get("query") or original_message).strip()
        if not query:
            raise ValueError("query 为必填项")
        answer, citations, results = await _semantic_search_async(repo.id, query, context)
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=conversation.id if conversation else None,
            tool_name="rag.search_similar_documents",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if conversation else "project",
            mode="hybrid",
            metadata={
                "source_type": context.get("source_type"),
                "limit": context.get("limit", 5),
                "original_message": original_message,
                "query_rewritten": bool(original_message and query != original_message),
            },
        )
        return {
            "route": "semantic_search",
            "tool_calls": [
                {
                    "tool_name": "rag.search_similar_documents",
                    "arguments": {"repo_id": str(repo.id), "query": query},
                    "status": "success",
                }
            ],
            "answer": answer,
            "citations": citations,
            "retrieved_contexts": [
                str(item.get("snippet") or "")
                for item in results
                if str(item.get("snippet") or "").strip()
            ],
            "retrieval_results": [
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                    "snippet": str(item.get("snippet") or ""),
                    "path": (item.get("metadata") or {}).get("path"),
                    "retrieval": item.get("retrieval") or {},
                }
                for item in results
            ],
        }

    async def analyze_issue(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)
        issue = db.query(Issue).filter(Issue.repo_id == repo.id).order_by(Issue.created_at.desc()).first()
        if not issue:
            return {"route": "issue_analysis", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 Issue。", "citations": []}
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            agent_context(context),
        )
        return {
            "route": "issue_analysis",
            "tool_calls": [{"tool_name": "issue_agent.analyze_issue", "arguments": {"issue_id": str(issue.id)}, "status": "success"}],
            "answer": (
                f"Issue #{issue.number} 的结论：{result['conclusion']}。\n\n"
                f"- 分类：{result['category']}\n"
                f"- 优先级：{result['priority']}\n"
                f"- 复杂度：{result['complexity']}\n"
                f"- 建议负责人：{result['suggested_owner']}，{result['owner_reason']}\n\n"
                f"理由：{result['conclusion_reason']}\n\n"
                f"下一步：{_join_items(result.get('checklist') or result.get('action_items') or [], '补充复现步骤、影响范围和验收标准。')}"
            ),
            "citations": [{"type": "issue", "id": str(issue.id), "title": issue.title}],
        }

    async def analyze_pr(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)
        pr = select_pr(input_data)
        snapshot = await pr_snapshot(pr)
        code_references = []
        source_warnings = []
        if snapshot.get("status") == "ready":
            root = Path(snapshot["snapshot_path"]).resolve()
            remaining = 14000
            for file in pr.files:
                if file.status == "removed" or len(code_references) >= 6 or remaining <= 0:
                    continue
                try:
                    text = _read_workspace_text(_workspace_path(root, file.filename))
                    excerpt = "\n".join(f"{i}: {line}" for i, line in enumerate(text.splitlines()[:160], 1))[:remaining]
                    code_references.append({"title": file.filename, "snippet": excerpt, "commit_sha": snapshot["commit_sha"]})
                    remaining -= len(excerpt)
                except ValueError as exc:
                    source_warnings.append(f"{file.filename}: {exc}")
        else:
            source_warnings.append(snapshot.get("message") or "PR 源码未就绪，仅依据已同步 patch 和评论分析")
        # The snapshot is pinned for this request. Reusing its review avoids
        # repeating an expensive model call when the controller rereads a result.
        if pr.id not in pr_reviews:
            pr_reviews[pr.id] = await PRReviewAgent().run(
                {
                    "number": pr.number,
                    "title": pr.title,
                    "body": pr.body,
                    "files": [{"filename": file.filename, "status": file.status, "patch": file.patch} for file in pr.files],
                    "comments": [{"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author} for comment in pr.review_comments],
                },
                agent_context(context) | {"repo_id": str(repo.id), "pr_snapshot": {k: v for k, v in snapshot.items() if k != "snapshot_path"},
                                          "code_references": code_references, "source_warnings": source_warnings},
            )
        result = pr_reviews[pr.id]
        return {
            "route": "pr_review",
            "tool_calls": [{"tool_name": "pr_review_agent.analyze_pr", "arguments": {"pr_id": str(pr.id), "pr_number": pr.number, "commit_sha": snapshot.get("commit_sha"), "source_status": snapshot.get("status")}, "status": "success"}],
            "answer": (
                f"PR #{pr.number} 元数据（已同步 GitHub 记录）：\n标题：{pr.title}\n"
                f"状态：{pr.state}；已合并：{pr.merged_at.isoformat() if pr.merged_at else '否'}\n"
                f"PR head SHA：{pr.head_sha or '未知'}\n源码版本：{snapshot.get('commit_sha') or '未就绪'}\n"
                f"PR 说明：\n{pr.body or '未提供'}\n\n摘要：{result['summary']}"
                + "\n已读取源码：" + (", ".join(item["title"] for item in code_references) or "无")
                + f"\n变更文件共 {len(pr.files)} 个（展示前 12 个）：" + ", ".join(file.filename for file in pr.files[:12])
                + f"\n主要风险：{_join_items(result['risk_points'], '没有明显风险。')}\n测试建议：{_join_items(result['test_suggestions'], '运行相关测试和核心流程检查。')}"
                + ("\n源码限制：" + "；".join(source_warnings) if source_warnings else "")
                + "\n\n实际变更范围与 patch（审查应围绕这些变更及其相关依赖）：\n" + _pr_diff_evidence(pr)
            ),
            "citations": [{"type": "pull_request", "id": str(pr.id), "title": pr.title}],
        }

    async def analyze_ci(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)
        if bound_pr is not None or is_pr_request or input_data.get("pr_number") or input_data.get("pr_id"):
            pr = select_pr(input_data)
            sha = pr.head_sha
            scope = f"PR #{pr.number}（{pr.title}）@ {sha or '未知提交'}"
            arguments = {"pr_number": pr.number, "pr_id": str(pr.id), "commit_sha": sha}
            citations = [{"type": "pull_request", "id": str(pr.id), "title": pr.title}]
            answer = f"{scope}\nCI 状态未知：PR 缺少 head SHA，请先同步该 PR。"
            if sha:
                token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
                try:
                    evidence = await list_pr_workflow_runs(repo.owner, repo.name, pr.number, sha, token=token, base_url=repo.api_base_url)
                    runs = evidence["runs"]
                    arguments["github_run_ids"] = [run["id"] for run in runs]
                    rows = [f"{scope}\nGitHub Actions：按上述提交 SHA 查询；关联其他 PR 的运行已排除。"]
                    for run in runs:
                        rows.append(f"- {run.get('name')} | run_id={run['id']} | event={run.get('event')} | "
                                    f"head_sha={run.get('head_sha')} | 关联依据={run.get('matched_by')} | "
                                    f"status={run.get('status')} | conclusion={run.get('conclusion') or '未完成'} | {run.get('html_url') or ''}")
                        citations.append({"type": "workflow_run", "id": str(run['id']), "title": str(run.get('name') or 'CI')})
                    if not runs:
                        rows.append("CI 状态未知：没有返回该 PR 提交的 Actions 运行；不能用仓库其他提交的成功或失败替代。")
                    elif not evidence["complete"]:
                        rows.append("查询达到 GitHub 分页上限，运行清单不完整，不能据此判定全部通过。")
                    elif all(run.get("status") == "completed" and run.get("conclusion") == "success" for run in runs):
                        rows.append("上述提交的已返回 Actions 工作流均已通过。此结果不代表已核实所有分支保护要求或人工审批。")
                    else:
                        rows.append("上述提交仍有未完成或非 success 的 Actions 运行，请按各运行的实际状态判断。")
                    failures = [run for run in runs if run.get("conclusion") == "failure"]
                    if failures:
                        failed = failures[0]
                        try:
                            jobs = await list_workflow_run_jobs(repo.owner, repo.name, failed['id'], token=token, base_url=repo.api_base_url)
                            logs = await get_workflow_run_logs(repo.owner, repo.name, failed['id'], token=token, base_url=repo.api_base_url)
                            result = await CIDebugAgent().run({"name": failed.get("name"), "status": failed.get("status"),
                                                              "conclusion": failed.get("conclusion"), "jobs": jobs, "logs_text": logs},
                                                             agent_context(context) | {"repo_id": str(repo.id)})
                            rows.append(f"run {failed['id']} 的失败诊断：{result['failure_summary']}\n排查步骤：{_join_items(result['debug_steps'], '无')}")
                        except httpx.HTTPError:
                            rows.append(f"run {failed['id']} 的失败日志读取失败；失败结论来自运行状态，根因尚未核实。")
                    answer = "\n".join(rows)
                except httpx.HTTPError as exc:
                    status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
                    answer = f"{scope}\nCI 状态未知：GitHub Actions 查询失败（{status}）。不会用仓库其他提交的失败记录替代。"
            return {"route": "ci_debug", "answer": answer, "citations": citations,
                    "tool_calls": [{"tool_name": "ci_debug_agent.analyze_workflow_run", "arguments": arguments, "status": "success"}]}
        run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.conclusion == "failure").order_by(WorkflowRun.created_at.desc()).first()
        if not run:
            issues, prs, runs = _repo_payload(repo.id, db)
            return {
                "route": "repo_health",
                "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id), "reason": "no_failed_ci"}, "status": "success"}],
                "answer": _repo_health_answer(repo, issues, prs, runs) + "\n\n说明：这个仓库目前还没有同步到失败 CI 运行。",
                "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
            }
        result = await CIDebugAgent().run({"name": run.name, "status": run.status, "conclusion": run.conclusion, "jobs": run.jobs or [], "logs_text": run.logs_text}, agent_context(context) | {"repo_id": str(repo.id)})
        return {
            "route": "ci_debug",
            "tool_calls": [{"tool_name": "ci_debug_agent.analyze_workflow_run", "arguments": {"run_id": str(run.id)}, "status": "success"}],
            "answer": f"最新失败 CI：{run.name}。失败类型：{result['failure_type']}。\n\n{result['failure_summary']}\n\n排查步骤：{_join_items(result['debug_steps'], '没有明确可用的排查步骤。')}",
            "citations": [{"type": "workflow_run", "id": str(run.id), "title": run.name}],
        }

    async def generate_report(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        start_at, end_at = _date_window_from_context(context)
        scoped_issues = [item for item in issues if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        scoped_prs = [item for item in prs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        scoped_runs = [item for item in runs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at])]
        result = await ReportAgent().run(
            {
                "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in scoped_issues],
                "pull_requests": [{"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at} for item in scoped_prs],
                "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in scoped_runs],
                "metrics": {
                    "issues": len(scoped_issues),
                    "pull_requests": len(scoped_prs),
                    "merged_prs": len([item for item in scoped_prs if item.merged_at]),
                    "open_prs": len([item for item in scoped_prs if item.state == "open"]),
                    "failed_ci": len([item for item in scoped_runs if item.conclusion == "failure"]),
                },
            },
            {"repo_id": str(repo.id), "repo_name": repo.full_name, "start_date": start_at.date().isoformat(), "end_date": end_at.date().isoformat()},
        )
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
            result["report_markdown"],
            {
                "display_title": f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
                "start_date": start_at.date().isoformat(),
                "end_date": end_at.date().isoformat(),
                "repo_name": repo.full_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "chat",
            },
            commit=False,
        )
        return {
            "route": "weekly_report",
            "tool_calls": [{"tool_name": "report_agent.generate_weekly_report", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": result["report_markdown"],
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    async def check_safety(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        action = _draft_action(input_data["message"], _risky_action(input_data["message"]))
        comment_pr = _comment_pr(input_data["message"], prs) if action == "comment" else None
        draft, citations = _draft_for_action(action, input_data["message"], repo, issues, prs, runs, comment_pr=comment_pr)
        safety = await SafetyAgent().run({"action": action, "draft": draft}, agent_context(context))
        latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
        target_type = "repository"
        target_id = None
        metadata: dict[str, Any] = {"source": "chat", "message": input_data["message"]}
        if comment_pr is not None:
            target_type = "pull_request"
            target_id = comment_pr.id
            metadata["pr_number"] = comment_pr.number
        elif action in {"comment", "create_issue", "close_issue"} and latest_issue:
            target_type = "issue"
            target_id = latest_issue.id
            metadata["issue_number"] = latest_issue.number
        draft_type = {
            "comment": "pr_comment" if target_type == "pull_request" else "issue_comment",
            "create_issue": "create_issue",
            "send_report": "send_report",
            "write_back": "comment",
        }.get(action, action)
        user = ensure_demo_user(db)
        action_draft = ActionDraft(
            repo_id=repo.id,
            draft_type=draft_type,
            target_type=target_type,
            target_id=target_id,
            title=f"{draft_type} 草稿",
            content=str(safety.get("draft") or draft),
            risk_level=str(safety.get("risk_level") or "medium"),
            created_by=user.id,
            meta=metadata,
        )
        db.add(action_draft)
        db.flush()
        write_audit_log(
            db,
            user=user,
            repo_id=repo.id,
            action="action_draft.create_from_chat",
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            request_json={"action": action, "draft_type": draft_type},
            result_json={"draft_id": str(action_draft.id)},
        )
        return {
            "route": "safety_draft",
            "tool_calls": [{"tool_name": "safety_agent.classify_action_risk", "arguments": {"action": action, "draft_id": str(action_draft.id)}, "status": "success"}],
            "answer": f"这个请求可能写入 GitHub 或发送到外部渠道，已生成待确认安全草稿。\n\n草稿 ID：{action_draft.id}\n\n{str(safety.get('draft') or draft)}",
            "citations": citations,
        }

    async def run_engineering_workflow(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if repo is None:
            return await repo_health(input_data, context)

        message = str(input_data.get("message") or input_data.get("query") or "").strip()
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, message, context)
        report = await WorkflowOrchestrator(task_runners).run({"message": message or "Run an engineering workflow."}, workflow_context)
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        return _engineering_workflow_response(repo, workflow_run_id, report)

    return {
        "workspace.list_files": list_workspace_files,
        "workspace.read_file": read_workspace_file,
        "workspace.search_code": search_workspace_code,
        "repo_health.generate_summary": repo_health,
        "rag.search_similar_documents": semantic_search,
        "issue_agent.analyze_issue": analyze_issue,
        "pr_review_agent.analyze_pr": analyze_pr,
        "ci_debug_agent.analyze_workflow_run": analyze_ci,
        "report_agent.generate_weekly_report": generate_report,
        "safety_agent.classify_action_risk": check_safety,
        "workflow.run_engineering_review": run_engineering_workflow,
        "devflow_search_evidence": search_evidence_mcp,
        "devflow_get_thread_context": get_thread_context_mcp,
        "devflow_read_session_events": read_session_events_mcp,
    }


def _skill_runtime_context() -> dict[str, Any]:
    registry = get_skill_registry()
    return {
        "_skill_registry": registry,
        "registered_skills": registry.prompt_context(),
        "skill_trace_by_tool": registry.trace_metadata_by_tool(),
        "skill_description_by_tool": registry.tool_description_suffixes(),
    }


def _engineering_workflow_response(repo: Repository, workflow_run_id: uuid.UUID, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": "engineering_workflow",
        "tool_calls": [
            {
                "tool_name": "workflow.run_engineering_review",
                "arguments": {"repo_id": str(repo.id), "workflow_run_id": str(workflow_run_id)},
                "status": "success",
            },
            *report.get("tool_calls", []),
        ],
        "answer": report.get("final_answer") or "",
        "citations": report.get("citations") or [],
        "agent_events": report.get("agent_events") or [],
    }


def _workflow_plan_answer(spec: WorkflowSpec) -> str:
    lines = [
        "已创建执行计划，确认后我会按这个 WorkflowSpec 执行。",
        "",
        f"目标：{spec.goal}",
        "",
        "任务：",
    ]
    for index, task in enumerate(spec.tasks, start=1):
        deps = f"；依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
        critical = "；关键任务" if task.critical else ""
        lines.append(f"{index}. {task.agent_name} / {task.task_type}: {task.objective}{deps}{critical}")
    if spec.acceptance_criteria:
        lines.extend(["", "验收标准："])
        lines.extend(f"- {item}" for item in spec.acceptance_criteria)
    return "\n".join(lines)


async def _conversation_plan(payload: ChatRequest, db: Session) -> dict[str, Any]:
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    conversation_id = str(conversation.id)
    db.commit()
    intent = await select_plan_intent(payload.message)
    selected = set(intent.tasks) - intent.blocked_tasks()
    action = _risky_action(payload.message)
    comment_draft = (bool(action) or "safety_review" in selected) and _draft_action(payload.message, action) == "comment"
    context = {
        "repo_id": str(payload.repo_id),
        "requires_safety": bool(action),
    }
    # A preview needs only the identity of each target, never its diff, body or logs.
    unavailable = []
    for task_type, model, number_column, target_type, key in (
        ("issue_analysis", Issue, Issue.number, "issue", "issue"),
        ("pr_review", PullRequest, PullRequest.number, "pull_request", "pr"),
        ("ci_debug", WorkflowRun, WorkflowRun.github_run_id, "workflow_run", "workflow_run"),
    ):
        needs_comment_pr = model is PullRequest and comment_draft
        if task_type not in selected and not needs_comment_pr:
            continue
        query = db.query(model.id, number_column).filter(model.repo_id == payload.repo_id)
        if model is WorkflowRun:
            number, excluded_runs = _ci_target_numbers(payload.message)
            if excluded_runs:
                query = query.filter(number_column.is_(None) | number_column.notin_(excluded_runs))
        else:
            number = _requested_workflow_target_number(payload.message, target_type)
        if number is not None:
            query = query.filter(number_column == number)
        elif model is WorkflowRun:
            query = query.filter(WorkflowRun.conclusion == "failure")
        target = query.order_by(func.coalesce(model.updated_at, model.created_at).desc().nulls_last(), model.id).first()
        if target is None:
            label = _WORKFLOW_TARGET_LABELS[target_type]
            if number is not None:
                raise WorkflowTargetResolutionError(f"当前仓库没有同步 {label} #{number}，请检查目标编号。")
            if needs_comment_pr and re.search(_PR_SUBJECT, payload.message, re.IGNORECASE):
                raise WorkflowTargetResolutionError("当前仓库没有可用于评论的 PR。")
            if task_type in selected:
                unavailable.append("符合条件的失败 CI 运行" if model is WorkflowRun else label)
            continue
        if needs_comment_pr:
            context["comment_pr_id"] = str(target[0])
        context["has_ci_run" if model is WorkflowRun else f"has_{key}"] = True
        context[f"{key}_id"] = str(target[0])
        context[f"{key}_number"] = target[1]
    spec = PlannerAgent().build_spec(payload.message, context, intent)
    missing = "当前仓库缺少：" + "、".join(unavailable) if unavailable else ""
    if not spec.tasks:
        raise WorkflowTargetResolutionError(missing or "没有匹配到可执行任务，请说明想检查的工程问题。")
    spec_json = spec.model_dump(mode="json")
    return {
        "route": "engineering_plan",
        "conversation_id": conversation_id,
        "answer": _workflow_plan_answer(spec) + (f"\n\n{missing}，相应任务未列入计划。" if missing else ""),
        "plan": spec_json,
        "tool_calls": [
            {
                "tool_name": "planner_agent.create_workflow_spec",
                "arguments": {"repo_id": str(payload.repo_id), "workflow_id": spec.workflow_id},
                "status": "success",
            }
        ],
        "agent_events": [
            {
                "type": "workflow_spec",
                "workflow_id": spec.workflow_id,
                "goal": spec.goal,
                "tasks": [
                    {
                        "id": task.id,
                        "agent_name": task.agent_name,
                        "task_type": task.task_type,
                        "dependencies": task.dependencies,
                        "claim": task.claim.model_dump(mode="json"),
                    }
                    for task in spec.tasks
                ],
            }
        ],
    }


def _workflow_execution_trace_events(report: dict[str, Any], workflow_tool_id: str) -> list[dict[str, Any]]:
    events = [
        {
            "event": "tool_result",
            "data": {
                "tool_use_id": workflow_tool_id,
                "tool_name": "workflow.run_engineering_review",
                "status": "success",
                "route": "engineering_workflow",
                "content": {
                    "answer_preview": "Confirmed plan executed.",
                    "citations_count": len(report.get("citations") or []),
                },
            },
        }
    ]
    for item in report.get("agent_events") or []:
        if not isinstance(item, dict):
            continue
        event_type = item.get("type")
        if event_type == "workflow_task_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_{item.get('task_id')}",
                        "tool_name": f"workflow.{item.get('task_type')}",
                        "status": item.get("status") or "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": item.get("summary") or "",
                            "citations_count": item.get("evidence_count") or 0,
                        },
                    },
                }
            )
        elif event_type == "workflow_replan":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_replan_{item.get('iteration')}",
                        "tool_name": "planner_agent.replan_workflow_spec",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": (
                                f"Replan #{item.get('iteration')}: {item.get('reason')}; "
                                f"added tasks={', '.join(str(task) for task in item.get('added_tasks') or []) or 'none'}"
                            ),
                            "citations_count": 0,
                        },
                    },
                }
            )
        elif event_type == "workflow_observer_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_observer",
                        "tool_name": "workflow.observer",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": f"Observer confidence={item.get('overall_confidence')}, findings={len(item.get('findings') or [])}",
                            "citations_count": 0,
                        },
                    },
                }
            )
    return events


async def _conversation_plan_execute_sse(payload: ChatPlanExecuteRequest, db: Session):
    repo = _repo_or_404(db, payload.repo_id)

    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    if conversation:
        db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    workflow_tool_id = f"confirmed_{payload.workflow_spec.get('workflow_id') or uuid.uuid4().hex[:8]}"
    try:
        spec = WorkflowSpec.model_validate(payload.workflow_spec)
        task_types = {task.task_type for task in spec.tasks}
        if task_types.intersection(spec.excluded_task_types):
            raise WorkflowTargetResolutionError("确认计划包含已排除的任务，请重新生成计划。")
        if _risky_action(payload.message) and ("safety_review" not in task_types or not spec.requires_human_confirmation):
            raise WorkflowTargetResolutionError("此操作需要安全审查和人工确认，请重新生成计划。")
        yield encode_sse(
            {
                "event": "tool_use",
                "data": {
                    "id": workflow_tool_id,
                    "tool_name": "workflow.run_engineering_review",
                    "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id, "confirmed_plan": True},
                },
            }
        )
        workflow_context, task_runners = _build_engineering_workflow_runtime(
            repo, db, payload.message, context, excluded_task_types=tuple(spec.excluded_task_types)
        )
        report = await WorkflowOrchestrator(task_runners).run_spec(spec, workflow_context)
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        for event in _workflow_execution_trace_events(report, workflow_tool_id):
            yield encode_sse(event)

        result = _engineering_workflow_response(repo, workflow_run_id, report)
        run_id = uuid.uuid4()
        result["run_id"] = str(run_id)
        result["compression_stats"] = assembled.compression_stats
        result["progressive_compaction"] = compaction_stats
        for index in range(0, len(result["answer"]), 48):
            yield encode_sse({"event": "delta", "data": {"content": result["answer"][index : index + 48]}})

        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            route=result["route"],
            tool_calls=result["tool_calls"],
            meta={
                "run_id": str(run_id),
                "agent_events": result.get("agent_events", []),
                "model_usage": result.get("model_usage", []),
                "context_diagnostics": result.get("context_diagnostics", {}),
                "compression_stats": assembled.compression_stats,
                "progressive_compaction": compaction_stats,
            },
        )
        result["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route=result["route"],
            tool_calls=result["tool_calls"],
            agent_events=result.get("agent_events", []),
            model_usage=result.get("model_usage", []),
            context_diagnostics=result.get("context_diagnostics", {}),
        )
        db.commit()
        yield encode_sse({"event": "final", "data": result})
    except Exception as exc:
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"执行计划失败：{exc}"
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        db.commit()
        yield encode_sse(
            {
                "event": "final",
                "data": {
                    "run_id": str(run_id),
                    "message_id": str(assistant_message.id),
                    "route": "error",
                    "tool_calls": [],
                    "answer": answer,
                    "citations": [],
                    "agent_events": [],
                },
            }
        )


async def _conversation_response(payload: ChatRequest, db: Session) -> dict[str, Any]:
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    context["tools"] = _build_conversation_tools(repo, conversation, db, payload.message)
    result = await ChatAgent().run({"message": payload.message}, context)
    run_id = uuid.uuid4()
    result["run_id"] = str(run_id)
    result["compression_stats"] = assembled.compression_stats
    result["progressive_compaction"] = compaction_stats
    assistant_message: ChatMessage | None = None
    assistant_message = message_store.append(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        route=result["route"],
        tool_calls=result["tool_calls"],
        meta={
            "run_id": str(run_id),
            "agent_events": result.get("agent_events", []),
            "model_usage": result.get("model_usage", []),
            "context_diagnostics": result.get("context_diagnostics", {}),
            "compression_stats": assembled.compression_stats,
            "progressive_compaction": compaction_stats,
        },
    )
    result["message_id"] = str(assistant_message.id)
    db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
    touch_conversation(db, conversation, user_message=payload.message, increment=2)
    await SessionSealer(db).seal(
        repo=repo,
        conversation=conversation,
        user_message=user_message,
        assistant_message=assistant_message,
        route=result["route"],
        tool_calls=result["tool_calls"],
        agent_events=result.get("agent_events", []),
        model_usage=result.get("model_usage", []),
        context_diagnostics=result.get("context_diagnostics", {}),
    )
    db.commit()
    return result


async def _conversation_sse(payload: ChatRequest, db: Session):
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    context["tools"] = _build_conversation_tools(repo, conversation, db, payload.message)
    try:
        async for event in ChatAgent().run_stream({"message": payload.message}, context):
            if event["event"] == "final":
                run_id = uuid.uuid4()
                event["data"]["run_id"] = str(run_id)
                event["data"]["compression_stats"] = assembled.compression_stats
                event["data"]["progressive_compaction"] = compaction_stats
                assistant_message = message_store.append(
                    repo_id=repo.id,
                    conversation_id=conversation.id,
                    role="assistant",
                    content=event["data"]["answer"],
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    meta={
                        "run_id": str(run_id),
                        "agent_events": event["data"].get("agent_events", []),
                        "model_usage": event["data"].get("model_usage", []),
                        "context_diagnostics": event["data"].get("context_diagnostics", {}),
                        "compression_stats": assembled.compression_stats,
                        "progressive_compaction": compaction_stats,
                    },
                )
                event["data"]["message_id"] = str(assistant_message.id)
                db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=event["data"]["route"], tool_calls=event["data"]["tool_calls"], final_answer=event["data"]["answer"], status="success"))
                touch_conversation(db, conversation, user_message=payload.message, increment=2)
                await SessionSealer(db).seal(
                    repo=repo,
                    conversation=conversation,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    agent_events=event["data"].get("agent_events", []),
                    model_usage=event["data"].get("model_usage", []),
                    context_diagnostics=event["data"].get("context_diagnostics", {}),
                )
                db.commit()
            yield encode_sse(event)
    except Exception as exc:
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"请求失败：{exc}"
        failed_event = {
            "event": "final",
            "data": {
                "run_id": str(run_id),
                "route": "error",
                "tool_calls": [],
                "answer": answer,
                "citations": [],
            },
        }
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        failed_event["data"]["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        db.commit()
        yield encode_sse(failed_event)


@router.post("/plan")
async def chat_plan(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return await _conversation_plan(payload, db)
    except PlanSelectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WorkflowTargetResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/plan/execute/stream")
async def chat_plan_execute_stream(payload: ChatPlanExecuteRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    return StreamingResponse(_conversation_plan_execute_sse(payload, db), media_type="text/event-stream")


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    return await _conversation_response(payload, db)


@router.post("/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    return StreamingResponse(_conversation_sse(payload, db), media_type="text/event-stream")


@router.post("/recommendations", response_model=PromptRecommendationResponse)
async def prompt_recommendations(payload: PromptRecommendationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    repo = _repo_or_404(db, payload.repo_id)
    issues, prs, runs = _repo_payload(repo.id, db)
    result = await RecommendationAgent().run(
        _recommendation_payload(repo, issues, prs, runs, payload.exclude, payload.limit),
        {"repo_id": str(repo.id)},
    )
    return {
        "agent": result["agent"],
        "suggestions": result["suggestions"],
    }
