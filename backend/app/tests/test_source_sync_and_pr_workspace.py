import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base, Repository, PullRequest, PRFile
from app.services import code_analysis as code
from app.services import worktree_manager as worktrees
from app.api.routes import chat, repos
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse


def git(root, *args):
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def commit(root, name, content):
    (root / name).write_text(content, encoding="utf-8")
    git(root, "add", name)
    git(root, "commit", "-m", content[:40])
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    remote = tmp_path / "origin.git"
    author = tmp_path / "author"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(tmp_path, "clone", remote.as_uri(), str(author))
    git(author, "config", "user.name", "Source Sync Test")
    git(author, "config", "user.email", "test@example.invalid")
    first = commit(author, "README.md", "base source\n")
    git(author, "push", "origin", "main")
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path / "checkouts"))
    repo = SimpleNamespace(id=uuid.uuid4(), owner="test", name="source", full_name="test/source",
                           clone_url=remote.as_uri(), default_branch="main", local_path=None,
                           checkout_mode="managed", provider="github", github_token_encrypted=None)
    root = code._sync_checkout(repo, None)
    return repo, author, remote, root, first


def database():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_recovers_existing_disconnected_shallow_history_and_future_updates(repository):
    repo, author, _, root, first = repository
    commit(author, "README.md", "second version\n")
    latest = commit(author, "added.py", "new_feature = True\n")
    git(author, "push", "origin", "main")
    git(root, "fetch", "--depth", "1", "origin", "main")
    assert git(root, "rev-parse", "HEAD") == first
    assert subprocess.run(["git", "merge-base", "HEAD", "origin/main"], cwd=root, capture_output=True).returncode == 1
    assert code._sync_checkout(repo, None) == root
    assert git(root, "rev-parse", "HEAD") == latest
    assert git(root, "rev-parse", "--is-shallow-repository") == "false"
    assert (root / "added.py").read_text() == "new_feature = True\n"
    newer = commit(author, "README.md", "third version\n")
    git(author, "push", "origin", "main")
    code._sync_checkout(repo, None)
    assert git(root, "rev-parse", "HEAD") == newer


def test_real_divergence_and_uncommitted_changes_are_preserved(repository):
    repo, author, _, root, _ = repository
    git(root, "config", "user.name", "Local Test")
    git(root, "config", "user.email", "local@example.invalid")
    local_sha = commit(root, "local.py", "local_only = True\n")
    commit(author, "remote.py", "remote_only = True\n")
    git(author, "push", "origin", "main")
    with pytest.raises(code.CheckoutSyncError, match="分叉"):
        code._sync_checkout(repo, None)
    assert git(root, "rev-parse", "HEAD") == local_sha
    (root / "README.md").write_text("user edits", encoding="utf-8")
    with pytest.raises(code.CheckoutSyncError, match="未提交修改"):
        code._sync_checkout(repo, None)
    assert (root / "README.md").read_text() == "user edits"


@pytest.mark.asyncio
async def test_code_failure_is_persisted_reported_and_blocks_stale_reads(monkeypatch, tmp_path):
    with database() as db:
        repo = Repository(owner="test", name="repo", full_name="test/repo", local_path=str(tmp_path),
                          code_sync={"status": "ready", "commit_sha": "old"})
        db.add(repo)
        db.commit()
        def fail(*args):
            raise code.CheckoutSyncError("源码更新失败：网络不可达")
        monkeypatch.setattr(code, "_sync_checkout", fail)
        with pytest.raises(code.CheckoutSyncError):
            await code.sync_repository_code_analysis(db, repo)
        db.expire_all()
        assert repo.code_sync["status"] == "failed"
        assert repo.code_sync["commit_sha"] == "old"
        view = code.repository_sync_view(repo, {"status": "completed", "synced": {"pull_requests": 1}})
        assert view["status"] == "failed"
        assert "网络不可达" in view["message"]
        assert view["synced"]["pull_requests"] == 1
        with pytest.raises(ValueError, match="不能把旧文件"):
            chat._workspace_root(repo)


@pytest.mark.asyncio
async def test_sync_endpoint_does_not_report_completion_before_code_stage(monkeypatch):
    with database() as db:
        repo = Repository(owner="test", name="repo", full_name="test/repo")
        db.add(repo)
        db.commit()
        async def facts(*args):
            return RepoSyncResponse(repo_id=repo.id, status="completed", synced={"pull_requests": 2})
        monkeypatch.setattr(repos, "sync_repository", facts)
        tasks = BackgroundTasks()
        result = await repos.sync_repo(repo.id, RepoSyncRequest(), tasks, db)
        assert result.status == "running"
        assert result.code_sync["status"] == "pending"
        assert len(tasks.tasks) == 1
        assert code.repository_sync_view(repo, {"status": "completed"})["status"] == "running"
        code.recover_interrupted_code_syncs(db)
        assert code.repository_sync_view(repo, {"status": "completed"})["status"] == "failed"


@pytest.mark.asyncio
async def test_scheduled_sync_also_updates_source_and_persists_failure(monkeypatch):
    from app.services import repos_sync
    with database() as db:
        repo = Repository(owner="test", name="repo", full_name="test/repo")
        db.add(repo); db.commit()
        repo_id = repo.id
        monkeypatch.setattr("app.db.session.SessionLocal", lambda: db)

        async def facts(*args):
            return RepoSyncResponse(repo_id=repo_id, status="completed", synced={"pull_requests": 1})

        def fail_checkout(*args):
            raise code.CheckoutSyncError("定时更新源码失败")

        monkeypatch.setattr(repos_sync, "sync_repository", facts)
        monkeypatch.setattr(code, "_sync_checkout", fail_checkout)
        with pytest.raises(code.CheckoutSyncError, match="定时更新"):
            await repos_sync.sync_repository_by_id(repo_id, RepoSyncRequest())
        assert db.get(Repository, repo_id).code_sync["status"] == "failed"


def test_pr_ref_works_without_head_branch_and_rejects_mismatched_version(repository, monkeypatch):
    repo, author, remote, root, first = repository
    git(author, "checkout", "-b", "feature")
    head = commit(author, "only_in_pr.py", "pr_content = 13\n")
    git(author, "push", "origin", "HEAD:refs/pull/13/head")
    with database() as db:
        saved_repo = Repository(**vars(repo))
        db.add(saved_repo)
        db.flush()
        pr = PullRequest(repo_id=repo.id, number=13, title="Feature", head_branch="deleted-feature", head_sha=head)
        db.add(pr)
        db.commit()
        monkeypatch.setattr(worktrees, "_sync_pr_snapshot_analysis", lambda *args: 0)
        result = worktrees.prepare_pr_worktree(db, saved_repo, pr)
        assert result["status"] == "ready", result
        assert result["commit_sha"] == head
        assert Path(result["snapshot_path"], "only_in_pr.py").read_text() == "pr_content = 13\n"
        assert not (root / "only_in_pr.py").exists()
        assert git(root, "rev-parse", "HEAD") == first
        assert worktrees.prepare_pr_worktree(db, saved_repo, pr)["status"] == "ready"
        pr.head_sha = first
        db.commit()
        mismatch = worktrees.prepare_pr_worktree(db, saved_repo, pr)
        assert mismatch["status"] == "failed"
        assert "不一致" in mismatch["message"]
        assert mismatch["checkout_kind"] == "unavailable"


def test_pr_snapshot_supports_deep_windows_paths(repository, monkeypatch):
    repo, author, _, _, _ = repository
    relative = Path("benchmarks") / ("deep-java-module-" * 3) / ("nested-package-" * 3) / "src/main/java/LongPath.java"
    (author / relative).parent.mkdir(parents=True)
    head = commit(author, str(relative), "class LongPath {}\n")
    git(author, "push", "origin", "HEAD:refs/pull/13/head")
    with database() as db:
        saved_repo = Repository(**vars(repo))
        db.add(saved_repo); db.flush()
        pr = PullRequest(repo_id=repo.id, number=13, title="Long path", head_sha=head)
        db.add(pr); db.commit()
        monkeypatch.setattr(worktrees, "_sync_pr_snapshot_analysis", lambda *args: 0)
        result = worktrees.prepare_pr_worktree(db, saved_repo, pr)
        assert result["status"] == "ready", result
        target = Path(result["snapshot_path"]) / relative
        assert len(str(target)) > 260
        assert target.read_text() == "class LongPath {}\n"


@pytest.mark.asyncio
async def test_chat_reads_requested_pr_snapshot_and_never_falls_back_to_main(tmp_path, monkeypatch):
    main = tmp_path / "main"
    snapshot = tmp_path / "snapshot"
    main.mkdir(); snapshot.mkdir()
    (main / "feature.py").write_text("WRONG_MAIN_VERSION", encoding="utf-8")
    (snapshot / "feature.py").write_text("CORRECT_PR_VERSION", encoding="utf-8")
    with database() as db:
        repo = Repository(owner="team", name="repo", full_name="team/repo", local_path=str(main))
        other_repo = Repository(owner="team", name="other", full_name="team/other")
        db.add_all([repo, other_repo]); db.flush()
        pr = PullRequest(repo_id=repo.id, number=13, title="Requested", head_sha="a" * 40)
        newer = PullRequest(repo_id=repo.id, number=14, title="Wrong newer PR")
        foreign = PullRequest(repo_id=other_repo.id, number=13, title="Wrong repository")
        db.add_all([pr, newer, foreign]); db.flush()
        db.add(PRFile(pr_id=pr.id, filename="feature.py", status="added", patch="+CORRECT_PR_VERSION"))
        db.commit()
        calls = []
        def prepare(repo_id, pr_id):
            calls.append((repo_id, pr_id))
            return {"status": "ready", "snapshot_path": str(snapshot), "commit_sha": pr.head_sha}
        monkeypatch.setattr(chat, "prepare_pr_worktree_by_id", prepare)
        tools = chat._build_conversation_tools(repo, None, db, "请审查PR13并读取代码")
        read = await tools["workspace.read_file"]({"path": "feature.py"}, {})
        assert "CORRECT_PR_VERSION" in read["answer"]
        assert "WRONG_MAIN" not in read["answer"]
        assert read["tool_calls"][0]["arguments"]["pr_number"] == 13
        await tools["workspace.list_files"]({}, {})
        assert calls == [(repo.id, pr.id)]
        with pytest.raises(ValueError):
            await tools["workspace.read_file"]({"path": "feature.py", "pr_id": str(foreign.id)}, {})
        with pytest.raises(ValueError):
            await tools["workspace.read_file"]({"path": "feature.py", "pr_number": 14}, {})
        review_calls = []
        async def review(self, payload, context):
            review_calls.append(payload["number"])
            assert payload["number"] == 13
            assert "CORRECT_PR_VERSION" in context["code_references"][0]["snippet"]
            return {"summary": "Grounded review", "risk_points": [], "test_suggestions": []}
        monkeypatch.setattr(chat.PRReviewAgent, "run", review)
        analysis = await tools["pr_review_agent.analyze_pr"]({}, {})
        assert "Grounded review" in analysis["answer"]
        await tools["pr_review_agent.analyze_pr"]({}, {})
        assert review_calls == [13]
        monkeypatch.setattr(chat, "prepare_pr_worktree_by_id", lambda *args: {"status": "failed", "message": "快照失败"})
        failed_tools = chat._build_conversation_tools(repo, None, db, "审查PR13")
        with pytest.raises(ValueError, match="快照失败"):
            await failed_tools["workspace.read_file"]({"path": "feature.py"}, {})
