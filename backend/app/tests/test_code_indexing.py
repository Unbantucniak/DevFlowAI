import base64
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import code_analysis


def test_sync_checkout_reports_failure_instead_of_returning_stale_code(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
    )
    checkout = code_analysis._repo_checkout_path(repo)
    (checkout / ".git").mkdir(parents=True)

    def fail_run(args, cwd=None, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="network failed")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    with pytest.raises(code_analysis.CheckoutSyncError, match="源码更新失败"):
        code_analysis._sync_checkout(repo, None)


def test_sync_checkout_uses_configured_local_repo_without_updating(tmp_path, monkeypatch) -> None:
    local_repo = tmp_path / "local-repo"
    local_repo.mkdir()
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(local_repo),
        checkout_mode="local",
    )

    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: path == local_repo.resolve())

    def fail_run(args, cwd=None):
        raise AssertionError("local repository indexing should not fetch, checkout, or pull")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == local_repo.resolve()


def test_sync_checkout_rejects_non_empty_configured_download_target(tmp_path, monkeypatch) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "README.md").write_text("# occupied\n", encoding="utf-8")
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(target),
        checkout_mode="managed",
    )

    monkeypatch.setattr(code_analysis.shutil, "which", lambda name: "git")
    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: False)

    try:
        code_analysis._sync_checkout(repo, None)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected configured non-empty download target to be rejected")


@pytest.mark.parametrize("repo_token", [None, "repository-token"])
def test_private_checkout_authenticates_clone_and_updates_without_token_in_url(tmp_path, monkeypatch, repo_token):
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    monkeypatch.setattr(settings, "github_token", "configured-token")
    monkeypatch.setattr(code_analysis.shutil, "which", lambda name: "git")
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    repo = SimpleNamespace(
        id=uuid.uuid4(), full_name="owner/private-repo",
        clone_url="https://github.com/owner/private-repo.git", default_branch="main",
    )
    calls = []
    monkeypatch.setattr(code_analysis, "_run_git", lambda args, cwd=None, **kwargs: calls.append((args, kwargs.get("env"))))
    checkout = code_analysis._sync_checkout(repo, repo_token)
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(code_analysis, "_git_text", lambda args, cwd: "a" * 40 if args[0] == "rev-parse" and args[-1] != "--is-shallow-repository" else None)
    code_analysis._sync_checkout(repo, repo_token)
    network_calls = [(args, env) for args, env in calls if args[0] in {"clone", "fetch", "pull"}]
    assert [args[0] for args, _ in network_calls] == ["clone", "fetch"]
    assert repo.clone_url in network_calls[0][0]
    for args, env in network_calls:
        assert "configured-token" not in " ".join(args)
        assert "repository-token" not in " ".join(args)
        assert env["GIT_CONFIG_KEY_0"] == f"http.{repo.clone_url}.extraHeader"
        auth = env["GIT_CONFIG_VALUE_0"].removeprefix("Authorization: Basic ")
        assert base64.b64decode(auth).decode() == f"x-access-token:{repo_token or 'configured-token'}"
        assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_git_auth_header_is_scoped_to_repository(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    url = "https://github.com/owner/private-repo.git"
    env = code_analysis._git_auth_env(url, "test-token")
    matched = subprocess.run(
        ["git", "config", "--get-urlmatch", "http.extraHeader", url + "/info/refs"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert matched.stdout.startswith("Authorization: Basic ")
    unrelated = subprocess.run(
        ["git", "config", "--get-urlmatch", "http.extraHeader", "https://example.com/other.git"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert unrelated.returncode == 1
    assert not unrelated.stdout
