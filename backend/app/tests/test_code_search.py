import uuid
from types import SimpleNamespace

import pytest
from app.services import code_search
from app.services.code_search import query_terms, search_checkout_code, search_repository_code


def test_query_terms_prioritizes_code_identifiers() -> None:
    terms = query_terms("为什么 refreshAccessToken 在 auth/service.py 里失败？")

    assert "refreshaccesstoken" in terms
    assert "auth/service.py" in terms


def test_search_checkout_code_reads_current_files_without_database_index(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "auth.py"
    source.write_text(
        "def refresh_access_token():\n"
        "    return rotate_credentials()\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("REFRESH_ACCESS_TOKEN=secret\n", encoding="utf-8")

    results = search_checkout_code(tmp_path, "refresh_access_token", limit=5)

    assert results
    assert results[0]["source_type"] == "workspace_file"
    assert results[0]["source_id"] == "src/auth.py"
    assert results[0]["metadata"]["retrieval_mode"] == "workspace"
    assert all(item["source_id"] != ".env" for item in results)


def test_search_checkout_code_respects_worktree_subdirectory(tmp_path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / ".github" / "workflows" / "test.yml").write_text("name: backend-tests\n", encoding="utf-8")
    (tmp_path / "src" / "test.py").write_text("backend_tests = True\n", encoding="utf-8")

    results = search_checkout_code(
        tmp_path,
        "backend-tests",
        relative_path=".github/workflows",
    )

    assert results and {item["source_id"] for item in results} == {".github/workflows/test.yml"}


def test_search_repository_code_uses_configured_local_checkout(tmp_path) -> None:
    (tmp_path / "service.py").write_text("def handle_payment_webhook():\n    pass\n", encoding="utf-8")
    repo = SimpleNamespace(id=uuid.uuid4(), local_path=str(tmp_path), full_name="local/demo")

    results = search_repository_code(repo, "handle_payment_webhook")

    assert results and results[0]["metadata"]["repo_id"] == str(repo.id)


def test_search_checkout_code_observes_file_changes_without_reindex(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def old_handler():\n    pass\n", encoding="utf-8")
    assert search_checkout_code(tmp_path, "old_handler")

    source.write_text("def current_handler():\n    pass\n", encoding="utf-8")

    assert search_checkout_code(tmp_path, "current_handler")
    assert search_checkout_code(tmp_path, "old_handler") == []


@pytest.mark.parametrize("fallback", [False, True])
def test_search_keeps_multiple_matching_lines_in_one_file(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(code_search.shutil, "which", lambda name: None)
    (tmp_path / "worker.py").write_text("semantic_version = 1\nother = 2\nsemantic_version = 3\n", encoding="utf-8")
    results = search_checkout_code(tmp_path, "semantic_version")
    assert [r["metadata"]["line"] for r in results] == [1, 3]
    assert len(search_checkout_code(tmp_path, "semantic_version", limit=1)) == 1


@pytest.mark.parametrize("fallback", [False, True])
def test_pattern_search_uses_regex_and_not_query_or_filename(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(code_search.shutil, "which", lambda name: None)
    (tmp_path / "refresh.py").write_text("def version():\n    pass\ndef refresh():\n    pass\n_runtime = LocalSemantics()\n", encoding="utf-8")
    results = search_checkout_code(tmp_path, "semantic runtime", pattern=r"def version|def refresh", relative_path="refresh.py")
    assert [r["metadata"]["line"] for r in results] == [1, 3]
    with pytest.raises(ValueError, match="正则表达式"):
        search_checkout_code(tmp_path, "ignored", pattern="[")
