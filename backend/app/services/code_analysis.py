"""Repository checkout synchronization and code-graph analysis."""

import asyncio
import base64
import logging
import os
import shutil
import subprocess
import uuid
import re
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, Document, Repository
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

_CHECKOUT_LOCKS: dict[str, threading.RLock] = {}
_CHECKOUT_LOCKS_GUARD = threading.Lock()


class CheckoutSyncError(RuntimeError):
    """A safe, user-facing failure; never silently return stale source."""


def checkout_lock(repo_id: uuid.UUID) -> threading.RLock:
    with _CHECKOUT_LOCKS_GUARD:
        return _CHECKOUT_LOCKS.setdefault(str(repo_id), threading.RLock())

SKIP_DIRS = {
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

TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".md",
    ".mdx",
    ".txt",
    ".dockerfile",
}

TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
}

PY_SYMBOL_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
CALL_RE = re.compile(r"\b([A-Za-z_][$A-Za-z0-9_]*)\s*\(")
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.\/-]+)\s+import|import\s+([\w.\/-]+)|import\s+.*?\s+from\s+['\"]([^'\"]+)['\"])", re.MULTILINE)


def _checkout_root() -> Path:
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_checkout_path(repo: Repository) -> Path:
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / f"{repo.id}-{safe_name}"


def _stored_checkout_path(repo: Repository) -> Path | None:
    raw_path = getattr(repo, "local_path", None)
    if not raw_path:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _clone_url(repo: Repository) -> str:
    if repo.clone_url:
        return repo.clone_url
    return f"https://github.com/{repo.owner}/{repo.name}.git"


def _git_auth_env(url: str, token: str | None) -> dict[str, str]:
    """Pass Git credentials for this URL without storing them in the remote."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    parsed = urlparse(url)
    if token and parsed.scheme in {"http", "https"} and "@" not in parsed.netloc:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        for key, value in (
            (f"http.{url}.extraHeader", f"Authorization: Basic {auth}"),
            ("credential.helper", ""),
        ):
            env[f"GIT_CONFIG_KEY_{count}"] = key
            env[f"GIT_CONFIG_VALUE_{count}"] = value
            count += 1
        env["GIT_CONFIG_COUNT"] = str(count)
    return env


def _run_git(args: list[str], cwd: Path | None = None, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["git", "-c", "core.longpaths=true", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        env=env,
    )


def _is_git_work_tree(path: Path) -> bool:
    if not path.exists() or not path.is_dir() or not shutil.which("git"):
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(path),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _sync_checkout(repo: Repository, token: str | None) -> Path:
    with checkout_lock(repo.id):
        return _sync_checkout_locked(repo, token)


def _sync_checkout_locked(repo: Repository, token: str | None) -> Path:
    stored_path = _stored_checkout_path(repo)
    checkout_mode = getattr(repo, "checkout_mode", None) or "managed"

    if stored_path and checkout_mode == "local":
        if not _is_git_work_tree(stored_path):
            raise RuntimeError(f"已配置的本地仓库路径不是 git 工作树：{stored_path}")
        return stored_path

    target = stored_path or _repo_checkout_path(repo)
    clone_url = _clone_url(repo)
    git_env = _git_auth_env(clone_url, token or settings.github_token)
    branch = repo.default_branch or "main"

    if not shutil.which("git"):
        raise RuntimeError("未找到 git 可执行文件，无法克隆仓库代码")

    if (target / ".git").exists() or _is_git_work_tree(target):
        if _git_text(["status", "--porcelain"], target):
            raise CheckoutSyncError("源码目录存在未提交修改，已停止自动更新并保留原文件。请先处理本地修改。")
        try:
            # A depth-1 fetch disconnects successive tips. Recover ancestry once,
            # then use ordinary incremental fetches. Never reset local commits.
            fetch_args = ["fetch"]
            if _git_text(["rev-parse", "--is-shallow-repository"], target) == "true":
                fetch_args.append("--unshallow")
            _run_git([*fetch_args, "origin", branch], cwd=target, env=git_env)
            target_sha = _git_text(["rev-parse", "FETCH_HEAD"], target)
            if not target_sha:
                raise CheckoutSyncError("未能确定远端源码版本，已停止更新。")
            _run_git(["checkout", branch], cwd=target)
            _run_git(["merge", "--ff-only", target_sha], cwd=target)
            if _git_text(["rev-parse", "HEAD"], target) != target_sha:
                raise CheckoutSyncError("本地分支包含远端没有的提交，未覆盖本地代码；请处理分叉后重试。")
        except subprocess.CalledProcessError as exc:
            action = next((arg for arg in exc.cmd if arg in {"fetch", "checkout", "merge"}), "git") if isinstance(exc.cmd, list) else "git"
            reason = ("本地与远端不能直接快进，请处理提交分叉后重试。" if action == "merge"
                      else "请检查远端分支、网络和仓库访问权限。")
            raise CheckoutSyncError(f"源码更新失败（{action}）；原目录已保留。{reason}") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CheckoutSyncError("源码更新超时或 Git 无法运行，原目录已保留，请稍后重试。") from exc
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise RuntimeError(f"Configured download target already exists and is not an empty git work tree: {target}")
    clone_args = ["clone", "--depth", "1"]
    if branch:
        clone_args.extend(["--branch", branch])
    clone_args.extend([clone_url, str(target)])
    try:
        _run_git(clone_args, env=git_env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CheckoutSyncError("源码下载失败，请检查 Git、网络和仓库访问权限后重试。") from exc
    return target


def _is_text_candidate(path: Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True
    suffix = path.suffix.lower()
    return suffix in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 300_000) -> str | None:
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _language_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or path.name


def _git_text(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _symbol_matches(path: Path, text: str) -> list[dict[str, object]]:
    language = _language_for(path)
    matches = []
    if path.suffix.lower() == ".py":
        for match in PY_SYMBOL_RE.finditer(text):
            matches.append({"name": match.group(2), "kind": match.group(1), "language": language, "start_line": _line_for_offset(text, match.start())})
    elif path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        for match in TS_SYMBOL_RE.finditer(text):
            declaration = match.group(0)
            kind = "class" if "class " in declaration else "interface" if "interface " in declaration else "type" if "type " in declaration else "function"
            matches.append({"name": match.group(1), "kind": kind, "language": language, "start_line": _line_for_offset(text, match.start())})
    return matches


def _sync_code_graph(
    db: Session,
    repo: Repository,
    checkout_path: Path,
    *,
    branch: str | None,
    commit_sha: str | None,
    pr_id: uuid.UUID | None = None,
    pr_number: int | None = None,
    max_files: int = 500,
) -> int:
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id, CodeRelation.branch == branch, CodeRelation.pr_id == pr_id, CodeRelation.pr_number == pr_number).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id, CodeSymbol.branch == branch, CodeSymbol.pr_id == pr_id, CodeSymbol.pr_number == pr_number).delete(synchronize_session=False)
    symbol_rows: list[CodeSymbol] = []
    files_seen = 0
    for path in checkout_path.rglob("*"):
        if files_seen >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts):
            continue
        if not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        symbols = _symbol_matches(path, text)
        if not symbols:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()
        files_seen += 1
        for item in symbols:
            row = CodeSymbol(
                repo_id=repo.id,
                path=relative_path,
                name=str(item["name"]),
                kind=str(item["kind"]),
                language=str(item["language"]),
                start_line=int(item["start_line"]),
                end_line=None,
                branch=branch,
                commit_sha=commit_sha,
                pr_id=pr_id,
                pr_number=pr_number,
                meta={},
            )
            db.add(row)
            symbol_rows.append(row)
    db.flush()
    by_name = {row.name: row for row in symbol_rows}
    for path in checkout_path.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts) or not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()
        import_targets = [next((part for part in match.groups() if part), "") for match in IMPORT_RE.finditer(text)]
        for target in [item for item in import_targets if item]:
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    source_name=relative_path,
                    target_name=target,
                    relation_type="imports",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    meta={},
                )
            )
        local_symbols = [row for row in symbol_rows if row.path == relative_path]
        source = local_symbols[0] if local_symbols else None
        seen_calls: set[str] = set()
        for match in CALL_RE.finditer(text):
            target_name = match.group(1)
            if target_name in seen_calls or target_name in {"if", "for", "while", "return", "print", "len", "str", "int"}:
                continue
            seen_calls.add(target_name)
            target = by_name.get(target_name)
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    source_symbol_id=source.id if source else None,
                    target_symbol_id=target.id if target else None,
                    source_name=source.name if source else relative_path,
                    target_name=target_name,
                    relation_type="calls",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    meta={"line": _line_for_offset(text, match.start())},
                )
            )
    return len(symbol_rows)


def purge_legacy_code_documents(db: Session, repo_id: uuid.UUID | None = None) -> int:
    query = db.query(Document).filter(Document.source_type == "code_file")
    if repo_id is not None:
        query = query.filter(Document.repo_id == repo_id)
    return query.delete(synchronize_session=False)


async def sync_repository_code_analysis(db: Session, repo: Repository) -> int:
    previous = dict(repo.code_sync or {})
    repo.code_sync = {**previous, "status": "running", "message": "协作数据已同步，正在更新源码…",
                      "started_at": datetime.now(timezone.utc).isoformat()}
    db.commit()
    try:
        token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
        checkout_repo = SimpleNamespace(**{key: getattr(repo, key) for key in
            ("id", "full_name", "owner", "name", "clone_url", "default_branch", "local_path", "checkout_mode")})
        checkout_path = await asyncio.to_thread(_sync_checkout, checkout_repo, token)
        branch = _git_text(["branch", "--show-current"], checkout_path) or repo.default_branch or "main"
        commit_sha = _git_text(["rev-parse", "HEAD"], checkout_path)
        # Keep this Session on its owning thread. Git/network I/O above is offloaded.
        symbol_count = _sync_code_graph(db, repo, checkout_path, branch=branch, commit_sha=commit_sha)
        purge_legacy_code_documents(db, repo.id)
        repo.clone_url = repo.clone_url or _clone_url(repo)
        local = repo.checkout_mode == "local"
        message = ("已读取本地源码（本地模式不会自动拉取远端）" if local else "源码已更新")
        repo.code_sync = {"status": "local" if local else "ready", "message": message,
                          "branch": branch, "commit_sha": commit_sha,
                          "finished_at": datetime.now(timezone.utc).isoformat()}
        db.commit()
        return symbol_count
    except Exception as exc:
        db.rollback()
        message = str(exc) if isinstance(exc, CheckoutSyncError) else "源码更新或代码分析失败，请检查后台日志后重试。"
        repo.code_sync = {**previous, "status": "failed", "message": message,
                          "finished_at": datetime.now(timezone.utc).isoformat()}
        db.commit()
        raise


def recover_interrupted_code_syncs(db: Session) -> None:
    for repo in db.query(Repository).all():
        if (repo.code_sync or {}).get("status") in {"pending", "running"}:
            repo.code_sync = {**repo.code_sync, "status": "failed", "message": "服务重启中断了源码更新，请重新同步。"}
    db.commit()


def queue_code_sync(db: Session, repo: Repository) -> dict:
    repo.code_sync = {**(repo.code_sync or {}), "status": "pending",
                      "message": "协作数据已同步，源码更新排队中…",
                      "started_at": datetime.now(timezone.utc).isoformat()}
    db.commit()
    return dict(repo.code_sync)


def repository_sync_view(repo: Repository, facts: dict | None) -> dict:
    result = dict(facts or {"repo_id": str(repo.id), "status": "completed" if repo.last_sync_at else "not_started"})
    code = dict(repo.code_sync or {})
    result.update(code_sync=code, last_sync_at=repo.last_sync_at)
    if result.get("status") in {"running", "failed"}:
        return result
    status = code.get("status")
    if status in {"pending", "running", "ready", "local", "failed"}:
        started_at = result.get("started_at") or code.get("started_at")
        finished_at = None if status in {"pending", "running"} else code.get("finished_at")
        result["finished_at"] = finished_at
        if started_at:
            try:
                start = datetime.fromisoformat(started_at)
                end = datetime.fromisoformat(finished_at) if finished_at else datetime.now(timezone.utc)
                result["elapsed_seconds"] = round(max((end - start).total_seconds(), 0), 1)
            except (TypeError, ValueError):
                pass
    if status in {"pending", "running"}:
        result.update(status="running", message=code["message"])
    elif status == "failed":
        result.update(status="failed", message="协作数据已保留；" + code["message"])
    elif status in {"ready", "local"}:
        sha = str(code.get("commit_sha") or "")[:12]
        result["message"] = f"{result.get('message') or '协作数据已同步'}；{code['message']} {sha}".strip()
    else:
        result["message"] = "协作数据已同步，源码版本尚未确认，请同步仓库。" if repo.last_sync_at else "仓库还没有同步过。"
    return result


async def sync_repository_code_analysis_by_id(repo_id: uuid.UUID) -> int | None:
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        try:
            return await sync_repository_code_analysis(db, repo)
        except Exception as exc:
            logger.warning("Failed to sync repository code analysis for %s: %s", repo_id, exc)
            return None
    finally:
        db.close()
