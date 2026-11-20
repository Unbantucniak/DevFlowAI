from io import BytesIO
from zipfile import BadZipFile, ZipFile

from app.services.github.client import GitHubClient


async def list_workflow_runs(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs", {"per_page": limit})
    return payload.get("workflow_runs", [])


async def list_pr_workflow_runs(owner: str, repo: str, pr_number: int, head_sha: str,
                                token: str | None = None, base_url: str | None = None) -> dict:
    """Read Actions for the reviewed commit, retaining the latest run per workflow/event."""
    if not head_sha:
        return {"runs": [], "complete": False}
    client = GitHubClient(token, base_url)
    latest: dict[tuple, dict] = {}
    complete = False
    # GitHub caps filtered workflow searches at 1,000 results.
    for page in range(1, 11):
        payload = await client.get(f"/repos/{owner}/{repo}/actions/runs",
                                   {"head_sha": head_sha, "per_page": 100, "page": page})
        runs = payload.get("workflow_runs", [])
        for run in runs:
            if run.get("head_sha") != head_sha:
                continue
            associated = run.get("pull_requests") or []
            if associated and not any(item.get("number") == pr_number for item in associated):
                continue
            key = (run.get("workflow_id") or run.get("path") or run.get("name"),
                   run.get("event"), run.get("head_branch"))
            order = lambda item: (int(item.get("run_number") or 0), int(item.get("id") or 0), int(item.get("run_attempt") or 1))
            if key not in latest or order(run) > order(latest[key]):
                latest[key] = {**run, "matched_by": "pr_number + head_sha" if associated else "head_sha"}
        if len(runs) < 100 or page * 100 >= int(payload.get("total_count") or 0):
            complete = True
            break
    return {"runs": sorted(latest.values(), key=lambda item: int(item.get("id") or 0), reverse=True),
            "complete": complete}


async def list_workflow_run_jobs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", {"per_page": 100})
    return payload.get("jobs", [])


async def get_workflow_run_logs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None, max_chars: int = 80000) -> str:
    client = GitHubClient(token, base_url)
    payload = await client.get_bytes(f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            parts: list[str] = []
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                raw = archive.read(name)
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"### {name}\n{text}")
                if sum(len(part) for part in parts) >= max_chars:
                    break
            return "\n\n".join(parts)[:max_chars]
    except BadZipFile:
        return payload.decode("utf-8", errors="replace")[:max_chars]
