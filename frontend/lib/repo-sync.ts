import { apiGet, apiPost } from "./api";

export type RepoSyncStatus = {
  repo_id: string;
  status: string;
  message?: string;
  elapsed_seconds?: number;
  completed?: number;
  total?: number;
  synced?: Record<string, number>;
  code_sync?: { status: string; message: string; branch?: string; commit_sha?: string; finished_at?: string };
};

// Both refresh buttons follow the same bounded operation and progress endpoint.
export async function syncRepository(repoId: string, onProgress?: (status: RepoSyncStatus) => void): Promise<RepoSyncStatus> {
  let observing = true;
  let pendingPoll: Promise<RepoSyncStatus | undefined> | undefined;
  function poll() {
    if (pendingPoll) return pendingPoll;
    pendingPoll = apiGet<RepoSyncStatus>(`/api/repos/${repoId}/sync-status`)
      .then((status) => {
        if (observing) onProgress?.(status);
        return status;
      })
      .catch(() => undefined)
      .finally(() => { pendingPoll = undefined; });
    return pendingPoll;
  }
  const timer = setInterval(() => void poll(), 1_200);
  try {
    const result = await apiPost<RepoSyncStatus>(`/api/repos/${repoId}/sync`, {
      sync_issues: true,
      sync_pull_requests: true,
      sync_workflow_runs: true,
      limit: 50
    }, { timeoutMs: 105_000 });
    if (result.status === "completed") return result;
    if (result.status !== "running") throw new Error(result.message || "同步未完成，请稍后重试。");

    // Another request already owns this sync. Follow it instead of submitting
    // another full refresh after it finishes.
    onProgress?.(result);
    // Source synchronization runs after the bounded GitHub facts request.
    // Keep observing until that independent stage succeeds or reports failure.
    const deadline = Date.now() + 600_000;
    while (Date.now() < deadline) {
      const status = await poll();
      if (status?.status === "completed") return status;
      if (status?.status === "failed") throw new Error(status.message || "同步失败，请稍后重试。");
      await new Promise((resolve) => setTimeout(resolve, 1_200));
    }
    throw new Error("同步状态暂时无法确认，已保留当前页面数据，请稍后重试。");
  } finally {
    observing = false;
    clearInterval(timer);
  }
}
