const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function load(name, api = {}, extra = {}) {
  const source = fs.readFileSync(path.join(__dirname, "../lib", name + ".ts"), "utf8");
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const module = { exports: {} };
  const intervals = new Map();
  let sequence = 0;
  vm.runInNewContext(code, {
    module, exports: module.exports,
    require: () => api,
    process: { env: {} }, AbortController, Response,
    setInterval: (callback) => { const id = ++sequence; intervals.set(id, callback); return id; },
    clearInterval: (id) => intervals.delete(id),
    setTimeout: (callback) => setTimeout(callback, 0), clearTimeout,
    ...extra,
  });
  return { ...module.exports, intervals };
}

test("a complete refresh reports progress and finishes without extra work", async () => {
  let finish;
  let postCount = 0;
  const updates = [];
  const api = {
    apiPost: async (url, payload, options) => {
      postCount++;
      assert.equal(url, "/api/repos/project/sync");
      assert.equal(payload.limit, 50);
      assert.equal(options.timeoutMs, 105000);
      return new Promise((resolve) => { finish = resolve; });
    },
    apiGet: async () => ({ repo_id: "project", status: "running", completed: 30, total: 74 }),
  };
  const runtime = load("repo-sync", api);
  const promise = runtime.syncRepository("project", (status) => updates.push(status));
  await [...runtime.intervals.values()][0]();
  await new Promise((resolve) => setImmediate(resolve));
  finish({ repo_id: "project", status: "completed", synced: { pull_requests: 12, workflow_runs: 50 } });
  const result = await promise;
  assert.equal(updates[0].completed, 30);
  assert.equal(result.synced.workflow_runs, 50);
  assert.equal(postCount, 1);
  assert.equal(runtime.intervals.size, 0);
});

test("an existing sync is followed without submitting a second sync", async () => {
  let posts = 0;
  const states = [
    { status: "running", completed: 1 },
    { status: "completed", synced: { pull_requests: 12 } },
  ];
  const runtime = load("repo-sync", {
    apiPost: async () => { posts++; return { status: "running" }; },
    apiGet: async (url) => {
      assert.equal(url, "/api/repos/project/sync-status");
      return states.shift();
    },
  });
  const result = await runtime.syncRepository("project");
  assert.equal(result.status, "completed");
  assert.equal(posts, 1);
  assert.equal(states.length, 0);
  assert.equal(runtime.intervals.size, 0);
});

test("a failed shared sync stops observation and exposes the error", async () => {
  const runtime = load("repo-sync", {
    apiPost: async () => ({ status: "running" }),
    apiGet: async () => ({ status: "failed", message: "GitHub 同步超时，原有数据已保留" }),
  });
  await assert.rejects(runtime.syncRepository("project"), /GitHub 同步超时/);
  assert.equal(runtime.intervals.size, 0);
});

test("a stalled POST has a bounded wait and aborts its request", async () => {
  let signal;
  const runtime = load("api", {}, {
    setTimeout,
    fetch: async (_url, options) => {
      signal = options.signal;
      return new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      });
    },
  });
  await assert.rejects(runtime.apiPost("/sync", {}, { timeoutMs: 5 }), /请求等待超时/);
  assert.equal(signal.aborted, true);
});
