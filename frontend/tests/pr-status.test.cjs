const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

const source = fs.readFileSync(path.join(__dirname, "../lib/pr-status.ts"), "utf8");
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
const statusModule = { exports: {} };
vm.runInNewContext(code, { module: statusModule, exports: statusModule.exports });
const { classifyPr } = statusModule.exports;

test("merged PR #12 remains completed even when its diff contains conflict", () => {
  assert.equal(classifyPr({
    number: 12, state: "closed", merged_at: "2026-11-15T14:37:14Z", mergeable: false,
    title: "feat: add validated V3 retrieval and experience merge suggestions",
    files: [{ filename: "backend/experience/store.py", patch: "+ raise ExperienceConflictError()" }],
  }), "done");
});

test("keywords in an open PR title, body, filename or patch do not indicate a merge conflict", () => {
  assert.equal(classifyPr({
    state: "open", author: "developer", mergeable: true,
    title: "Fix conflict handling", body: "No conflicts remain",
    files: [{ filename: "conflicts.py", patch: "+ test_conflict()" }],
  }), "reviewing");
});

test("an explicit GitHub mergeability result detects conflicts without keywords", () => {
  assert.equal(classifyPr({ state: "open", author: "developer", mergeable: false, title: "Update README" }), "conflict");
});

test("unknown mergeability does not become a conflict", () => {
  assert.equal(classifyPr({ state: "open", author: "developer", mergeable: null }), "reviewing");
  assert.equal(classifyPr({ state: "open", author: "developer" }), "reviewing");
});

test("a closed unmerged PR remains closed even with stale conflict information", () => {
  assert.equal(classifyPr({ state: "closed", merged_at: null, mergeable: false }), "closed");
});
