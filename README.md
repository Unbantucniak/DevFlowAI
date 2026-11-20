# DevFlowAI

DevFlowAI 把 GitHub 仓库的 Issue、Pull Request 和 Actions 运行记录放在同一个工作台里。审查 PR 时可以查看改动文件、检索对应版本的源码；排查 CI 时可以展开失败的 Job，在对话里追问具体的报错。

## 连接仓库后的用法

- **审查 PR**　打开一个 PR，查看变更摘要、受影响的文件和测试建议。要判断能否合并，可以在对话里同时检查 diff 和 CI 结果，追问某个函数有哪些调用方。
- **处理 Issue 和 CI**　在 Issue 页面查看相似问题与待办事项，在 CI 页面查看失败步骤和日志。需要回复时可以生成评论草稿，确认后发回 GitHub。
- **查询项目知识**　向知识库上传文件后，系统会切分文本、生成向量，结合关键词检索和重排选出相关段落。回答中的编号引用可以用来核对原文。
- **生成周报**　按仓库汇总近期的 Issue、PR 和工作流运行情况。

涉及多个问题时，[工作流编排器](backend/app/services/agents/workflow_orchestrator.py) 会把代码审查和 CI 排障拆成单独的任务，再汇总结果。知识问答的评测页面使用 Ragas 记录检索召回和回答忠实度，可以展开单个用例查看回答、引用和评分。

后端使用 Python、FastAPI 和 SQLAlchemy，前端使用 Next.js 和 TypeScript。PostgreSQL 保存仓库记录、会话和操作草稿，Milvus 保存知识库向量。仓库同步和检索实现在 [backend/app/services](backend/app/services)，页面位于 [frontend/app](frontend/app)。

## 本地运行

以下命令适用于 Windows PowerShell。需要 Git、Python 3.12、Node.js 20 或更高版本，以及带 Compose 插件的 Docker。

**1. 配置环境。** 首次安装时，在仓库根目录复制配置示例。

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`，用于对话和代码审查。连接私有仓库时，可以在连接页面填写有该仓库读取权限的 GitHub Token，也可以配置 `GITHUB_TOKEN`。默认的向量和重排模型分别为 `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Reranker-0.6B`，首次使用时会下载到本地。

**2. 启动数据库和向量服务，安装依赖。** 以下命令均在仓库根目录执行。

```powershell
docker compose up -d --build --wait postgres etcd minio milvus
py -3.12 -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -r backend\requirements-local-embedding.txt
npm --prefix frontend ci
```

**3. 分别启动后端和前端。** 在第一个终端启动 FastAPI。

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

在第二个终端，从仓库根目录启动 Next.js。

```powershell
npm --prefix frontend run dev
```

打开 `http://localhost:3000/repos`，填写仓库地址并同步。同步完成后即可查看 Issue、PR 和 CI 记录。API 接口列表在 `http://localhost:8000/docs`。

### 已配置的 Windows GPU 环境

[start.cmd](start.cmd) 用于已经登记 Docker 数据卷、配置好 CUDA 和本地模型的环境。它会检查数据卷和模型，启动后端（`8001`）和前端（`3000`），然后打开首页。[stop.cmd](stop.cmd) 会备份数据库、上传文件和配置，再停止容器。备份目录为 `.data/backups/`。

## 测试

在 `backend` 目录使用项目虚拟环境运行后端测试。

```powershell
.\.venv\Scripts\python.exe -m pytest app/tests
```

从仓库根目录运行前端的 PR 状态和仓库同步测试。

```powershell
node --test frontend/tests/pr-status.test.cjs frontend/tests/repo-sync.test.cjs
```
