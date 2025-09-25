# AIHelper - LangChain + FastAPI 脚手架

本项目提供一个最小可运行的 LangChain 后端（FastAPI）与简单前端（HTML+JS），对接火山引擎方舟 OpenAI 兼容接口（Doubao 模型）。

## 环境
- Python 3.11（建议）
- Poetry（依赖管理）

## 快速开始
1. 复制环境变量模板并填写：
   - 将 `.env.example` 复制为 `.env.local`，填入 `ARK_API_KEY` 等真实配置。
2. 安装依赖（首次）：
   - `poetry install`
3. 启动后端服务：
   - `poetry run uvicorn app.server:app --reload --host 127.0.0.1 --port 8000`
4. 打开前端：
   - 直接用浏览器打开 `frontend/index.html`（或使用任意静态服务器）。

### 一键启动（Windows）
- 复制环境变量模板并填写 `.env.local`
- 在仓库根目录双击 `dev.bat` 或在 CMD 执行：

````bat
cd /d D:\Git\aihelper
.\dev.bat 9020  ^  REM 可选端口参数，默认 9020
````

提示：若你使用 9020 端口，请将 `frontend/main.js` 中的 `API_BASE` 改为 `http://127.0.0.1:9020`，或用 `curl`/`/docs` 直接调试后端。


## API 说明
- `GET /health` 健康检查
- `POST /chat` 文本对话（默认启用 RAG：会在回答前检索本地文档并注入上下文，可通过 RAG_ENABLE_FOR_CHAT 关闭；亦可在请求体用 `use_rag` 临时覆盖）
  - 请求：`{"message": "你好", "memoryid": "user-123", "use_rag": true}`
- `POST /chat-vision` 多模态对话（图片 URL）
  - 请求：`{"prompt": "描述这张图片", "image_urls": ["https://..."], "memoryid": "user-123"}`

- `POST /chat-structured` 结构化输出（基于 JSON Schema）
  - 请求：`{"message": "请从这段文本中提取要点", "schema": {"type":"object","properties":{"title":{"type":"string"},"keywords":{"type":"array","items":{"type":"string"}}},"required":["title","keywords"],"additionalProperties":false}, "memoryid":"user-123"}`
  - 响应：`{"data": {"title": "...", "keywords": ["..."]}}`

- `POST /rag/reindex` 重新索引文档（读取本地 .md、切分、向量化并落盘）
- `POST /rag/query` 基于向量检索 + 上下文回答（TopK=5，过滤分数<0.75）


> 备注：会话记忆使用 LangChain InMemoryChatMessageHistory，按 `memoryid` 隔离；未提供则使用 `"default"`。为进程内存储，重启服务将清空。

## 环境变量
- `ARK_API_KEY`：方舟 API Key（敏感）
- `ARK_BASE_URL`：默认 `https://ark.cn-beijing.volces.com/api/v3`
- `ARK_MODEL`：默认 `doubao-seed-1-6-vision-250815`
- `SYSTEM_PROMPT`：系统提示词（可选）。设置后会作为 system 角色注入到每次对话开头。
- `RAG_DOCS_DIR`：本地文档目录（默认 `local_docs`）
- `RAG_STORE_PATH`：向量库保存路径（默认 `local_docs/index.json`）
- `ARK_EMBEDDING_MODEL`：向量模型（默认 `doubao-embedding-vision-250615`）
- `RAG_CHUNK_SIZE`：切片最大长度（默认 1000）
- `RAG_CHUNK_OVERLAP`：切片重叠（默认 200）
- `RAG_TOP_K`：检索条数（默认 5）
- `RAG_MIN_SCORE`：最小相似度阈值（默认 0.75）
- `RAG_ENABLE_FOR_CHAT`：是否在普通 `/chat` 对话中启用 RAG（默认 `true`）
- `GET /rag/status`：返回 RAG 可用性与开关状态（has_api_key、healthy、enabled_for_chat、embed_model）

- `CORS_ORIGINS`：默认允许 `http://localhost:5173,http://127.0.0.1:5173`

> 注意：`.env.local` 已加入 `.gitignore`，请勿提交到仓库。

