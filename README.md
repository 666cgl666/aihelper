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

## API 说明
- `GET /health` 健康检查
- `POST /chat` 文本对话
  - 请求：`{"message": "你好"}`
- `POST /chat-vision` 多模态对话（图片 URL）
  - 请求：`{"prompt": "描述这张图片", "image_urls": ["https://..."]}`

## 环境变量
- `ARK_API_KEY`：方舟 API Key（敏感）
- `ARK_BASE_URL`：默认 `https://ark.cn-beijing.volces.com/api/v3`
- `ARK_MODEL`：默认 `doubao-seed-1-6-vision-250815`
- `CORS_ORIGINS`：默认允许 `http://localhost:5173,http://127.0.0.1:5173`

> 注意：`.env.local` 已加入 `.gitignore`，请勿提交到仓库。

