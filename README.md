# OCR Arena

本项目提供本地双模型 OCR 对比能力（`PaddleOCR-VL-1.5` vs `GLM-OCR`），并支持 Web 界面上传图片、异步任务轮询、模型下载管理与 Docker 一键启动。

## 功能概览

- 本地 OCR Pipeline 固定链路：`PageLoader -> LayoutDetector -> OCR Backend -> ResultFormatter`
- Web API（FastAPI）：
  - `GET /api/models/status`
  - `POST /api/models/{model_key}/download`
  - `POST /api/arena/jobs`
  - `POST /api/arena/ocr`
  - `GET /api/tasks/{task_id}`
- Web 前端（React + Tailwind）：
  - 上传校验（格式 + 10MB）
  - 双栏 Overlay 与文本展示
  - 文本复制 + `jsdiff` 差异高亮
  - 下载进度与 OCR 任务轮询

## 本地开发

### 1) 安装依赖

```bash
uv sync
```

### 2) 启动后端 API

```bash
uv run ocr-arena-web
```

默认地址：`http://127.0.0.1:8000`

可通过环境变量指定配置：

```bash
export OCR_ARENA_CONFIG=./ocr_arena/config.yaml
```

### 3) 启动前端

```bash
cd web
npm install
npm run dev
```

默认地址：`http://localhost:5173`（`/api` 自动代理到 `http://localhost:8000`）。

## Docker 一键启动

```bash
docker compose up --build
```

- 单服务入口（前端 + API）：`http://localhost:8000`

`docker-compose.yml` 已将端口限制为 `127.0.0.1:8000:8000`，仅本机可访问。

### 模型目录约定

`docker-compose.yml` 已挂载：

- `./models:/app/models`
- `./output:/app/output`

其中 `PP-DocLayout-v3` 已打包进镜像（位于 `/opt/pp-doclayout-v3`，使用 `ocr_arena/config.docker.yaml`）。

## 配置约定

- `OCR_ARENA_CONFIG`：配置文件路径（默认自动发现 `ocr_arena/config.yaml`）
- Web 监听地址：`web.host`（默认 `127.0.0.1`）
- 输出目录：`pipeline.output.base_output_dir`（默认 `./output`）
- 上传上限：`web.upload_max_mb`（默认 `10`）
- 单模型超时：`web.model_timeout_seconds`（默认 `120`）
- 任务保留：`web.task_ttl_seconds`（默认 `1800` 秒）

## 常见问题

1. `MODEL_NOT_READY`：检查 `config.yaml` 中 `model_dir` 与 `required_model_files` 是否齐全。
2. 下载失败：确认 `modelscope` 或 `hf` 命令可用，并检查网络/权限。
3. 任务查询 410：任务可能不存在或已过期（默认保留 30 分钟）。
