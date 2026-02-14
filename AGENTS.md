# OCR Arena - AI 协作开发规范

本文件作用域：仓库根目录及其所有子目录。

## 项目目标与边界

- 本项目用于本地 OCR 能力对比，当前核心是 **GLM-OCR** 与 **PaddleOCR-VL** 两类后端。
- 处理对象为本地文件（单张图片或 PDF）；不在主流程中引入远程 OCR API。
- 主流程固定为：`PageLoader -> LayoutDetector -> OCR Backend -> ResultFormatter`。
- `OCRPipeline` 当前仅支持 `enable_layout=true`，开发时请保持该约束一致。

## 关键代码结构（改动前先定位）

- `ocr_arena/ocr_pipeline.py`：主编排逻辑、并发识别、生命周期管理。
- `ocr_arena/backend/`：OCR 后端抽象与实现（`BaseBackend`、GLM、Paddle）。
- `ocr_arena/layout/`：版面检测抽象与实现（`BaseLayoutDetector`、PPDocLayout）。
- `ocr_arena/postprocess/`：结果格式化与后处理。
- `ocr_arena/parser_result/`：结构化结果对象与落盘行为。
- `ocr_arena/config.py` + `ocr_arena/config.yaml`：配置模型与默认配置，必须保持同步。
- `ocr_arena/cli/`：本地命令行入口（`ocr-arena`、`ocr-arena-layout`）。

## 环境与常用命令

- Python 版本：`>=3.12`。
- 推荐使用 `uv`：
  - 安装依赖：`uv sync`
  - 查看命令：`uv run ocr-arena --help`
  - 图片/PDF 快速运行：`uv run ocr-arena <input_path> --output ./output --stdout`
  - 仅版面检测入口：`uv run ocr-arena-layout <input_path> --output ./output`
- 代码质量检查（提交前至少执行其一）：
  - `uv run pre-commit run -a`
  - 或 `uv run black .` + `uv run flake8 ocr_arena`

## 代码风格与实现约束

- 保持“小改动、强兼容”：优先修根因，不做无关重构。
- 为新增/修改函数补充明确类型标注；除必要兼容层外避免扩大 `Any` 使用。
- 日志统一使用 `ocr_arena.utils.logging` 中的 `get_logger/get_profiler`。
- 错误处理需显式：
  - 输入错误用 `ValueError/TypeError/FileNotFoundError` 等明确异常。
  - 推理失败应记录可定位日志，避免静默吞错。
- 禁止硬编码机器相关绝对路径、密钥、token。
- 不提交模型权重、缓存、输出产物（如 `output/`、临时可视化目录）。

## 扩展规范（新增能力时必须遵守）

### 新增 OCR 后端

1. 在 `ocr_arena/backend/` 新建实现类并继承 `BaseBackend`。
2. 实现并验证 `start()` / `stop()` / `process()` 生命周期与线程安全。
3. 在 `config.py` 增加对应配置模型，并在 `config.yaml` 补默认项。
4. 在 `ocr_pipeline.py` 统一接入，保持现有后端选择逻辑兼容。

### 新增 Layout 检测器

1. 继承 `BaseLayoutDetector`，保持 `process(images)->List[List[Dict]]` 契约。
2. 输出字段需与后续裁剪/识别链路兼容（`bbox_2d`、`task_type` 等）。
3. 如引入可选依赖，沿用 `layout/__init__.py` 的延迟导入与错误包装模式。

### 修改配置项

- 同时更新：
  - `ocr_arena/config.py`（Pydantic 模型）
  - `ocr_arena/config.yaml`（默认值）
  - 对应 CLI 参数/文档（若行为有外显变化）

## AI Agent 工作流（建议）

1. 先读本文件，再读目标模块与其直接调用链。
2. 先给出最小可行改动方案，再执行实现。
3. 优先跑与改动最相关的最小验证；再视情况扩大验证范围。
4. 回复中需说明：改了什么、为何这样改、如何验证、已知限制。

## 验收清单（交付前自检）

- [ ] 变更范围聚焦在需求本身，无无关修改。
- [ ] 配置模型与默认配置保持一致。
- [ ] CLI 与核心流程在至少一个示例输入上可执行。
- [ ] 日志与异常信息可定位问题。
- [ ] 通过格式化/静态检查（至少一套）。


## Subagent 协作策略（重点）

- 按任务依赖关系与相关性灵活决定是否启用 subagent：
  - 强依赖串行任务（上游输出未稳定）优先单 agent 顺序推进。
  - 弱依赖或可解耦任务（模块边界清晰）可拆分给多个 subagent 并行。
- 当前项目在“接口契约明确”后可前后端并行开发：
  - 以后端 schema / API contract（路径、字段、状态码）冻结为并行起点。
  - 并行期间若接口字段变更，必须先更新契约文档，再同步前后端任务。
- 推荐并行拆分方式：
  - subagent-A：后端 API/服务编排/模型管理。
  - subagent-B：前端页面、状态机、接口接线、可视化渲染。
  - 主 agent：负责契约守护、冲突协调与最终集成。
- 测试任务必须在独立 subagent 中执行，避免实现上下文干扰：
  - 开发完成后，将 QA/验收任务切换到专用 testing subagent。
  - testing subagent 仅关注验证与缺陷回传，不承担功能实现。
  - 至少覆盖：接口契约、核心流程联调、异常分支、性能基线、发布前冒烟。
- 若并行开发与测试结论冲突，以测试 subagent 的复现结果为准回流修复。

## 测试执行原则（重点）

- 测试输入优先使用仓库样例：`examples/source/` 下图片与 PDF 文件可直接用于实测。
- 默认假设本地模型权重已就绪：GLM / Paddle / PP-DocLayout-v3 已放置在各自配置目录。
  - 测试阶段优先复用现有本地权重，不重复下载。
  - 仅当权重缺失或校验失败时，才进入下载/修复流程。
- 测试若出现错误，不仅要报告，还要尽量完成修复闭环：
  - 先定位根因并在需求范围内修复，避免只做表面绕过。
  - 修复后必须执行最小复测，并尽量补充一条防回归验证。
  - 若错误属于当前需求范围外，需明确记录现象、影响与建议处理。
