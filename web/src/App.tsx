import { useEffect, useMemo, useState } from "react";
import * as Progress from "@radix-ui/react-progress";
import { diffWords } from "diff";
import {
  Copy,
  Download,
  LoaderCircle,
  RefreshCcw,
  Upload,
} from "lucide-react";

import { createOCRJob, getModelsStatus, getTask, startModelDownload } from "./api";
import type {
  ArenaModelResult,
  ArenaResult,
  ModelStatus,
  ModelsStatusResponse,
} from "./types";

type UIState = "idle" | "image_ready" | "processing" | "success" | "error";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_EXT = ["jpg", "jpeg", "png", "webp", "bmp", "gif"];

const EMPTY_RESULTS: Record<"paddle" | "glm", ArenaModelResult> = {
  paddle: {
    name: "PaddleOCR-VL-1.5",
    latency_ms: null,
    confidence_avg: null,
    text: "",
    boxes: [],
    error: null,
  },
  glm: {
    name: "GLM-OCR",
    latency_ms: null,
    confidence_avg: null,
    text: "",
    boxes: [],
    error: null,
  },
};

const EMPTY_STATUS: Record<"paddle" | "glm", ModelStatus> = {
  paddle: { name: "PaddleOCR-VL-1.5", status: "not_ready", progress: 0 },
  glm: { name: "GLM-OCR", status: "not_ready", progress: 0 },
};

type ImageSize = {
  width: number | null;
  height: number | null;
};

function getFileExtension(filename: string): string {
  const chunks = filename.toLowerCase().split(".");
  return chunks.length > 1 ? chunks[chunks.length - 1] : "";
}

function ModelCard({
  model,
  preview,
  showBoxes,
  imageSize,
}: {
  model: ArenaModelResult;
  preview: string | null;
  showBoxes: boolean;
  imageSize: ImageSize;
}) {
  const [copied, setCopied] = useState(false);
  const confidenceText =
    model.confidence_avg == null ? "--" : model.confidence_avg.toFixed(3);
  const latencyText =
    model.latency_ms == null ? "--" : `${(model.latency_ms / 1000).toFixed(3)} 秒`;
  const hasImageRatio =
    typeof imageSize.width === "number" &&
    imageSize.width > 0 &&
    typeof imageSize.height === "number" &&
    imageSize.height > 0;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="font-semibold text-slate-800">{model.name}</div>
        <div className="text-xs text-slate-500">
          耗时 {latencyText} · 置信度 {confidenceText}
        </div>
      </div>

      <div className="space-y-4 p-4">
        <div>
          <div className="mb-2 text-xs font-semibold text-slate-500">内容定位</div>
          <div
            className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100"
            style={{ aspectRatio: hasImageRatio ? `${imageSize.width} / ${imageSize.height}` : "16 / 9" }}
          >
            {preview ? (
              <img
                src={preview}
                alt="preview"
                className={`h-full w-full object-contain ${showBoxes ? "opacity-50" : "opacity-100"}`}
              />
            ) : null}
            {preview && showBoxes
              ? model.boxes.map((box) => {
                  const [x1, y1, x2, y2] = box.bbox_2d;
                  return (
                    <div
                      key={`${model.name}-${box.index}`}
                      className="absolute border-2 border-blue-500 bg-blue-500/10"
                      style={{
                        left: `${x1 / 10}%`,
                        top: `${y1 / 10}%`,
                        width: `${(x2 - x1) / 10}%`,
                        height: `${(y2 - y1) / 10}%`,
                      }}
                    />
                  );
                })
              : null}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-500">
            <span>识别文本</span>
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded p-1 text-slate-500 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={!model.text}
              onClick={async () => {
                if (!model.text) {
                  return;
                }
                await navigator.clipboard.writeText(model.text);
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1200);
              }}
            >
              {copied ? <span className="text-[11px] text-emerald-600">已复制</span> : null}
              <Copy size={14} />
            </button>
          </div>
          <div className="h-40 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs whitespace-pre-wrap">
            {model.error ? `错误: ${model.error}` : model.text || "暂无结果"}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [uiState, setUiState] = useState<UIState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const [models, setModels] = useState<Record<"paddle" | "glm", ModelStatus>>(EMPTY_STATUS);
  const [results, setResults] = useState<Record<"paddle" | "glm", ArenaModelResult>>(EMPTY_RESULTS);
  const [imageSize, setImageSize] = useState<ImageSize>({ width: null, height: null });
  const [layoutPreviewUrl, setLayoutPreviewUrl] = useState<string | null>(null);
  const [ocrTaskId, setOCRTaskId] = useState<string | null>(null);
  const [downloadTaskIds, setDownloadTaskIds] = useState<
    Partial<Record<"paddle" | "glm", string>>
  >({});

  const textDiff = useMemo(() => {
    const left = results.paddle.text || "";
    const right = results.glm.text || "";
    return diffWords(left, right);
  }, [results.glm.text, results.paddle.text]);

  useEffect(() => {
    getModelsStatus()
      .then((payload: ModelsStatusResponse) => setModels(payload.models))
      .catch((err: Error) => setErrorMessage(err.message));
  }, []);

  useEffect(() => {
    if (!ocrTaskId && !downloadTaskIds.paddle && !downloadTaskIds.glm) {
      return;
    }
    const timer = window.setInterval(async () => {
      if (ocrTaskId) {
        try {
          const task = await getTask(ocrTaskId);
          if (task.status === "failed") {
            setUiState("error");
            setErrorMessage(task.error?.message ?? task.message ?? "OCR 失败");
            setOCRTaskId(null);
          }
          if (task.status === "succeeded" && task.result) {
            const payload = task.result as ArenaResult;
            setResults({ paddle: payload.paddle, glm: payload.glm });
            setImageSize({
              width: payload.input?.width ?? null,
              height: payload.input?.height ?? null,
            });
            setLayoutPreviewUrl(`/api/tasks/${task.task_id}/layout?page=0`);
            setUiState("success");
            setOCRTaskId(null);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "轮询失败";
          setUiState("error");
          setErrorMessage(message);
          setOCRTaskId(null);
        }
      }

      for (const modelKey of ["paddle", "glm"] as const) {
        const taskId = downloadTaskIds[modelKey];
        if (!taskId) {
          continue;
        }
        try {
          const task = await getTask(taskId);
          if (task.status === "succeeded" || task.status === "failed") {
            const statusPayload = await getModelsStatus();
            setModels(statusPayload.models);
            setDownloadTaskIds((prev) => ({ ...prev, [modelKey]: undefined }));
          } else {
            setModels((prev) => ({
              ...prev,
              [modelKey]: {
                ...prev[modelKey],
                status: "downloading",
                progress: task.progress,
                message: task.message,
              },
            }));
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "下载任务查询失败";
          setErrorMessage(message);
          setDownloadTaskIds((prev) => ({ ...prev, [modelKey]: undefined }));
        }
      }
    }, 1500);

    return () => window.clearInterval(timer);
  }, [downloadTaskIds.glm, downloadTaskIds.paddle, ocrTaskId]);

  function onFileSelected(nextFile: File) {
    const ext = getFileExtension(nextFile.name);
    if (!ACCEPTED_EXT.includes(ext)) {
      setErrorMessage(`不支持文件类型: ${ext}`);
      setUiState("error");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage("文件超过 10MB 限制");
      setUiState("error");
      return;
    }

    const objectUrl = URL.createObjectURL(nextFile);
    setFile(nextFile);
    setPreview(objectUrl);
    setLayoutPreviewUrl(null);
    setImageSize({ width: null, height: null });
    setErrorMessage("");
    setResults(EMPTY_RESULTS);
    setUiState("image_ready");
  }

  async function startCompare() {
    if (!file || uiState === "processing") {
      return;
    }
    try {
      setUiState("processing");
      setErrorMessage("");
      setLayoutPreviewUrl(null);
      const { task_id } = await createOCRJob(file);
      setOCRTaskId(task_id);
    } catch (err) {
      const message = err instanceof Error ? err.message : "创建任务失败";
      setUiState("error");
      setErrorMessage(message);
    }
  }

  async function triggerDownload(modelKey: "paddle" | "glm") {
    try {
      const { task_id } = await startModelDownload(modelKey);
      setDownloadTaskIds((prev) => ({ ...prev, [modelKey]: task_id }));
      setModels((prev) => ({
        ...prev,
        [modelKey]: { ...prev[modelKey], status: "downloading", progress: 0 },
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "下载触发失败";
      setErrorMessage(message);
    }
  }

  return (
    <div className="mx-auto min-h-screen max-w-6xl space-y-8 px-4 py-6">
      <header className="space-y-4">
        <h1 className="text-3xl font-black text-slate-800">OCR Arena</h1>

        <div className="grid gap-4 lg:grid-cols-3 lg:auto-rows-fr">
          <label className="group relative flex min-h-[240px] cursor-pointer items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed border-slate-300 bg-white lg:h-full">
            <input
              type="file"
              className="absolute inset-0 z-10 cursor-pointer opacity-0"
              accept="image/*"
              onChange={(event) => {
                const nextFile = event.target.files?.[0];
                if (nextFile) {
                  onFileSelected(nextFile);
                }
              }}
            />
            {preview ? (
              <img src={preview} alt="preview" className="h-full w-full object-contain p-2" />
            ) : (
              <div className="text-center">
                <Upload className="mx-auto mb-2 text-slate-500" />
                <p className="text-sm font-semibold text-slate-700">上传图片</p>
              </div>
            )}
          </label>

          <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4 lg:col-span-2">
            <div className="flex items-center justify-between">
              <div className="font-semibold text-slate-700">模型</div>
              <button
                type="button"
                onClick={startCompare}
                disabled={uiState === "processing" || !file}
                className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                {uiState === "processing" ? <LoaderCircle className="animate-spin" size={14} /> : <RefreshCcw size={14} />}
                开始对比
              </button>
            </div>

            {(["paddle", "glm"] as const).map((key) => {
              const status = models[key];
              const downloading = status.status === "downloading";
              return (
                <div key={key} className="rounded-xl border border-slate-200 p-3">
                  <div className="mb-2 grid grid-cols-[1fr_auto] items-center gap-3 text-sm">
                    <span className="font-medium">{status.name}</span>
                    <span className="w-14 text-center text-xs text-slate-500">{status.status}</span>
                  </div>
                  <div className="grid grid-cols-[1fr_auto] items-center gap-3">
                    <Progress.Root className="relative h-2 w-full overflow-hidden rounded-full bg-slate-200">
                      <Progress.Indicator
                        className="h-full bg-blue-600 transition-all"
                        style={{ width: `${status.progress}%` }}
                      />
                    </Progress.Root>
                    <button
                      type="button"
                      disabled={downloading}
                      onClick={() => triggerDownload(key)}
                      className="inline-flex w-14 shrink-0 items-center justify-center gap-1 whitespace-nowrap rounded border border-slate-300 px-2 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      <Download size={12} /> <span>下载</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {errorMessage ? <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{errorMessage}</div> : null}
      </header>

      {uiState === "processing" ? (
        <div className="rounded-2xl border border-slate-200 bg-white py-24 text-center">
          <LoaderCircle className="mx-auto animate-spin text-blue-600" />
          <p className="mt-3 text-sm text-slate-600">处理中...</p>
        </div>
      ) : (
        <section className="grid gap-4 md:grid-cols-2">
          <ModelCard
            model={results.paddle}
            preview={layoutPreviewUrl ?? preview}
            showBoxes={!layoutPreviewUrl}
            imageSize={imageSize}
          />
          <ModelCard
            model={results.glm}
            preview={layoutPreviewUrl ?? preview}
            showBoxes={!layoutPreviewUrl}
            imageSize={imageSize}
          />
        </section>
      )}

      <section className="rounded-2xl border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700">文本差异 (Paddle vs GLM)</h2>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs leading-6">
          {textDiff.map((part, idx) => {
            if (part.added) {
              return (
                <mark key={idx} className="bg-emerald-100 text-emerald-800">
                  {part.value}
                </mark>
              );
            }
            if (part.removed) {
              return (
                <mark key={idx} className="bg-rose-100 text-rose-800 line-through">
                  {part.value}
                </mark>
              );
            }
            return <span key={idx}>{part.value}</span>;
          })}
        </div>
      </section>
    </div>
  );
}
