export type TaskStatus = "queued" | "running" | "succeeded" | "failed" | "expired";

export type ArenaBox = {
  index: number;
  label: string;
  score: number | null;
  bbox_2d: [number, number, number, number];
  polygon?: number[][] | null;
};

export type ArenaModelResult = {
  name: "PaddleOCR-VL-1.5" | "GLM-OCR" | string;
  latency_ms: number | null;
  confidence_avg: number | null;
  text: string;
  boxes: ArenaBox[];
  error?: string | null;
};

export type ArenaResult = {
  request_id: string;
  input: {
    filename: string;
    width: number | null;
    height: number | null;
  };
  paddle: ArenaModelResult;
  glm: ArenaModelResult;
  manifest?: string | null;
};

export type ModelStatus = {
  name: string;
  status: "not_ready" | "downloading" | "ready" | "error";
  progress: number;
  message?: string | null;
};

export type ModelsStatusResponse = {
  models: Record<"paddle" | "glm", ModelStatus>;
};

export type TaskSnapshot = {
  task_id: string;
  task_type: "ocr" | "download";
  status: TaskStatus;
  progress: number;
  stage?: string | null;
  message?: string | null;
  model_key?: "paddle" | "glm";
  result?: ArenaResult;
  error?: {
    code: string;
    message: string;
    request_id: string;
  } | null;
};

export type ApiError = {
  code: string;
  message: string;
  request_id: string;
};

