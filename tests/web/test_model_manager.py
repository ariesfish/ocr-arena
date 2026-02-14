from __future__ import annotations

from ocr_arena import load_config
from ocr_arena.web.services.model_manager import ModelManager
from ocr_arena.web.stores.task_store import TaskStore


def test_build_download_command_modelscope() -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    command = manager.build_download_command("glm", "modelscope")
    assert command[0] == "modelscope"
    assert "--model" in command
    assert "--local_dir" in command


def test_build_download_command_huggingface() -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    command = manager.build_download_command("paddle", "huggingface")
    assert command[0] == "hf"
    assert "--local-dir" in command
