from __future__ import annotations


class ArenaAPIError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)


class TaskLookupError(ArenaAPIError):
    def __init__(self, task_id: str, expired: bool = False):
        code = "TASK_EXPIRED" if expired else "TASK_NOT_FOUND"
        message = (
            f"Task has expired: {task_id}" if expired else f"Task not found: {task_id}"
        )
        super().__init__(status_code=410, code=code, message=message)
