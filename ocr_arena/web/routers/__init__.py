from .arena import router as arena_router
from .health import router as health_router
from .models import router as models_router
from .tasks import router as tasks_router

__all__ = [
    "arena_router",
    "health_router",
    "models_router",
    "tasks_router",
]
