from .logging import LogManager
from .memory import MemoryManager
from .error import ErrorManager

# export singletons/instances for convenient use
memory_manager = MemoryManager()
error_manager = ErrorManager()

__all__ = [
    "LogManager",
    "MemoryManager",
    "ErrorManager",
    "memory_manager",
    "error_manager",
]
