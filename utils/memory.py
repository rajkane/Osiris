import psutil


class _SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MemoryManager(metaclass=_SingletonMeta):
    """Provides utilities for monitoring and managing memory usage.

    Implemented as a singleton in case we want to hold configuration/state later.
    """

    def _get_memory_usage_mb(self) -> float:
        process = psutil.Process()
        mem_bytes = process.memory_info().rss
        return mem_bytes / (1024 * 1024)

    def _check_memory_limit(self, limit_mb: float) -> bool:
        """Instance method: Returns True if current memory usage is below the limit."""
        return self._get_memory_usage_mb() < limit_mb

    # Backwards-compatible classmethod wrappers
    @classmethod
    def get_memory_usage_mb(cls) -> float:
        return cls()._get_memory_usage_mb()

    @classmethod
    def check_memory_limit(cls, limit_mb: float) -> bool:
        return cls()._check_memory_limit(limit_mb)
