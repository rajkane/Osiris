import psutil


class MemoryManager:
    """Provides utilities for monitoring and managing memory usage."""

    @staticmethod
    def get_memory_usage_mb():
        process = psutil.Process()
        mem_bytes = process.memory_info().rss
        return mem_bytes / (1024 * 1024)

    @staticmethod
    def check_memory_limit(limit_mb: float) -> bool:
        """Returns True if current memory usage is below the limit."""
        return MemoryManager.get_memory_usage_mb() < limit_mb
