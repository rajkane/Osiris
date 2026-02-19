from dataclasses import dataclass


@dataclass
class DefaultConfig:
    """Application default configuration."""

    default_method: str = "average"
    align_by_default: bool = False
    memory_limit_mb: float = 1024.0  # safety default
    logging_level: str = "INFO"
