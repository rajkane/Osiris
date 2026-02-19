from loguru import logger


class LogManager:
    """Manages application-wide logging configuration and loggers using loguru.

    Uses tqdm.write when tqdm is available so that log messages don't break
    progress bars. Keeps a single configured logger (singleton-like behavior).
    """

    _configured = False

    @classmethod
    def get_logger(cls):
        if not cls._configured:
            # Configure loguru logger only once
            logger.remove()
            try:
                from tqdm import tqdm

                def _sink(msg):
                    tqdm.write(msg, end="")

                sink = _sink
            except Exception:
                # Fallback to plain print
                def _sink(msg):
                    print(msg, end="")

                sink = _sink

            logger.add(
                sink,
                format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
                level="INFO",
            )
            cls._configured = True
        return logger

    @classmethod
    def set_level(cls, level):
        # Reset configuration with new level
        logger.remove()
        try:
            from tqdm import tqdm

            def _sink(msg):
                tqdm.write(msg, end="")

            sink = _sink
        except Exception:
            def _sink(msg):
                print(msg, end="")

            sink = _sink

        logger.add(
            sink,
            format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
            level=level,
        )
        cls._configured = True
